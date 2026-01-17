#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <stdexcept>

#include "csv.h"
#include "raw16.h"
#include "metrics.h"

// ★あなたの DLL ヘッダ
#include "ImageProcApi.h"

namespace fs = std::filesystem;

static bool metric_pass(const std::string& metric, double value, double pass_value) {
    if (metric == "exact")     return value == 0.0;
    if (metric == "max_abs")   return value <= pass_value;
    if (metric == "mae")       return value <= pass_value;
    if (metric == "psnr")      return value >= pass_value;
    return false;
}

static MetricResult compute_metric(const std::string& metric,
    const std::vector<uint16_t>& out,
    const std::vector<uint16_t>& gold)
{
    if (metric == "exact")   return metric_exact(out, gold);
    if (metric == "max_abs") return metric_max_abs(out, gold);
    if (metric == "mae")     return metric_mae(out, gold);
    if (metric == "psnr")    return metric_psnr(out, gold);
    throw std::runtime_error("Unknown metric: " + metric);
}

static IPC_Params make_params_from_csv(const CsvRow& r, int w, int h)
{
    IPC_Params p{};
    p.width = w;
    p.height = h;
    p.sizeBytes = (uint32_t)sizeof(IPC_Params);
    p.version = 1;

    p.window = r.i("window");
    p.level = r.i("level");
    p.enableEdge = r.i("enable_edge");

    // reserved[0] = enablePostFilter (とコメントにあるが、あなたの実装では
    // reserved[0..3] を blur/invert/threshold に使っている前提で合わせる
    p.reserved[0] = r.i("enable_blur");
    p.reserved[1] = r.i("enable_invert");
    p.reserved[2] = r.i("enable_threshold");
    p.reserved[3] = r.i("threshold_value");
    return p;
}

static void safe_shutdown_and_release(void*& ioBuf)
{
    // DLL内部がioBufを参照している可能性があるので、先にShutdown
    IPC_Shutdown();

    if (ioBuf) {
        IPC_ReleaseD3D11Resource(ioBuf); // COM Release
        ioBuf = nullptr;
    }
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cout << "Usage: ImageProcTestRunner <tests/test_cases.csv> [--gpu 0] [--outdir tests/out] [--no-save-out]\n";
        return 2;
    }

    std::string csvPath = argv[1];
    int gpuId = 0;
    fs::path outdir = "tests/out";
    bool saveOut = true;
    bool generateGolden = false;

    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--gpu" && i + 1 < argc) gpuId = std::stoi(argv[++i]);
        else if (a == "--outdir" && i + 1 < argc) outdir = argv[++i];
        else if (a == "--no-save-out") saveOut = false;
        else if (a == "--generate-golden") generateGolden = true;
    }

    fs::create_directories(outdir);

    auto rows = read_csv(csvPath);

    fs::path resultCsv = outdir / "results.csv";
    std::ofstream ofs(resultCsv.string());
    ofs << "name,metric,value,pass_value,status,input,golden,out,detail\n";

    int nPass = 0, nFail = 0;

    for (const auto& r : rows)
    {
        const std::string name = r.at("name");
        const std::string inPath = r.at("input");
        const std::string goldPath = r.at("golden");
        const int w = r.i("width");
        const int h = r.i("height");
        const std::string metric = r.at("metric");
        const double pass_value = std::stod(r.at("pass_value"));

        fs::path outPath = outdir / (name + ".raw");

        void* ioBuf = nullptr;

        try {
            auto in = load_raw16(inPath, w, h);
            std::vector<uint16_t> gold;
            if (!generateGolden) {
                gold = load_raw16(goldPath, w, h);
            }
            // 1) IO buffer作成（ID3D11Buffer* を void* として受け取る）
            ioBuf = IPC_CreateIoBuffer(gpuId, w, h);
            if (!ioBuf) {
                int32_t hr = IPC_GetLastHr();
                const char* msg = IPC_GetLastErr();
                throw std::runtime_error(std::string("IPC_CreateIoBuffer failed. hr=0x")
                    + [](int32_t v) { char b[16]; sprintf_s(b, "%08X", (unsigned)v); return std::string(b); } (hr)
                    +" " + (msg ? msg : ""));
            }

            // 2) Init（buffer版）
            int32_t rc = IPC_InitWithIoBuffer(gpuId, ioBuf);
            if (rc != IPC_OK) throw std::runtime_error("IPC_InitWithIoBuffer failed: " + std::to_string(rc));

            // 3) Params
            IPC_Params p = make_params_from_csv(r, w, h);
            rc = IPC_SetParams(&p);
            if (rc != IPC_OK) throw std::runtime_error("IPC_SetParams failed: " + std::to_string(rc));

            // 4) Upload（buffer版）
            rc = IPC_UploadRaw16ToBuffer(in.data(), (int32_t)(in.size() * sizeof(uint16_t)), w, h);
            if (rc != IPC_OK) throw std::runtime_error("IPC_UploadRaw16ToBuffer failed: " + std::to_string(rc));

            // 5) Execute
            rc = IPC_Execute();
            if (rc != IPC_OK) throw std::runtime_error("IPC_Execute failed: " + std::to_string(rc));

            // 6) Readback（buffer版）
            std::vector<uint16_t> out((size_t)w * (size_t)h);
            rc = IPC_ReadbackRaw16FromBuffer(out.data(), (int32_t)(out.size() * sizeof(uint16_t)));
            if (rc != IPC_OK) throw std::runtime_error("IPC_ReadbackRaw16FromBuffer failed: " + std::to_string(rc));

            // 7) shutdown & release
            safe_shutdown_and_release(ioBuf);

            if (!generateGolden) {
                // 保存
                if (saveOut) save_raw16(outPath.string(), out);

                // 評価
                auto mr = compute_metric(metric, out, gold);
                bool ok = metric_pass(metric, mr.value, pass_value);


                //ofs << "name,metric,value,pass_value,status,input,golden,out,detail\n";
                ofs << name << "," << metric << "," << mr.value << "," << pass_value << ","
                    << (ok ? "PASS" : "FAIL") << ","
                    << inPath << "," << goldPath << "," << (saveOut ? outPath.string() : "") << ",OK\n";

                if (ok) nPass++; else nFail++;

                std::cout << "[" << (ok ? "PASS" : "FAIL") << "] " << name
                    << " metric=" << metric << " value=" << mr.value << "\n";
            }
            else
            {
                fs::path gpath(goldPath);
                if (!gpath.parent_path().empty()) {
                    fs::create_directories(gpath.parent_path());
                }
                save_raw16(gpath.string(), out);

                //ofs << "name,metric,value,pass_value,status,input,golden,out,detail\n";
                ofs << name << "," << metric << "," << "" << "," << pass_value << ","
                    << "GENERATED" << ","
                    << inPath << "," << goldPath << "," << (saveOut ? outPath.string() : "") << ",Wrote golden\n";
            }

        }
        catch (const std::exception& e)
        {
            safe_shutdown_and_release(ioBuf);

            ofs << name << "," << metric << ",nan," << pass_value << ",ERROR,"
                << inPath << "," << goldPath << "," << (saveOut ? outPath.string() : "") << ","
                << "\"" << e.what() << "\"\n";

            nFail++;
            std::cout << "[ERROR] " << name << " : " << e.what() << "\n";
        }
    }

    std::cout << "Done. PASS=" << nPass << " FAIL=" << nFail << "\n";
    std::cout << "Results: " << resultCsv.string() << "\n";
    return (nFail == 0) ? 0 : 1;
}
