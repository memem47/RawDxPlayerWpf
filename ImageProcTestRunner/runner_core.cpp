#include "runner_core.h"

#include <fstream>
#include <stdexcept>
#include <cctype>

#include "raw16.h"
#include "metrics.h"

// DLL header
#include "ImageProcApi.h"

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

    // reserved[0..3] は blur/invert/threshold に使用（既存 runner の挙動を維持）
    p.reserved[0] = r.i("enable_blur");
    p.reserved[1] = r.i("enable_invert");
    p.reserved[2] = r.i("enable_threshold");
    p.reserved[3] = r.i("threshold_value");
    return p;
}

static void safe_shutdown_and_release(void*& ioBuf)
{
    // DLL内部がioBufを参照している可能性があるため、先にShutdown
    IPC_Shutdown();

    if (ioBuf) {
        IPC_ReleaseD3D11Resource(ioBuf); // COM Release
        ioBuf = nullptr;
    }
}

void ResultsCollector::Add(const CaseResult& r)
{
    std::lock_guard<std::mutex> lk(m_);
    results_.push_back(r);
}

static void csv_escape_and_write(std::ostream& os, const std::string& s)
{
    // results.csv は基本カンマ区切りだが detail は例外的に quote する
    // 既存 runner の出力を壊さないため、必要時のみ quote
    bool need = false;
    for (char c : s) {
        if (c == '"' || c == ',' || c == '\n' || c == '\r') { need = true; break; }
    }
    if (!need) { os << s; return; }

    os << '"';
    for (char c : s) {
        if (c == '"') os << "\"\"";
        else os << c;
    }
    os << '"';
}

void ResultsCollector::WriteCsv(const fs::path& outdir) const
{
    fs::create_directories(outdir);
    fs::path resultCsv = outdir / "results.csv";

    std::ofstream ofs(resultCsv.string(), std::ios::out | std::ios::trunc);
    ofs << "name,metric,value,pass_value,status,input,golden,out,detail\n";

    std::lock_guard<std::mutex> lk(m_);
    for (const auto& r : results_) {
        ofs << r.name << ",";
        ofs << r.metric << ",";
        if (r.status == CaseStatus::GENERATED) ofs << ""; // 既存 runner と同様、生成時は値空欄
        else if (r.status == CaseStatus::ERROR) ofs << "nan";
        else ofs << r.value;
        ofs << ",";
        ofs << r.pass_value << ",";

        switch (r.status) {
        case CaseStatus::PASS:      ofs << "PASS"; break;
        case CaseStatus::FAIL:      ofs << "FAIL"; break;
        case CaseStatus::GENERATED: ofs << "GENERATED"; break;
        case CaseStatus::ERROR:     ofs << "ERROR"; break;
        }
        ofs << ",";
        ofs << r.input << ",";
        ofs << r.golden << ",";
        ofs << r.out << ",";
        csv_escape_and_write(ofs, r.detail);
        ofs << "\n";
    }
}

int ResultsCollector::PassCount() const
{
    std::lock_guard<std::mutex> lk(m_);
    int c = 0;
    for (auto& r : results_) if (r.status == CaseStatus::PASS) c++;
    return c;
}
int ResultsCollector::FailCount() const
{
    std::lock_guard<std::mutex> lk(m_);
    int c = 0;
    for (auto& r : results_) if (r.status == CaseStatus::FAIL) c++;
    return c;
}
int ResultsCollector::ErrorCount() const
{
    std::lock_guard<std::mutex> lk(m_);
    int c = 0;
    for (auto& r : results_) if (r.status == CaseStatus::ERROR) c++;
    return c;
}
int ResultsCollector::GeneratedCount() const
{
    std::lock_guard<std::mutex> lk(m_);
    int c = 0;
    for (auto& r : results_) if (r.status == CaseStatus::GENERATED) c++;
    return c;
}

CaseResult RunOneCase(const CsvRow& r, const RunnerOptions& opt)
{
    CaseResult cr{};

    const std::string name = r.at("name");
    const std::string inPath = r.at("input");
    const std::string goldPath = r.at("golden");
    const int w = r.i("width");
    const int h = r.i("height");
    const std::string metric = r.at("metric");
    const double pass_value = std::stod(r.at("pass_value"));

    cr.name = name;
    cr.metric = metric;
    cr.pass_value = pass_value;
    cr.input = inPath;
    cr.golden = goldPath;

    fs::create_directories(opt.outdir);
    fs::path outPath = opt.outdir / (name + ".raw");
    if (opt.saveOut) cr.out = outPath.string();
    else cr.out = "";

    void* ioBuf = nullptr;

    try {
        auto in = load_raw16(inPath, w, h);

        std::vector<uint16_t> gold;
        if (!opt.generateGolden) {
            gold = load_raw16(goldPath, w, h);
        }

        // 1) IO buffer 作成（ID3D11Buffer* を void* として受け取る）
        ioBuf = IPC_CreateIoBuffer(opt.gpuId, w, h);
        if (!ioBuf) {
            int32_t hr = IPC_GetLastHr();
            const char* msg = IPC_GetLastErr();
            char b[64];
            sprintf_s(b, "IPC_CreateIoBuffer failed. hr=0x%08X", (unsigned)hr);
            throw std::runtime_error(std::string(b) + " " + (msg ? msg : ""));
        }

        // 2) Init
        int32_t rc = IPC_InitWithIoBuffer(opt.gpuId, ioBuf);
        if (rc != IPC_OK) throw std::runtime_error("IPC_InitWithIoBuffer failed: " + std::to_string(rc));

        // 3) Params
        IPC_Params p = make_params_from_csv(r, w, h);
        rc = IPC_SetParams(&p);
        if (rc != IPC_OK) throw std::runtime_error("IPC_SetParams failed: " + std::to_string(rc));

        // 4) Upload
        rc = IPC_UploadRaw16ToBuffer(in.data(), (int32_t)(in.size() * sizeof(uint16_t)), w, h);
        if (rc != IPC_OK) throw std::runtime_error("IPC_UploadRaw16ToBuffer failed: " + std::to_string(rc));

        // 5) Execute
        rc = IPC_Execute();
        if (rc != IPC_OK) throw std::runtime_error("IPC_Execute failed: " + std::to_string(rc));

        // 6) Readback
        std::vector<uint16_t> out((size_t)w * (size_t)h);
        rc = IPC_ReadbackRaw16FromBuffer(out.data(), (int32_t)(out.size() * sizeof(uint16_t)));
        if (rc != IPC_OK) throw std::runtime_error("IPC_ReadbackRaw16FromBuffer failed: " + std::to_string(rc));

        // 7) shutdown & release
        safe_shutdown_and_release(ioBuf);

        if (opt.generateGolden) {
            fs::path gpath(goldPath);
            if (!gpath.parent_path().empty()) {
                fs::create_directories(gpath.parent_path());
            }
            save_raw16(gpath.string(), out);

            if (opt.saveOut) save_raw16(outPath.string(), out);

            cr.status = CaseStatus::GENERATED;
            cr.detail = "Wrote golden";
            return cr;
        }

        // 保存
        if (opt.saveOut) save_raw16(outPath.string(), out);

        // 評価
        auto mr = compute_metric(metric, out, gold);
        cr.value = mr.value;
        const bool ok = metric_pass(metric, mr.value, pass_value);
        cr.status = ok ? CaseStatus::PASS : CaseStatus::FAIL;
        cr.detail = "OK";
        return cr;
    }
    catch (const std::exception& e)
    {
        safe_shutdown_and_release(ioBuf);
        cr.status = CaseStatus::ERROR;
        cr.detail = e.what();
        return cr;
    }
}

std::string SanitizeGTestName(const std::string& s)
{
    // gtest の test name は基本 [A-Za-z_][A-Za-z0-9_]* が安全
    std::string t;
    t.reserve(s.size());
    for (char c : s) {
        if (std::isalnum((unsigned char)c)) t.push_back(c);
        else t.push_back('_');
    }
    if (t.empty()) t = "Case";
    if (std::isdigit((unsigned char)t[0])) t = std::string("C_") + t;
    return t;
}