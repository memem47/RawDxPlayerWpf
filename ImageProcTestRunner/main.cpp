#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>

#include <gtest/gtest.h>

#include "csv.h"
#include "runner_core.h"

namespace fs = std::filesystem;

// ------------------------------------------------------------
// globals (shared between main() and TEST body)
// ------------------------------------------------------------
static RunnerOptions g_opt;
static ResultsCollector g_results;
static std::vector<CsvRow> g_rows;

// gtest_filter を gtestのAPIに頼らず argv から取得して保持
static std::string g_gtestFilter = "*";

// ------------------------------------------------------------
// results.csv output on teardown
// ------------------------------------------------------------
class ResultsEnv : public ::testing::Environment
{
public:
    explicit ResultsEnv(fs::path outdir) : outdir_(std::move(outdir)) {}

    void TearDown() override
    {
        g_results.WriteCsv(outdir_);
        std::cout << "Results: " << (outdir_ / "results.csv").string() << "\n";
    }

private:
    fs::path outdir_;
};

// ------------------------------------------------------------
// wildcard match (* and ?)
// ------------------------------------------------------------
static bool WildcardMatch(const char* pat, const char* str)
{
    if (!pat || !str) return false;

    while (*pat) {
        if (*pat == '*') {
            ++pat;
            if (!*pat) return true;
            while (*str) {
                if (WildcardMatch(pat, str)) return true;
                ++str;
            }
            return false;
        }
        if (*pat == '?') {
            if (!*str) return false;
            ++pat; ++str;
            continue;
        }
        if (*pat != *str) return false;
        ++pat; ++str;
    }
    return *str == '\0';
}

static bool MatchesFilterExpr(const std::string& filter,
    const std::string& suite,
    const std::string& test)
{
    // gtest filter 互換（最低限）
    // pos1:pos2-neg1:neg2
    const std::string full = suite + "." + test;

    if (filter.empty()) return true;

    std::string pos = filter;
    std::string neg;

    const auto dash = filter.find('-');
    if (dash != std::string::npos) {
        pos = filter.substr(0, dash);
        neg = filter.substr(dash + 1);
    }

    auto anyMatchOr = [&](const std::string& part) -> bool {
        if (part.empty()) return true;
        size_t start = 0;
        while (start <= part.size()) {
            size_t end = part.find(':', start);
            if (end == std::string::npos) end = part.size();
            std::string token = part.substr(start, end - start);
            if (!token.empty()) {
                if (WildcardMatch(token.c_str(), full.c_str())) return true;
            }
            start = end + 1;
        }
        return false;
        };

    const bool posOk = anyMatchOr(pos);
    const bool negHit = !neg.empty() && anyMatchOr(neg);
    return posOk && !negHit;
}

// ------------------------------------------------------------
// argv helpers
// ------------------------------------------------------------
static bool StartsWith(const std::string& s, const std::string& pfx)
{
    return s.rfind(pfx, 0) == 0;
}

static void PrintUsage()
{
    std::cout
        << "Usage:\n"
        << "  ImageProcTestRunner <tests/test_cases.csv>\n"
        << "      [--gpu 0]\n"
        << "      [--outdir tests/out]\n"
        << "      [--no-save-out]\n"
        << "      [--generate-golden]\n"
        << "      [--help]\n"
        << "      (and any GoogleTest flags, e.g. --gtest_filter=..., --gtest_output=xml:...)\n"
        << "\n"
        << "Examples:\n"
        << "  ImageProcTestRunner tests/test_cases.csv --gpu 0 --outdir tests/out\n"
        << "  ImageProcTestRunner tests/test_cases.csv --generate-golden --outdir tests/out --gtest_filter=CsvCases.*Sobel*\n"
        << "\n"
        << "Notes:\n"
        << "  - results.csv is written to <outdir>/results.csv\n"
        << "  - gtest XML is written to <outdir>/gtest.xml by default unless you pass --gtest_output\n";
}

// ------------------------------------------------------------
// Single TEST runs selected CSV rows.
// Filter comes from argv-parsed g_gtestFilter (no gtest API dependence).
// ------------------------------------------------------------
TEST(CsvCases, RunSelected)
{
    ASSERT_FALSE(g_rows.empty()) << "CSV rows are empty. Check csv path.";

    const std::string suite = "CsvCases";
    const std::string filter = g_gtestFilter; // from argv

    int selected = 0;

    for (const auto& r : g_rows) {
        const std::string rawName = r.at("name");
        const std::string testName = SanitizeGTestName(rawName);

        if (!MatchesFilterExpr(filter, suite, testName)) {
            continue;
        }

        selected++;

        auto cr = RunOneCase(r, g_opt);
        g_results.Add(cr);

        if (g_opt.generateGolden) {
            if (cr.status == CaseStatus::ERROR) {
                ADD_FAILURE() << "[ERROR] " << cr.name << " : " << cr.detail;
            }
            continue;
        }

        if (cr.status == CaseStatus::ERROR) {
            ADD_FAILURE() << "[ERROR] " << cr.name << " : " << cr.detail;
            continue;
        }

        EXPECT_EQ(cr.status, CaseStatus::PASS)
            << "case=" << cr.name
            << " metric=" << cr.metric
            << " value=" << cr.value
            << " pass_value=" << cr.pass_value;
    }

    EXPECT_GT(selected, 0) << "No CSV cases matched filter: " << filter;
}

// ------------------------------------------------------------
// main
// ------------------------------------------------------------
int main(int argc, char** argv)
{
    if (argc < 2) {
        PrintUsage();
        return 2;
    }

    // runner args defaults
    g_opt.csvPath = argv[1];
    g_opt.gpuId = 0;
    g_opt.outdir = fs::path("tests/out");
    g_opt.saveOut = true;
    g_opt.generateGolden = false;

    // rebuild argv for gtest
    std::vector<std::string> kept;
    kept.reserve((size_t)argc);
    kept.push_back(argv[0]);

    // parse our args + capture --gtest_filter if present
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];

        if (a == "--help" || a == "-h") {
            PrintUsage();
            return 0;
        }

        if (a == "--gpu" && i + 1 < argc) { g_opt.gpuId = std::stoi(argv[++i]); continue; }
        if (a == "--outdir" && i + 1 < argc) { g_opt.outdir = argv[++i]; continue; }
        if (a == "--no-save-out") { g_opt.saveOut = false; continue; }
        if (a == "--generate-golden") { g_opt.generateGolden = true; continue; }

        // Capture filter without relying on gtest macros
        if (StartsWith(a, "--gtest_filter=")) {
            // ユーザーの gtest_filter は「CSVケース選択」に使う。gtest本体には渡さない。
            g_gtestFilter = a.substr(std::string("--gtest_filter=").size());
            continue;
        }

        kept.push_back(a);
    }

    // default filter if not provided
    if (g_gtestFilter.empty()) g_gtestFilter = "*";

    // default XML output to <outdir>/gtest.xml unless user specified --gtest_output
    const bool hasGTestOutput = std::any_of(kept.begin(), kept.end(),
        [](const std::string& s) { return StartsWith(s, "--gtest_output="); });

    if (!hasGTestOutput) {
        fs::create_directories(g_opt.outdir);
        kept.push_back(std::string("--gtest_output=xml:") + (g_opt.outdir / "gtest.xml").string());
    }

    kept.push_back("--gtest_filter=CsvCases.RunSelected");

    // InitGoogleTest with rebuilt argv
    std::vector<std::string> keptStorage = kept;
    std::vector<char*> newArgv;
    newArgv.reserve(keptStorage.size());
    for (auto& s : keptStorage) newArgv.push_back(s.data());
    int newArgc = (int)newArgv.size();

    ::testing::InitGoogleTest(&newArgc, newArgv.data());

    // load CSV
    g_rows = read_csv(g_opt.csvPath);
    if (g_rows.empty()) {
        std::cerr << "No rows loaded from csv: " << g_opt.csvPath << "\n";
        return 2;
    }

    ::testing::AddGlobalTestEnvironment(new ResultsEnv(g_opt.outdir));

    const int rc = RUN_ALL_TESTS();

    std::cout << "Done. PASS=" << g_results.PassCount()
        << " FAIL=" << g_results.FailCount()
        << " ERROR=" << g_results.ErrorCount()
        << " GENERATED=" << g_results.GeneratedCount()
        << "\n";
    std::cout << "OutDir: " << g_opt.outdir.string() << "\n";

    return rc;
}
