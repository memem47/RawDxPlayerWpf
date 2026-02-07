#pragma once
#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <filesystem>
#include <mutex>

#include "csv.h"

namespace fs = std::filesystem;

// 既存の runner の挙動を維持しつつ、GoogleTest から呼べる「1ケース実行」を切り出す。
// - generateGolden=true の場合：golden を生成して status=GENERATED
// - generateGolden=false の場合：metric 判定し status=PASS/FAIL/ERROR

struct RunnerOptions
{
    std::string csvPath;
    int gpuId = 0;
    fs::path outdir = "tests/out";
    bool saveOut = true;
    bool generateGolden = false;
};

enum class CaseStatus
{
    PASS,
    FAIL,
    GENERATED,
    ERROR
};

struct CaseResult
{
    std::string name;
    std::string metric;
    double value = 0.0;
    double pass_value = 0.0;
    CaseStatus status = CaseStatus::ERROR;
    std::string input;
    std::string golden;
    std::string out;
    std::string detail; // "OK" / error message / etc
};

// results.csv を「実行されたケースだけ」記録するためのスレッドセーフ collector
class ResultsCollector
{
public:
    void Add(const CaseResult& r);
    void WriteCsv(const fs::path& outdir) const;
    int PassCount() const;
    int FailCount() const;
    int ErrorCount() const;
    int GeneratedCount() const;

private:
    mutable std::mutex m_;
    std::vector<CaseResult> results_;
};

// 1ケース実行
CaseResult RunOneCase(const CsvRow& r, const RunnerOptions& opt);

// ユーティリティ：TestName で使える安全な文字列に変換（gtest はテスト名に制約あり）
std::string SanitizeGTestName(const std::string& s);