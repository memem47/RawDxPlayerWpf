#pragma once
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <unordered_map>
#include <stdexcept>
#include <algorithm>

inline std::string trim(std::string s) {
    auto notsp = [](int c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), notsp));
    s.erase(std::find_if(s.rbegin(), s.rend(), notsp).base(), s.end());
    return s;
}

// クォート対応の最低限CSV split（".." 内のカンマを許容）
inline std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    bool inq = false;
    for (size_t i = 0; i < line.size(); i++) {
        char c = line[i];
        if (c == '"') { inq = !inq; continue; }
        if (!inq && c == ',') { out.push_back(trim(cur)); cur.clear(); continue; }
        cur.push_back(c);
    }
    out.push_back(trim(cur));
    return out;
}

struct CsvRow {
    std::unordered_map<std::string, std::string> m;
    const std::string& at(const std::string& k) const {
        auto it = m.find(k);
        if (it == m.end()) throw std::runtime_error("CSV missing key: " + k);
        return it->second;
    }
    int i(const std::string& k) const { return std::stoi(at(k)); }
    double d(const std::string& k) const { return std::stod(at(k)); }
};

inline std::vector<CsvRow> read_csv(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) throw std::runtime_error("Failed to open csv: " + path);

    std::string line;
    std::vector<std::string> header;

    // ヘッダまで読み飛ばし（#コメント、空行スキップ）
    while (std::getline(ifs, line)) {
        line = trim(line);
        if (line.empty() || line.rfind("#", 0) == 0) continue;
        header = split_csv_line(line);
        break;
    }
    if (header.empty()) throw std::runtime_error("CSV header not found: " + path);

    std::vector<CsvRow> rows;
    while (std::getline(ifs, line)) {
        line = trim(line);
        if (line.empty() || line.rfind("#", 0) == 0) continue;
        auto cols = split_csv_line(line);
        if (cols.size() < header.size()) {
            throw std::runtime_error("CSV column count mismatch: " + line);
        }
        CsvRow r;
        for (size_t i = 0; i < header.size(); i++) {
            r.m[header[i]] = cols[i];
        }
        rows.push_back(std::move(r));
    }
    return rows;
}
