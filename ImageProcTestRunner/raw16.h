#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cstdint>

inline std::vector<uint16_t> load_raw16(const std::string& path, int w, int h) {
    const size_t n = (size_t)w * (size_t)h;
    std::vector<uint16_t> buf(n);

    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("Failed to open raw: " + path);

    ifs.read(reinterpret_cast<char*>(buf.data()), (std::streamsize)(n * sizeof(uint16_t)));
    if ((size_t)ifs.gcount() != n * sizeof(uint16_t)) {
        throw std::runtime_error("RAW size mismatch: " + path);
    }
    return buf;
}

inline void save_raw16(const std::string& path, const std::vector<uint16_t>& img) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) throw std::runtime_error("Failed to write raw: " + path);
    ofs.write(reinterpret_cast<const char*>(img.data()), (std::streamsize)(img.size() * sizeof(uint16_t)));
}
