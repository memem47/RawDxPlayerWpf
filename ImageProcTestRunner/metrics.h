#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>

struct MetricResult {
    double value = 0.0;  // metricíl
};

inline MetricResult metric_exact(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    // value=0:àÍívÅA1:ïsàÍív
    MetricResult r;
    if (a.size() != b.size()) { r.value = 1; return r; }
    r.value = std::equal(a.begin(), a.end(), b.begin()) ? 0.0 : 1.0;
    return r;
}

inline MetricResult metric_max_abs(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    MetricResult r;
    double m = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double d = std::abs((int)a[i] - (int)b[i]);
        m = std::max(m, d);
    }
    r.value = m;
    return r;
}

inline MetricResult metric_mae(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    MetricResult r;
    double s = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        s += std::abs((int)a[i] - (int)b[i]);
    }
    r.value = s / (double)a.size();
    return r;
}

inline MetricResult metric_psnr(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    MetricResult r;
    double mse = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double d = (double)((int)a[i] - (int)b[i]);
        mse += d * d;
    }
    mse /= (double)a.size();
    if (mse <= 0.0) { r.value = 99.0; return r; }
    const double peak = 65535.0;
    r.value = 10.0 * std::log10((peak * peak) / mse);
    return r;
}
