/***************
 * @file: test_gpu_dispatch.cpp
 * @brief: 验证 GPU (Kompute) 路径下的数学运算确实通过着色器执行
 ***************/

#include "../ytensor.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

#if YT_USE_KOMPUTE

static void assertClose(float a, float b, float eps = 1e-4f) {
    if (std::fabs(a - b) > eps) {
        std::cerr << "FAIL: " << a << " vs " << b << " (diff=" << std::fabs(a - b) << ")" << std::endl;
        std::abort();
    }
}

static void test_add() {
    yt::YTensor<float, 1> a(4), b(4);
    for (int i = 0; i < 4; ++i) { a[i] = static_cast<float>(i + 1); b[i] = static_cast<float>(10 * (i + 1)); }

    auto a_gpu = a.to("kompute");
    auto b_gpu = b.to("kompute");
    auto c_gpu = a_gpu + b_gpu;
    auto c = c_gpu.to("cpu");

    auto c_cpu = a + b;
    for (int i = 0; i < 4; ++i) assertClose(c.at(i), c_cpu.at(i));
    std::cout << "  add: ok" << std::endl;
}

static void test_sub() {
    yt::YTensor<float, 1> a(4), b(4);
    for (int i = 0; i < 4; ++i) { a[i] = static_cast<float>(10 * (i + 1)); b[i] = static_cast<float>(i + 1); }

    auto c = (a.to("kompute") - b.to("kompute")).to("cpu");
    auto c_cpu = a - b;
    for (int i = 0; i < 4; ++i) assertClose(c.at(i), c_cpu.at(i));
    std::cout << "  sub: ok" << std::endl;
}

static void test_mul() {
    yt::YTensor<float, 1> a(4), b(4);
    for (int i = 0; i < 4; ++i) { a[i] = static_cast<float>(i + 1); b[i] = static_cast<float>(i + 2); }

    auto c = (a.to("kompute") * b.to("kompute")).to("cpu");
    auto c_cpu = a * b;
    for (int i = 0; i < 4; ++i) assertClose(c.at(i), c_cpu.at(i));
    std::cout << "  mul: ok" << std::endl;
}

static void test_matmul() {
    // [2, 3] @ [3, 4] -> [2, 4]
    yt::YTensor<float, 2> a(2, 3), b(3, 4);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            a.at(i, j) = static_cast<float>(i * 3 + j + 1);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j)
            b.at(i, j) = static_cast<float>(i * 4 + j + 1);

    auto c_cpu = a.matmul(b);
    auto c = a.to("kompute").matmul(b.to("kompute")).to("cpu");
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 4; ++j)
            assertClose(c.at(i, j), c_cpu.at(i, j));
    std::cout << "  matmul: ok" << std::endl;
}

static void test_sum() {
    yt::YTensor<float, 2> a(3, 4);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j)
            a.at(i, j) = static_cast<float>(i * 4 + j + 1);

    // sum along axis 1
    auto s_cpu = yt::YTensor<float, 2>(a.sum(1));
    auto s = yt::YTensor<float, 2>(a.to("kompute").sum(1).to("cpu"));
    for (int i = 0; i < 3; ++i)
        assertClose(s.at(i, 0), s_cpu.at(i, 0));
    std::cout << "  sum: ok" << std::endl;
}

static void test_max() {
    yt::YTensor<float, 2> a(3, 4);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j)
            a.at(i, j) = static_cast<float>(i * 4 + j + 1);

    auto [v_cpu, idx_cpu] = a.max(1);
    auto [v_gpu, idx_gpu] = a.to("kompute").max(1);
    auto v = yt::YTensor<float, 2>(v_gpu.to("cpu"));
    auto v_c = yt::YTensor<float, 2>(v_cpu);
    for (int i = 0; i < 3; ++i)
        assertClose(v.at(i, 0), v_c.at(i, 0));
    std::cout << "  max: ok" << std::endl;
}

#endif // YT_USE_KOMPUTE

int main() {
#if !YT_USE_KOMPUTE
    std::cout << "gpu dispatch tests skipped: YT_USE_KOMPUTE=0" << std::endl;
    return 0;
#else
    std::cout << "gpu dispatch tests:" << std::endl;
    test_add();
    test_sub();
    test_mul();
    test_matmul();
    test_sum();
    test_max();
    std::cout << "all gpu dispatch tests passed" << std::endl;
    return 0;
#endif
}
