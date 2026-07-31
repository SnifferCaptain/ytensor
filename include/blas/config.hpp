#pragma once
/***************
 * @file config.hpp
 * @brief YBLAS物理内核可用性配置
 ***************/

// YBLAS是YTensor自研的BLAS实现；具体指令集只在本目录内部决定。
// 当前物理内核支持AVX2+FMA，后续ISA实现应在这里合并为统一的YBLAS可用性。
#ifndef YT_USE_AVX2
    #if defined(__AVX2__) && defined(__FMA__)
        #define YT_USE_AVX2 1
    #else
        #define YT_USE_AVX2 0
    #endif
#endif

#if YT_USE_AVX2 && (!defined(__AVX2__) || !defined(__FMA__))
    #error "YT_USE_AVX2=1 requires compiler support for AVX2 and FMA"
#endif

#ifndef YT_USE_YBLAS
    #define YT_USE_YBLAS YT_USE_AVX2
#endif

#if YT_USE_YBLAS && !YT_USE_AVX2
    #error "YT_USE_YBLAS=1 requires an enabled YBLAS physical kernel"
#endif
