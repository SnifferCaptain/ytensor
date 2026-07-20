#pragma once
/***************
* @file: ytensor.hpp
* @brief: 所有include文件的总入口。
* @author: SnifferCaptain
* @date: 2026-7-3
* @version 0.14
* @email: 3586554865@qq.com
***************/

#include "include/ytensor_concepts.hpp"
#include "include/ytensor_infos.hpp"
#include "include/ytensor_memory.hpp"
#include "include/ytensor_layout.hpp"
#include "include/ytensor_types.hpp"
#include "include/utils/parallel_for.hpp"
#include "include/type/type_dispatch.hpp"

// AVX2 kernel
#include "include/blas/avx2/hgemm.hpp"
#include "include/blas/avx2/sgemm.hpp"
#include "include/blas/avx2/sgemv.hpp"

// Backend switch: define YT_USE_LIB to use precompiled runtime backend.

/////////// ytensor class def ////////////
#include "include/ytensor_base.hpp"
#include "include/ytensor_strided.hpp"
#include "include/ytensor_core.hpp"

//////////// external /////////////
#include "include/ytensor_function.hpp"
#include "include/ytensor_io.hpp"

//////////// implementation /////////////
#include "src/ytensor_base_templates.inl"
#include "src/ytensor_io_templates.inl"

#if !YT_USE_LIB || defined(YT_LIBRARY_IMPLEMENTATION)
#include "src/ytensor_memory.inl"
#include "src/ytensor_layout.inl"
#include "src/ytensor_base.inl"
#include "src/ytensor_io.inl"
#endif

// Typed Strided templates must remain visible in YT_USE_LIB consumer mode.
// All definitions in this aggregate are inline in both consumer and library TUs.
#include "src/ytensor_strided.inl"

#if !YT_USE_LIB || defined(YT_LIBRARY_IMPLEMENTATION)
#include "src/ytensor_base_math.inl"
#endif

#include "src/ytensor_core.inl"
#include "src/ytensor_math.inl"

#include "src/ytensor_function.inl"

// In YT_USE_LIB consumer mode, suppress repeated builtin template instantiation.
#if YT_USE_LIB && !defined(YT_LIBRARY_IMPLEMENTATION)
#include "include/ytensor_extern_templates.hpp"
#endif
