#pragma once
/***************
* @file: ytensor.hpp
* @brief: 所有include文件的总入口。
* @author: SnifferCaptain
* @date: 2026-1-10
* @version 0.3
* @email: 3586554865@qq.com
***************/

#include "include/ytensor_concepts.hpp"
#include "include/ytensor_infos.hpp"
#include "include/ytensor_types.hpp"
#include "include/kernel/parallel_for.hpp"
#include "include/kernel/type_dispatch.hpp"

// Backend switch: define YT_USE_LIB to use precompiled runtime backend.

/////////// ytensor class def ////////////
#include "include/ytensor_base.hpp"
#include "include/ytensor_core.hpp"

//////////// external /////////////
#include "include/ytensor_function.hpp"
#include "include/ytensor_io.hpp"

//////////// implementation /////////////
#include "src/ytensor_base_templates.inl"
#include "src/ytensor_io_templates.inl"

#if !defined(YT_USE_LIB) || defined(YT_LIBRARY_IMPLEMENTATION)
#if defined(YT_LIBRARY_IMPLEMENTATION)
// Emit YTensorBase/YTensorIO runtime symbols in the library TU.
#define YT_RUNTIME_OUT_OF_LINE 1
#define inline
#endif
#include "src/ytensor_base.inl"
#include "src/ytensor_io.inl"
#if defined(YT_RUNTIME_OUT_OF_LINE)
#undef inline
#undef YT_RUNTIME_OUT_OF_LINE
#endif
#endif

#if !defined(YT_USE_LIB) || defined(YT_LIBRARY_IMPLEMENTATION)
#if defined(YT_LIBRARY_IMPLEMENTATION)
// Emit YTensorBase math symbols in the library TU.
#define YT_RUNTIME_OUT_OF_LINE 1
#define inline
#endif
#include "src/ytensor_base_math.inl"
#if defined(YT_RUNTIME_OUT_OF_LINE)
#undef inline
#undef YT_RUNTIME_OUT_OF_LINE
#endif
#endif

#include "src/ytensor_core.inl"
#include "src/ytensor_math.inl"

#include "src/ytensor_function.inl"

// In YT_USE_LIB consumer mode, suppress repeated builtin template instantiation.
#if defined(YT_USE_LIB) && !defined(YT_LIBRARY_IMPLEMENTATION)
#include "include/ytensor_extern_templates.hpp"
#endif
