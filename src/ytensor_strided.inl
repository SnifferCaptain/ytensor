#pragma once
/***************
* file: ytensor_strided.inl
* purpose: strided layout 聚合入口。
 ***************/

#if defined(YT_LIBRARY_IMPLEMENTATION)
#define YT_RESTORE_LIBRARY_IMPL_INLINE
#undef YT_IMPL_INLINE
#define YT_IMPL_INLINE inline
#endif

#include "strided/broadcast.inl"
#include "strided/view.inl"
#include "strided/copy.inl"
#include "strided/reduce.inl"
#include "strided/matmul.inl"

#if defined(YT_RESTORE_LIBRARY_IMPL_INLINE)
#undef YT_IMPL_INLINE
#define YT_IMPL_INLINE
#undef YT_RESTORE_LIBRARY_IMPL_INLINE
#endif
