#pragma once
/***************
 * file: blas/kernels/avx2/aligned_buffer.inl
 * purpose: 64字节对齐GEMM工作区。
 ***************/

#if defined(__AVX2__) && defined(__FMA__)
namespace yt::blas {

inline void* aligned_alloc_64(size_t size) {
    void* ptr = nullptr;
#ifdef _WIN32
    ptr = _aligned_malloc(size, 64);
#else
    if (posix_memalign(&ptr, 64, size) != 0) ptr = nullptr;
#endif
    return ptr;
}

inline void aligned_free_64(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

inline void AlignedBuffer::ensure(size_t n) {
    if (n <= capacity) return;
    auto* next = static_cast<float*>(aligned_alloc_64(n * sizeof(float)));
    if (!next) return;
    if (data) aligned_free_64(data);
    data = next;
    capacity = n;
}

}  // namespace yt::blas
#endif
