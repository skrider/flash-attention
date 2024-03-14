#pragma once
#include <cuda_runtime.h>
#include "cute/tensor.hpp"

namespace flash {

// writes an ECC to the tensor, returning the old one
template<class Tensor>
__device__ __forceinline__ uint32_t swap_ecc(Tensor &data, uint32_t code) {
    static constexpr uint16_t mask = 0x1;
    static constexpr int mask_interval = 4;
    auto data_raw = recast<uint16_t>(data);
    uint32_t ret = 0;
    #pragma unroll
    for (int i = 0; i < size(data) / mask_interval; i++) {
        ret += (data_raw(i * mask_interval) & mask) << i;
        data_raw(i * mask_interval) = (data_raw(i * mask_interval) & ~mask) | ((code >> i) & mask);
    }
    return ret;
}

template<typename Kernel_traits>
__device__ __forceinline__ uint32_t compute_ecc(const int seq_id, const int block_pidx, const int n_block, 
    const int actual_seqlen_k, const int page_block_size) {
    const int seq_offset = std::min(n_block * Kernel_traits::kBlockN + block_pidx * page_block_size, actual_seqlen_k) % page_block_size;
    return (seq_id << Kernel_traits::maxPageSizeLog2 + 1) | seq_offset;
}

template<typename Kernel_traits>
__device__ __forceinline__ uint32_t get_seq_id(const uint32_t ecc) {
    return (ecc >> Kernel_traits::maxPageSizeLog2 + 1);
}

template <typename Kernel_traits>
__forceinline__ __device__
void invalidate_page(const int block_pidx, const int n_block, const int page_block_size, 
                    int* page_fault_mask) {
    constexpr int kBlockN = Kernel_traits::kBlockN;

    const int pages_per_block = kBlockN / page_block_size;
    const int block_idx = block_pidx + pages_per_block * n_block;
    page_fault_mask[block_idx] = 1;
}

} // namespace flashinfer
