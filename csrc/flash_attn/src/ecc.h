#pragma once
#include <cuda_runtime.h>
#include "cute/tensor.hpp"

namespace flash {

// writes an ECC to the tensor, returning the old one
template<class Tensor>
__device__ __forceinline__ int swap_ecc(Tensor &data, int code) {
    static constexpr uint16_t mask = 0x1;
    static constexpr int mask_interval = 4;
    auto data_raw = recast<uint16_t>(data);
    int ret = 0;
    #pragma unroll
    for (int i = 0; i < size(data) / mask_interval; i++) {
        ret += (data_raw(i * mask_interval) & mask) << i;
        data_raw(i * mask_interval) = (data_raw(i * mask_interval) & ~mask) | ((code >> i) & mask);
    }
    return ret;
}

} // namespace flashinfer
