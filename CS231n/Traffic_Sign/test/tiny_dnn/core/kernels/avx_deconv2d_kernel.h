/*
    Copyright (c) 2016, Taiga Nomi, Edgar Riba
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/kernels/tiny_deconv2d_kernel.h"
#include "tiny_dnn/core/params/deconv_params.h"

namespace tiny_dnn {
namespace core {
namespace kernels {

inline void avx_deconv2d_kernel(const deconv_params &params,
                                const tensor_t &in,
                                const vec_t &W,
                                const vec_t &bias,
                                tensor_t &a,
                                const bool layer_parallelize) {
  // fallback to non-avx version
  tiny_deconv2d_kernel(params, in, W, bias, a, layer_parallelize);
}

}  // namespace kernels
}  // namespace core
}  // namespace tiny_dnn
