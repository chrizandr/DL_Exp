/*
    Copyright (c) 2016, Taiga Nomi, Edgar Riba
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/params/deconv_params.h"

namespace tiny_dnn {
namespace core {
namespace kernels {

inline void tiny_deconv2d_kernel(const deconv_params &params,
                                 const tensor_t &in,
                                 const vec_t &W,
                                 const vec_t &bias,
                                 tensor_t &a,
                                 const bool layer_parallelize) {
  for_i(layer_parallelize, in.size(), [&](int sample) {
    for (serial_size_t o = 0; o < params.out.depth_; o++) {
      for (serial_size_t inc = 0; inc < params.in.depth_; inc++) {
        if (!params.tbl.is_connected(o, inc)) continue;

        serial_size_t idx = 0;
        idx               = params.in.depth_ * o + inc;
        idx               = params.weight.get_index(0, 0, idx);
        assert(idx < W.size());
        const float_t *pw = &W[idx];

        idx = params.in.get_index(0, 0, inc);
        assert(static_cast<serial_size_t>(sample) < in.size() &&
               idx < in[sample].size());
        const float_t *pi = &in[sample][idx];

        idx = params.out.get_index(0, 0, o);
        assert(static_cast<serial_size_t>(sample) < a.size() &&
               idx < a[sample].size());
        float_t *pa = &a[sample][idx];

        for (serial_size_t y = 0; y < params.in.height_; y++) {
          for (serial_size_t x = 0; x < params.in.width_; x++) {
            const float_t *ppw = pw;
            const float_t *ppi = pi + y * params.in.width_ + x;
            // should be optimized for small kernel(3x3,5x5)
            for (serial_size_t wy = 0; wy < params.weight.height_; wy++) {
              for (serial_size_t wx = 0; wx < params.weight.width_; wx++) {
                pa[(y * params.h_stride + wy) * params.out.width_ +
                   (x * params.w_stride + wx)] +=
                  ppw[wy * params.weight.width_ + wx] * (*ppi);
              }
            }
          }
        }
      }

      if (params.has_bias) {
        float_t *pa  = &a[sample][params.out.get_index(0, 0, o)];
        float_t *paa = pa + params.out.width_ * params.out.height_;
        std::for_each(pa, paa, [&](float_t &f) { f += bias[o]; });
      }
    }
  });
}

}  // namespace kernels
}  // namespace core
}  // namespace tiny_dnn
