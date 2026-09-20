#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace cuda_kernels {

[[nodiscard]] std::vector<float> matmul_cpu(std::span<const float> left,
                                            std::span<const float> right,
                                            std::size_t m, std::size_t k, std::size_t n);
void softmax_cpu(std::span<float> values);
void layer_norm_cpu(std::span<const float> input, std::span<const float> gamma,
                    std::span<const float> beta, std::span<float> output,
                    float epsilon = 1.0e-5F);
[[nodiscard]] std::vector<float> scaled_dot_product_attention_cpu(
    std::span<const float> query, std::span<const float> key,
    std::span<const float> value, std::size_t sequence, std::size_t head_size,
    bool causal = true);

}  // namespace cuda_kernels
