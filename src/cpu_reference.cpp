#include "cuda_kernels/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace cuda_kernels {
std::vector<float> matmul_cpu(const std::span<const float> left, const std::span<const float> right, const std::size_t m, const std::size_t k, const std::size_t n) {
  if (left.size() != m * k || right.size() != k * n) throw std::invalid_argument("matmul storage mismatch"); std::vector<float> output(m * n); for (std::size_t i = 0; i < m; ++i) for (std::size_t p = 0; p < k; ++p) { const float a = left[i * k + p]; for (std::size_t j = 0; j < n; ++j) output[i * n + j] += a * right[p * n + j]; } return output;
}
void softmax_cpu(const std::span<float> values) { if (values.empty()) return; const float maximum = *std::max_element(values.begin(), values.end()); float sum = 0.0F; for (float& value : values) { value = std::exp(value - maximum); sum += value; } for (float& value : values) value /= sum; }
void layer_norm_cpu(const std::span<const float> input, const std::span<const float> gamma, const std::span<const float> beta, const std::span<float> output, const float epsilon) { if (input.size() != gamma.size() || input.size() != beta.size() || input.size() != output.size() || input.empty()) throw std::invalid_argument("layer norm shape mismatch"); const float mean = std::accumulate(input.begin(), input.end(), 0.0F) / static_cast<float>(input.size()); float variance = 0.0F; for (const float value : input) variance += (value - mean) * (value - mean); variance /= static_cast<float>(input.size()); const float inverse = 1.0F / std::sqrt(variance + epsilon); for (std::size_t i = 0; i < input.size(); ++i) output[i] = (input[i] - mean) * inverse * gamma[i] + beta[i]; }
std::vector<float> scaled_dot_product_attention_cpu(const std::span<const float> query, const std::span<const float> key, const std::span<const float> value, const std::size_t sequence, const std::size_t dimension, const bool causal) { const auto expected = sequence * dimension; if (query.size() != expected || key.size() != expected || value.size() != expected) throw std::invalid_argument("attention shape mismatch"); std::vector<float> output(expected); std::vector<float> scores(sequence); const float scale = 1.0F / std::sqrt(static_cast<float>(dimension)); for (std::size_t row = 0; row < sequence; ++row) { for (std::size_t column = 0; column < sequence; ++column) { if (causal && column > row) scores[column] = -std::numeric_limits<float>::infinity(); else { float dot = 0.0F; for (std::size_t item = 0; item < dimension; ++item) dot += query[row * dimension + item] * key[column * dimension + item]; scores[column] = dot * scale; } } softmax_cpu(scores); for (std::size_t column = 0; column < sequence; ++column) for (std::size_t item = 0; item < dimension; ++item) output[row * dimension + item] += scores[column] * value[column * dimension + item]; } return output; }
}  // namespace cuda_kernels
