#include <cuda_runtime.h>

#include <cmath>

extern "C" __global__ void research_matmul(const float* a, const float* b, float* c, int m, int k, int n) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y; const int column = blockIdx.x * blockDim.x + threadIdx.x; if (row >= m || column >= n) return; float sum = 0.0F; for (int item = 0; item < k; ++item) sum += a[row * k + item] * b[item * n + column]; c[row * n + column] = sum;
}

extern "C" __global__ void research_softmax_rows(float* values, int rows, int columns) {
  const int row = blockIdx.x; if (row >= rows || threadIdx.x != 0) return; float maximum = -INFINITY; for (int column = 0; column < columns; ++column) maximum = fmaxf(maximum, values[row * columns + column]); float sum = 0.0F; for (int column = 0; column < columns; ++column) { values[row * columns + column] = expf(values[row * columns + column] - maximum); sum += values[row * columns + column]; } for (int column = 0; column < columns; ++column) values[row * columns + column] /= sum;
}

extern "C" __global__ void research_layer_norm(const float* input, const float* gamma, const float* beta, float* output, int rows, int columns, float epsilon) {
  const int row = blockIdx.x; if (row >= rows || threadIdx.x != 0) return; float mean = 0.0F; for (int column = 0; column < columns; ++column) mean += input[row * columns + column]; mean /= columns; float variance = 0.0F; for (int column = 0; column < columns; ++column) { const float delta = input[row * columns + column] - mean; variance += delta * delta; } const float inverse = rsqrtf(variance / columns + epsilon); for (int column = 0; column < columns; ++column) output[row * columns + column] = (input[row * columns + column] - mean) * inverse * gamma[column] + beta[column];
}

extern "C" __global__ void research_causal_attention(const float* query, const float* key,
                                                       const float* value, float* output,
                                                       int sequence, int dimension) {
  const int row = blockIdx.x;
  const int item = threadIdx.x;
  if (row >= sequence || item >= dimension) return;
  extern __shared__ float scores[];
  if (item == 0) {
    float maximum = -INFINITY;
    for (int column = 0; column <= row; ++column) {
      float dot = 0.0F;
      for (int d = 0; d < dimension; ++d)
        dot += query[row * dimension + d] * key[column * dimension + d];
      scores[column] = dot * rsqrtf(static_cast<float>(dimension));
      maximum = fmaxf(maximum, scores[column]);
    }
    float sum = 0.0F;
    for (int column = 0; column <= row; ++column) {
      scores[column] = expf(scores[column] - maximum);
      sum += scores[column];
    }
    for (int column = 0; column <= row; ++column) scores[column] /= sum;
  }
  __syncthreads();
  float result = 0.0F;
  for (int column = 0; column <= row; ++column)
    result += scores[column] * value[column * dimension + item];
  output[row * dimension + item] = result;
}
