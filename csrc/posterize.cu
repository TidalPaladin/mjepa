#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <torch/extension.h>

template <typename scalar_t>
__global__ void posterize_kernel(const scalar_t *__restrict__ input,
                                 scalar_t *__restrict__ output,
                                 const float posterize_prob, const int bits,
                                 const int64_t batch_size,
                                 const int64_t seq_len, const int64_t seed) {
  const int batch_idx = blockIdx.y;
  const int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (seq_idx >= seq_len || batch_idx >= batch_size)
    return;

  // Load input value
  const int64_t idx = batch_idx * seq_len + seq_idx;
  scalar_t val = input[idx];

  // Decide if this batch entry should be posterized
  curandState state;
  curand_init(seed + batch_idx, 0, 0, &state);
  const bool apply_posterize = curand_uniform(&state) < posterize_prob;

  // Apply posterization if needed
  if (apply_posterize) {
    // Calculate the number of levels based on bits
    const int levels = (1 << bits) - 1;

    // Quantize the value to the specified number of bits
    // First scale to [0, levels], round to nearest integer, then scale back to
    // [0, 1]
    val = roundf(val * levels) / levels;
  }

  output[idx] = val;
}

torch::Tensor posterize(const torch::Tensor &input, const float posterize_prob,
                        const int bits, const int64_t seed) {
  // Validate bits parameter (typically between 1 and 8)
  TORCH_CHECK(bits >= 1 && bits <= 8, "Number of bits must be between 1 and 8");

  // Prepare output and infer dimensions
  auto output = torch::empty_like(input);
  const int64_t batch_size = input.size(0);
  const int64_t seq_len = input.numel() / batch_size;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
      "posterize", ([&] {
        int min_grid_size;
        int block_size;
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                           (void *)posterize_kernel<scalar_t>,
                                           0, 0);
        const unsigned int blocks_x = (seq_len + block_size - 1) / block_size;
        const dim3 blocks(blocks_x, batch_size);

        posterize_kernel<scalar_t><<<blocks, block_size>>>(
            input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
            posterize_prob, bits, batch_size, seq_len, seed);
      }));

  return output;
}

void posterize_(torch::Tensor &input, const float posterize_prob,
                const int bits, const int64_t seed) {
  // Validate bits parameter (typically between 1 and 8)
  TORCH_CHECK(bits >= 1 && bits <= 8, "Number of bits must be between 1 and 8");

  // Prepare output and infer dimensions
  const int64_t batch_size = input.size(0);
  const int64_t seq_len = input.numel() / batch_size;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
      "posterize", ([&] {
        int min_grid_size;
        int block_size;
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                           (void *)posterize_kernel<scalar_t>,
                                           0, 0);
        const unsigned int blocks_x = (seq_len + block_size - 1) / block_size;
        const dim3 blocks(blocks_x, batch_size);

        posterize_kernel<scalar_t><<<blocks, block_size>>>(
            input.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
            posterize_prob, bits, batch_size, seq_len, seed);
      }));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("posterize", &posterize, "Posterize operation");
  m.def("posterize_", &posterize_, "Posterize operation in-place");
}
