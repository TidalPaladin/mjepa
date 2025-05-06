/*
Implements a fused kernel for applying MixUp to a batch of images or categorical
labels.
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <torch/extension.h>

#define WARP_SIZE 32
#define UNKNOWN_LABEL -1

__device__ __forceinline__ float sample_beta(curandState *state,
                                             const float alpha) {
  const float divisor = 1.0f / alpha;
  float u = powf(curand_uniform(state), divisor);
  float v = powf(curand_uniform(state), divisor);
  return u / (u + v);
}

template <typename scalar_t, typename weight_t>
__device__ __forceinline__ scalar_t lerp(scalar_t val, scalar_t mixup_val,
                                         weight_t weight) {
  // w * x + (1 - w) * y = w * (x - y) + y
  return __fmaf_rn(weight, val - mixup_val, mixup_val);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t softplus(scalar_t x) {
  return fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x)));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t stable_sigmoid(scalar_t x) {
  if (x > 0.0f) {
    return 1.0f / (1.0f + expf(-x));
  } else {
    return expf(x - softplus(x));
  }
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_max(scalar_t val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val = max(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_sum(scalar_t val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

template <typename scalar_t>
__global__ void
get_weights_kernel(scalar_t *__restrict__ output, const float mixup_prob,
                   const float mixup_alpha, const int64_t batch_size,
                   const int64_t seed) {
  const int batch_idx = blockIdx.x;
  if (batch_idx >= batch_size)
    return;
  curandState batch_state;
  curand_init(seed + batch_idx, 0, 0, &batch_state);
  const bool apply_mixup = curand_uniform(&batch_state) < mixup_prob;
  const scalar_t weight = sample_beta(&batch_state, mixup_alpha);
  if (!apply_mixup) {
    output[batch_idx] = 1.0f;
  } else {
    output[batch_idx] = weight;
  }
}

torch::Tensor get_weights(const int64_t batch_size, const float mixup_prob,
                          const float mixup_alpha, const int64_t seed,
                          const torch::Device &device) {
  TORCH_CHECK(device.is_cuda(), "Device must be CUDA");
  auto output = torch::empty(
      {batch_size},
      torch::TensorOptions().dtype(torch::kFloat32).device(device));
  AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "get_weights", ([&] {
                               get_weights_kernel<scalar_t><<<batch_size, 1>>>(
                                   output.data_ptr<scalar_t>(), mixup_prob,
                                   mixup_alpha, batch_size, seed);
                             }));
  return output;
}

__global__ void is_mixed_kernel(bool *__restrict__ output,
                                const float mixup_prob, const float mixup_alpha,
                                const int64_t batch_size, const int64_t seed) {
  const int batch_idx = blockIdx.x;
  if (batch_idx >= batch_size)
    return;
  curandState batch_state;
  curand_init(seed + batch_idx, 0, 0, &batch_state);
  const bool apply_mixup = curand_uniform(&batch_state) < mixup_prob;
  output[batch_idx] = apply_mixup;
}

torch::Tensor is_mixed(const int64_t batch_size, const float mixup_prob,
                       const float mixup_alpha, const int64_t seed,
                       const torch::Device &device) {
  TORCH_CHECK(device.is_cuda(), "Device must be CUDA");
  auto output = torch::empty(
      {batch_size}, torch::TensorOptions().dtype(torch::kBool).device(device));
  is_mixed_kernel<<<batch_size, 1>>>(output.data_ptr<bool>(), mixup_prob,
                                     mixup_alpha, batch_size, seed);
  return output;
}

template <typename scalar_t>
__global__ void mixup_kernel(const scalar_t *__restrict__ input,
                             scalar_t *__restrict__ output,
                             const float mixup_prob, const float mixup_alpha,
                             const int64_t batch_size, const int64_t seq_len,
                             const int64_t seed) {
  const int batch_idx = blockIdx.y;
  const int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (seq_idx >= seq_len || batch_idx >= batch_size)
    return;

  // Load input value
  const int64_t idx = batch_idx * seq_len + seq_idx;
  scalar_t val = input[idx];

  // Decide if MixUp should be applied to this batch entry
  curandState batch_state;
  curand_init(seed + batch_idx, 0, 0, &batch_state);
  const bool apply_mixup = curand_uniform(&batch_state) < mixup_prob;
  if (!apply_mixup) {
    output[idx] = val;
    return;
  }

  // Generate a mixup weight for this batch entry
  const scalar_t weight = sample_beta(&batch_state, mixup_alpha);

  // Load the value to be mixed
  const int64_t mixup_batch_idx = (batch_idx + 1) % batch_size;
  const int64_t mixup_idx = mixup_batch_idx * seq_len + seq_idx;
  scalar_t mixup_val = input[mixup_idx];

  // Apply the mixup weight
  val = lerp(val, mixup_val, weight);

  output[idx] = val;
}

torch::Tensor mixup(const torch::Tensor &input, const float mixup_prob,
                    const float mixup_alpha, const int64_t seed) {
  // Prepare output and infer dimensions
  auto output = torch::empty_like(input);
  const int64_t batch_size = input.size(0);
  const int64_t seq_len = input.numel() / batch_size;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
      "mixup", ([&] {
        int min_grid_size;
        int block_size;
        cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size, &block_size, (void *)mixup_kernel<scalar_t>, 0, 0);
        const unsigned int blocks_x = (seq_len + block_size - 1) / block_size;
        const dim3 blocks(blocks_x, batch_size);

        mixup_kernel<scalar_t><<<blocks, block_size>>>(
            input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), mixup_prob,
            mixup_alpha, batch_size, seq_len, seed);
      }));

  return output;
}

template <typename scalar_t>
__global__ void cross_entropy_mixup_fwd_kernel(
    const scalar_t *__restrict__ logits, const int64_t *__restrict__ labels,
    scalar_t *__restrict__ output, float *__restrict__ denom,
    float *__restrict__ max_val_buffer, const float mixup_prob,
    const float mixup_alpha, const int64_t batch_size,
    const int64_t num_classes, const int64_t seed) {
  const int batch_idx = blockIdx.x;
  if (batch_idx >= batch_size)
    return;

  // Initialize RNG state for this batch element
  curandState batch_state;
  curand_init(seed + batch_idx, 0, 0, &batch_state);

  // Determine if we apply MixUp
  const bool apply_mixup = curand_uniform(&batch_state) < mixup_prob;

  // Get the label for current batch
  const int label_idx = static_cast<int>(labels[batch_idx]);
  scalar_t loss = 0.0f;

  // Check for unknown label, set loss to -1
  if (label_idx == UNKNOWN_LABEL) {
    output[batch_idx] = -1.0f;
    denom[batch_idx] = 1.0f;
    max_val_buffer[batch_idx] = -INFINITY;
    return;
  }

  // Softmax using online softmax trick (FP32 buffers)
  float max_val = -INFINITY;
  float sum_exp = 0.0f;
  for (int c = threadIdx.x; c < num_classes; c += blockDim.x) {
    float logit = static_cast<float>(logits[batch_idx * num_classes + c]);
    float old_max = max_val;
    max_val = fmaxf(old_max, warp_max(logit));
    max_val = __shfl_sync(0xffffffff, max_val, 0);
    float update = warp_sum(expf(logit - max_val));
    update = __shfl_sync(0xffffffff, update, 0);
    sum_exp = __fma_rn(sum_exp, expf(old_max - max_val), update);
  }
  const float log_sum_exp = logf(sum_exp);
  if (threadIdx.x == 0) {
    denom[batch_idx] = sum_exp;
    max_val_buffer[batch_idx] = max_val;
  }

  if (apply_mixup) {
    // Generate MixUp weight using same method as image MixUp
    const float weight = sample_beta(&batch_state, mixup_alpha);

    // Get the next batch's label for mixing
    const int mixup_batch_idx = (batch_idx + 1) % batch_size;
    const int mixup_label_idx = static_cast<int>(labels[mixup_batch_idx]);
    if (mixup_label_idx == UNKNOWN_LABEL) {
      output[batch_idx] = -1.0f;
      return;
    }

    const float log_softmax_val_orig =
        logits[batch_idx * num_classes + label_idx] - max_val - log_sum_exp;
    const float log_softmax_val_mix =
        logits[batch_idx * num_classes + mixup_label_idx] - max_val -
        log_sum_exp;
    loss -= lerp(log_softmax_val_orig, log_softmax_val_mix, weight);
  } else {
    // Standard cross entropy without MixUp
    const float log_softmax_val =
        static_cast<float>(logits[batch_idx * num_classes + label_idx]) -
        max_val - log_sum_exp;
    loss = -log_softmax_val;
  }

  if (threadIdx.x == 0) {
    output[batch_idx] = static_cast<scalar_t>(loss);
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
cross_entropy_mixup_fwd(const torch::Tensor &logits,
                        const torch::Tensor &labels, const float mixup_prob,
                        const float mixup_alpha, const int64_t seed) {
  TORCH_CHECK(logits.dim() == 2, "Logits must be 2D tensor");
  TORCH_CHECK(labels.dim() == 1, "Labels must be 1D tensor");
  TORCH_CHECK(logits.size(0) == labels.size(0), "Batch sizes must match");
  TORCH_CHECK(labels.scalar_type() == torch::kInt64,
              "Labels must be torch.long");

  const auto batch_size = logits.size(0);
  const auto num_classes = logits.size(1);
  auto output = torch::empty({batch_size}, logits.options());
  auto denom =
      torch::empty({batch_size}, logits.options().dtype(torch::kFloat32));
  auto max_val_buffer =
      torch::empty({batch_size}, logits.options().dtype(torch::kFloat32));

  const size_t block_size = WARP_SIZE;
  const size_t num_blocks = batch_size;
  const dim3 blocks(num_blocks);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, logits.scalar_type(),
      "cross_entropy_mixup_fwd", ([&] {
        cross_entropy_mixup_fwd_kernel<scalar_t><<<blocks, block_size>>>(
            logits.data_ptr<scalar_t>(), labels.data_ptr<int64_t>(),
            output.data_ptr<scalar_t>(), denom.data_ptr<float>(),
            max_val_buffer.data_ptr<float>(), mixup_prob, mixup_alpha,
            batch_size, num_classes, seed);
      }));

  return std::make_tuple(output, denom, max_val_buffer);
}

template <typename scalar_t>
__global__ void cross_entropy_mixup_bwd_kernel(
    const scalar_t *__restrict__ logits, const int64_t *__restrict__ labels,
    const float *__restrict__ denom_buffer,
    const float *__restrict__ max_val_buffer,
    const scalar_t *__restrict__ grad_output, scalar_t *__restrict__ output,
    const float mixup_prob, const float mixup_alpha, const int64_t batch_size,
    const int64_t num_classes, const int64_t seed) {
  const int batch_idx = blockIdx.y;
  const int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t idx = batch_idx * num_classes + seq_idx;
  if (seq_idx >= num_classes || batch_idx >= batch_size)
    return;

  // Initialize RNG state for this batch element
  curandState batch_state;
  curand_init(seed + batch_idx, 0, 0, &batch_state);

  // Determine if we apply MixUp
  const bool apply_mixup = curand_uniform(&batch_state) < mixup_prob;

  // Get the label for current batch
  const int label_idx = static_cast<int>(labels[batch_idx]);

  // Check for unknown label, set grad_output to 0
  if (label_idx == UNKNOWN_LABEL) {
    output[idx] = 0.0f;
    return;
  }

  // Get the logit to compute gradient for and compute a probability using
  // stored denominator
  const float max_val = max_val_buffer[batch_idx];
  const float logit = static_cast<float>(logits[idx]) - max_val;
  const float sum_exp = denom_buffer[batch_idx];
  const float prob = expf(logit) / sum_exp;

  // Compute the target probability accounting for MixUp
  float target_prob = 0.0f;
  if (apply_mixup) {
    // Generate MixUp weight using same method as image MixUp
    const float weight = sample_beta(&batch_state, mixup_alpha);

    // Check if the mixed label is unknown
    const int mixup_batch_idx = (batch_idx + 1) % batch_size;
    const int mixup_label_idx = static_cast<int>(labels[mixup_batch_idx]);
    if (mixup_label_idx == UNKNOWN_LABEL) {
      output[idx] = 0.0f;
      return;
    }

    const float target = (seq_idx == label_idx) ? 1.0f : 0.0f;
    const float mixed_target = (seq_idx == mixup_label_idx) ? 1.0f : 0.0f;
    target_prob = weight * target + (1.0f - weight) * mixed_target;
  } else {
    target_prob = seq_idx == label_idx ? 1.0f : 0.0f;
  }

  // Compute the gradient for this logit
  const float grad = (prob - target_prob);

  // Apply gradient to output
  output[idx] = static_cast<scalar_t>(grad);
}

torch::Tensor cross_entropy_mixup_bwd(
    const torch::Tensor &logits, const torch::Tensor &labels,
    const torch::Tensor &denom, const torch::Tensor &max_val,
    const torch::Tensor &grad_output, const float mixup_prob,
    const float mixup_alpha, const int64_t seed) {
  TORCH_CHECK(logits.dim() == 2, "Logits must be 2D tensor");
  TORCH_CHECK(labels.dim() == 1, "Labels must be 1D tensor");
  TORCH_CHECK(logits.size(0) == labels.size(0), "Batch sizes must match");
  TORCH_CHECK(labels.scalar_type() == torch::kInt64,
              "Labels must be torch.long");

  const auto batch_size = logits.size(0);
  const auto num_classes = logits.size(1);
  auto output = torch::empty_like(logits);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, logits.scalar_type(),
      "cross_entropy_mixup_bwd", ([&] {
        // Calculate grid dimensions
        int min_grid_size;
        int block_size;
        cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size, &block_size,
            (void *)cross_entropy_mixup_bwd_kernel<scalar_t>, 0, 0);
        const unsigned int blocks_x =
            (num_classes + block_size - 1) / block_size;
        const dim3 blocks(blocks_x, batch_size);
        cross_entropy_mixup_bwd_kernel<scalar_t><<<blocks, block_size>>>(
            logits.data_ptr<scalar_t>(), labels.data_ptr<int64_t>(),
            denom.data_ptr<float>(), max_val.data_ptr<float>(),
            grad_output.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
            mixup_prob, mixup_alpha, batch_size, num_classes, seed);
      }));

  return output;
}

template <typename scalar_t>
__global__ void bce_mixup_fwd_kernel(
    const scalar_t *__restrict__ logits, const scalar_t *__restrict__ labels,
    scalar_t *__restrict__ output, const float mixup_prob,
    const float mixup_alpha, const float pos_weight, const int64_t batch_size,
    const int64_t seq_len, const int64_t seed) {
  const int batch_idx = blockIdx.y;
  const int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t idx = batch_idx * seq_len + seq_idx;
  if (seq_idx >= seq_len || batch_idx >= batch_size)
    return;

  // MixUp probability and parameters
  curandState batch_state;
  curand_init(seed + batch_idx, 0, 0, &batch_state);
  const bool apply_mixup = curand_uniform(&batch_state) < mixup_prob;
  const int64_t mixup_batch_idx = (batch_idx + 1) % batch_size;
  const int64_t mixup_idx = mixup_batch_idx * seq_len + seq_idx;

  // Get the label for current batch
  const scalar_t logit = logits[idx];
  const scalar_t target = labels[idx];

  // Check for unknown label, set loss to -1
  if (target == UNKNOWN_LABEL) {
    output[idx] = -1.0f;
    return;
  }

  scalar_t loss = 0.0f;
  if (apply_mixup) {
    const scalar_t mixup_target = labels[mixup_idx];
    if (mixup_target == UNKNOWN_LABEL) {
      output[idx] = -1.0f;
      return;
    }
    const scalar_t weight = sample_beta(&batch_state, mixup_alpha);
    const scalar_t mixed_target = lerp(target, mixup_target, weight);
    loss = softplus(logit) - logit * mixed_target;
    if (pos_weight >= 0.0f) {
      loss *= lerp(mixed_target, 1 - mixed_target, pos_weight);
    }
  } else {
    loss = softplus(logit) - logit * target;
    if (pos_weight >= 0.0f) {
      loss *= lerp(target, 1 - target, pos_weight);
    }
  }

  output[idx] = loss;
}

torch::Tensor bce_mixup_fwd(const torch::Tensor &logits,
                            const torch::Tensor &labels, const float mixup_prob,
                            const float mixup_alpha, const float pos_weight,
                            const int64_t seed) {
  TORCH_CHECK(logits.dim() >= 2, "Logits must be >=2D tensor");
  TORCH_CHECK(labels.dim() >= 2, "Labels must be >=2D tensor");
  TORCH_CHECK(logits.sizes() == labels.sizes(),
              "Logits and labels must have the same shape");
  TORCH_CHECK(pos_weight < 0.0f || (pos_weight >= 0.0f && pos_weight <= 1.0f),
              "pos_weight must be in range [0, 1] or -1 for no weighting");

  const auto batch_size = logits.size(0);
  const auto seq_len = logits.numel() / batch_size;
  auto output = torch::empty_like(logits);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, logits.scalar_type(),
      "bce_mixup_fwd", ([&] {
        int min_grid_size;
        int block_size;
        cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size, &block_size, (void *)bce_mixup_fwd_kernel<scalar_t>,
            0, 0);
        const unsigned int blocks_x = (seq_len + block_size - 1) / block_size;
        const dim3 blocks(blocks_x, batch_size);
        bce_mixup_fwd_kernel<scalar_t><<<blocks, block_size>>>(
            logits.data_ptr<scalar_t>(), labels.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(), mixup_prob, mixup_alpha, pos_weight,
            batch_size, seq_len, seed);
      }));

  return output;
}

template <typename scalar_t>
__global__ void bce_mixup_bwd_kernel(
    const scalar_t *__restrict__ logits, const scalar_t *__restrict__ labels,
    const scalar_t *__restrict__ grad_output, scalar_t *__restrict__ output,
    const float mixup_prob, const float mixup_alpha, const float pos_weight,
    const int64_t batch_size, const int64_t seq_len, const int64_t seed) {
  const int batch_idx = blockIdx.y;
  const int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t idx = batch_idx * seq_len + seq_idx;
  if (seq_idx >= seq_len || batch_idx >= batch_size)
    return;

  // MixUp probability and parameters
  curandState batch_state;
  curand_init(seed + batch_idx, 0, 0, &batch_state);
  const bool apply_mixup = curand_uniform(&batch_state) < mixup_prob;
  const int64_t mixup_batch_idx = (batch_idx + 1) % batch_size;
  const int64_t mixup_idx = mixup_batch_idx * seq_len + seq_idx;

  // Get the label for current batch
  const scalar_t logit = logits[idx];
  const scalar_t target = labels[idx];

  // Check for unknown label, set grad to 0
  if (target == UNKNOWN_LABEL) {
    output[idx] = 0.0f;
    return;
  }

  scalar_t grad = 0.0f;
  if (apply_mixup) {
    const scalar_t mixup_target = labels[mixup_idx];
    if (mixup_target == UNKNOWN_LABEL) {
      output[idx] = 0.0f;
      return;
    }
    const scalar_t weight = sample_beta(&batch_state, mixup_alpha);
    const scalar_t mixed_target = lerp(target, mixup_target, weight);
    grad = stable_sigmoid(logit) - mixed_target;
    if (pos_weight >= 0.0f) {
      grad *= lerp(mixed_target, 1 - mixed_target, pos_weight);
    }
  } else {
    grad = stable_sigmoid(logit) - target;
    if (pos_weight >= 0.0f) {
      grad *= lerp(target, 1 - target, pos_weight);
    }
  }

  output[idx] = grad;
}

torch::Tensor bce_mixup_bwd(const torch::Tensor &logits,
                            const torch::Tensor &labels,
                            const torch::Tensor &grad_output,
                            const float mixup_prob, const float mixup_alpha,
                            const float pos_weight, const int64_t seed) {
  TORCH_CHECK(logits.dim() >= 2, "Logits must be >=2D tensor");
  TORCH_CHECK(labels.dim() >= 2, "Labels must be >=2D tensor");
  TORCH_CHECK(logits.sizes() == labels.sizes(),
              "Logits and labels must have the same shape");
  TORCH_CHECK(pos_weight < 0.0f || (pos_weight >= 0.0f && pos_weight <= 1.0f),
              "pos_weight must be in range [0, 1] or -1 for no weighting");

  const auto batch_size = logits.size(0);
  const auto seq_len = logits.numel() / batch_size;
  auto output = torch::empty_like(logits);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, logits.scalar_type(),
      "bce_mixup_bwd", ([&] {
        int min_grid_size;
        int block_size;
        cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size, &block_size, (void *)bce_mixup_bwd_kernel<scalar_t>,
            0, 0);
        const unsigned int blocks_x = (seq_len + block_size - 1) / block_size;
        const dim3 blocks(blocks_x, batch_size);
        bce_mixup_bwd_kernel<scalar_t><<<blocks, block_size>>>(
            logits.data_ptr<scalar_t>(), labels.data_ptr<scalar_t>(),
            grad_output.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
            mixup_prob, mixup_alpha, pos_weight, batch_size, seq_len, seed);
      }));

  return output;
}

torch::Tensor select_unmixed(const torch::Tensor &input, const float mixup_prob,
                             const float mixup_alpha, const int64_t seed) {
  TORCH_CHECK(input.dim() >= 2, "Input must be >=2D tensor");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_weights", &get_weights, "Get MixUp weights (CUDA)");
  m.def("is_mixed", &is_mixed, "Get mask of MixUp batches (CUDA)");
  m.def("mixup", &mixup, "MixUp operation (CUDA)");
  m.def("cross_entropy_mixup_fwd", &cross_entropy_mixup_fwd,
        "Cross-entropy with MixUp (CUDA)");
  m.def("cross_entropy_mixup_bwd", &cross_entropy_mixup_bwd,
        "Backward pass for Cross-entropy with MixUp (CUDA)");
  m.def("bce_mixup_fwd", &bce_mixup_fwd, "BCE with MixUp (CUDA)");
  m.def("bce_mixup_bwd", &bce_mixup_bwd,
        "Backward pass for BCE with MixUp (CUDA)");
  m.attr("UNKNOWN_LABEL") = py::cast(UNKNOWN_LABEL);
}
