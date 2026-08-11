#include "kvarn_decode_xe2.h"

#include <ATen/xpu/XPUContext.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUStream.h>

#include <cmath>
#include <cstdlib>
#include <limits>

#include "kvarn_decode.hpp"

namespace {

constexpr int kHeadDim = 256;
constexpr int kGroup = 128;
constexpr int kQueryHeads = 24;
constexpr int kKVHeads = 4;
constexpr int kPackedBytes = kHeadDim * kGroup / 2;
constexpr int kKSColOffset = kPackedBytes;
constexpr int kKZpOffset = kKSColOffset + kHeadDim * 2;
constexpr int kKSRowOffset = kKZpOffset + kHeadDim * 2;
constexpr int kVPackedOffset = kKSRowOffset + kGroup * 2;
constexpr int kVSColOffset = kVPackedOffset + kPackedBytes;
constexpr int kVSRowOffset = kVSColOffset + kHeadDim * 2;
constexpr int kVZpOffset = kVSRowOffset + kGroup * 2;
constexpr int kRecordBytes = kVZpOffset + kGroup * 2;

int native_split_count(int64_t max_seq_len) {
  auto const* value = std::getenv("KVARN_NATIVE_XPU_SPLITS");
  if (value == nullptr) return 1;
  int splits = std::atoi(value);
  TORCH_CHECK(
      splits == 1 || splits == 2 || splits == 4 || splits == 8 ||
          splits == 16 || splits == 17 || splits == 24 || splits == 32,
      "KVARN_NATIVE_XPU_SPLITS must be one of 1, 2, 4, 8, 16, 17, 24, or 32");
  // Do not launch more splits than the maximum sequence has 64-token work
  // units. Apart from wasting work, empty split partials amplify reduction
  // drift over many short-context model layers. The one-split direct-output
  // path is both faster and more accurate in this regime.
  int64_t const kv_tiles = (max_seq_len + 63) / 64;
  if (splits > 1 && kv_tiles < splits) return 1;
  return splits;
}

void check_xpu(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_xpu(), name, " must be an XPU tensor");
}

}  // namespace

std::tuple<at::Tensor, at::Tensor>
kvarn_fragment_coords_xe2(const at::Tensor& device_anchor) {
  TORCH_CHECK(device_anchor.is_xpu(), "device_anchor must be on XPU");

  using Config = KVarNDecodeD256G128Config;
  Config::TiledMMAQK mma_qk{};
  Config::TiledMMAPV mma_pv{};
  auto qk_slice = mma_qk.get_slice(0);
  auto pv_slice = mma_pv.get_slice(0);
  auto c_k = cute::make_identity_tensor(
      cute::select<1, 2>(Config::TileShapeQK{}));  // (token, dimension)
  auto c_v = cute::make_identity_tensor(
      cute::select<1, 2>(Config::TileShapePV{}));  // (dimension, token)
  auto k_fragment = qk_slice.partition_sg_fragment_B(c_k);
  auto v_fragment = pv_slice.partition_sg_fragment_B(c_v);

  constexpr int kSubgroupSize = cute::intel::sg_size;
  int const k_slots = k_fragment.size();
  int const v_slots = v_fragment.size();
  TORCH_CHECK(k_slots == 64, "unexpected K DPAS fragment size: ", k_slots);
  TORCH_CHECK(v_slots == 32, "unexpected V DPAS fragment size: ", v_slots);

  auto options =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
  auto k_coords = torch::empty({kSubgroupSize, k_slots, 2}, options);
  auto v_coords = torch::empty({kSubgroupSize, v_slots, 2}, options);
  auto* k_out = k_coords.data_ptr<std::int32_t>();
  auto* v_out = v_coords.data_ptr<std::int32_t>();
  for (int lane = 0; lane < kSubgroupSize; ++lane) {
    for (int slot = 0; slot < k_slots; ++slot) {
      auto coord = k_fragment.tv_layout()(lane, slot);
      auto offset = (lane * k_slots + slot) * 2;
      k_out[offset] = int(cute::get<0>(coord));
      k_out[offset + 1] = int(cute::get<1>(coord));
    }
    for (int slot = 0; slot < v_slots; ++slot) {
      auto coord = v_fragment.tv_layout()(lane, slot);
      auto offset = (lane * v_slots + slot) * 2;
      v_out[offset] = int(cute::get<0>(coord));
      v_out[offset + 1] = int(cute::get<1>(coord));
    }
  }
  return {std::move(k_coords), std::move(v_coords)};
}

void kvarn_decode_with_scratch_xe2(
    const at::Tensor& query,
    const at::Tensor& packed_cache,
    const at::Tensor& block_table,
    const at::Tensor& seq_lens,
    const at::Tensor& block_to_slot,
    const at::Tensor& tail_key,
    const at::Tensor& tail_value,
    at::Tensor& temp_output,
    at::Tensor& exp_sums,
    at::Tensor& max_logits,
    at::Tensor& output,
    int64_t max_seq_len,
    double softmax_scale) {
  check_xpu(query, "query");
  check_xpu(packed_cache, "packed_cache");
  check_xpu(block_table, "block_table");
  check_xpu(seq_lens, "seq_lens");
  check_xpu(block_to_slot, "block_to_slot");
  check_xpu(tail_key, "tail_key");
  check_xpu(tail_value, "tail_value");
  check_xpu(temp_output, "temp_output");
  check_xpu(exp_sums, "exp_sums");
  check_xpu(max_logits, "max_logits");
  check_xpu(output, "output");
  TORCH_CHECK(
      query.device() == packed_cache.device() &&
          query.device() == block_table.device() &&
          query.device() == seq_lens.device() &&
          query.device() == block_to_slot.device() &&
          query.device() == tail_key.device() &&
          query.device() == tail_value.device() &&
          query.device() == temp_output.device() &&
          query.device() == exp_sums.device() &&
          query.device() == max_logits.device() &&
          query.device() == output.device(),
      "all KVarN decode tensors must be on the same XPU");

  TORCH_CHECK(
      query.scalar_type() == at::kHalf, "query must have dtype float16");
  TORCH_CHECK(
      output.scalar_type() == at::kHalf, "output must have dtype float16");
  TORCH_CHECK(
      packed_cache.scalar_type() == at::kByte,
      "packed_cache must have dtype uint8");
  TORCH_CHECK(
      block_table.scalar_type() == at::kInt,
      "block_table must have dtype int32");
  TORCH_CHECK(
      seq_lens.scalar_type() == at::kInt, "seq_lens must have dtype int32");
  TORCH_CHECK(
      block_to_slot.scalar_type() == at::kInt,
      "block_to_slot must have dtype int32");
  TORCH_CHECK(
      tail_key.scalar_type() == at::kHalf &&
          tail_value.scalar_type() == at::kHalf,
      "tail_key and tail_value must have dtype float16");

  TORCH_CHECK(
      query.dim() == 3 && query.size(1) == kQueryHeads &&
          query.size(2) == kHeadDim,
      "query must have shape [B, 24, 256]");
  const int64_t batch = query.size(0);
  TORCH_CHECK(
      batch >= 1 && batch <= 12, "query batch size must be between 1 and 12");
  TORCH_CHECK(
      output.sizes() == query.sizes(),
      "output must have the same [B, 24, 256] shape as query");
  TORCH_CHECK(
      query.is_contiguous() && output.is_contiguous(),
      "query and output must be contiguous");
  TORCH_CHECK(!query.is_alias_of(output), "query and output must not alias");

  TORCH_CHECK(
      packed_cache.dim() == 3 && packed_cache.size(1) == kKVHeads,
      "packed_cache must have shape [num_blocks, 4, record_bytes]");
  TORCH_CHECK(
      packed_cache.size(0) > 0 && packed_cache.size(2) >= kRecordBytes,
      "packed_cache must contain at least one block and each K4V4 record must "
      "contain at least ",
      kRecordBytes,
      " bytes");
  TORCH_CHECK(
      packed_cache.is_contiguous(),
      "packed_cache must be contiguous with one record per KV head");
  TORCH_CHECK(
      packed_cache.size(2) % alignof(std::uint32_t) == 0,
      "packed_cache record stride must preserve uint32 alignment");

  TORCH_CHECK(
      block_table.dim() == 2 && block_table.size(0) == batch &&
          block_table.size(1) > 0,
      "block_table must have shape [B, max_pages_per_seq]");
  TORCH_CHECK(block_table.is_contiguous(), "block_table must be contiguous");
  TORCH_CHECK(
      seq_lens.dim() == 1 && seq_lens.size(0) == batch &&
          seq_lens.is_contiguous(),
      "seq_lens must be contiguous [B] int32");
  TORCH_CHECK(
      block_to_slot.dim() == 1 &&
          block_to_slot.size(0) >= packed_cache.size(0) &&
          block_to_slot.is_contiguous(),
      "block_to_slot must be contiguous int32 and cover all cache blocks");
  TORCH_CHECK(
      tail_key.dim() == 4 && tail_value.dim() == 4 &&
          tail_key.sizes() == tail_value.sizes() && tail_key.size(0) > 0 &&
          tail_key.size(1) == kGroup && tail_key.size(2) == kKVHeads &&
          tail_key.size(3) == kHeadDim && tail_key.is_contiguous() &&
          tail_value.is_contiguous(),
      "tail_key and tail_value must be contiguous [slots, 128, 4, 256]");
  TORCH_CHECK(
      block_table.size(1) <= std::numeric_limits<int>::max(),
      "block_table is too wide for the native kernel");
  TORCH_CHECK(
      max_seq_len > 0 && max_seq_len <= block_table.size(1) * kGroup &&
          max_seq_len <= std::numeric_limits<int>::max(),
      "max_seq_len must be in [1, max_pages_per_seq * 128]");
  // Do not inspect seq_lens values here: reducing an XPU tensor and calling
  // item() would synchronize every decode layer. Scheduler metadata owns the
  // invariant that each value is in [1, max_seq_len].
  TORCH_CHECK(
      std::isfinite(softmax_scale) && softmax_scale > 0.0,
      "softmax_scale must be finite and positive");

  cutlass::fmha::collective::KVarNK4V4Layout layout{
      static_cast<std::uint8_t const*>(packed_cache.const_data_ptr()),
      packed_cache.stride(0),
      packed_cache.stride(1),
      static_cast<int>(batch),
      0,
      kKSColOffset,
      kKZpOffset,
      kKSRowOffset,
      kVPackedOffset,
      kVSColOffset,
      kVSRowOffset,
      kVZpOffset};
  int const num_kv_splits = native_split_count(max_seq_len);
  TORCH_CHECK(
      temp_output.scalar_type() == at::kHalf && temp_output.dim() == 3 &&
          temp_output.size(0) == batch &&
          temp_output.size(1) >= kQueryHeads * num_kv_splits &&
          temp_output.size(2) == kHeadDim && temp_output.is_contiguous(),
      "temp_output must be contiguous fp16 [B, ",
      kQueryHeads * num_kv_splits,
      ", 256] for the configured split count");
  TORCH_CHECK(
      exp_sums.scalar_type() == at::kFloat &&
          max_logits.scalar_type() == at::kFloat && exp_sums.dim() == 3 &&
          exp_sums.sizes() == max_logits.sizes() && exp_sums.size(0) == batch &&
          exp_sums.size(1) == kQueryHeads &&
          exp_sums.size(2) >= num_kv_splits && exp_sums.is_contiguous() &&
          max_logits.is_contiguous(),
      "exp_sums and max_logits must be contiguous fp32 [B, 24, ",
      num_kv_splits,
      "] for the configured split count");
  TORCH_CHECK(
      !temp_output.is_alias_of(output) && !temp_output.is_alias_of(exp_sums) &&
          !temp_output.is_alias_of(max_logits) &&
          !exp_sums.is_alias_of(max_logits) && !exp_sums.is_alias_of(output) &&
          !max_logits.is_alias_of(output),
      "native KVarN scratch tensors must not alias each other or output");
  kvarn_decode_args_t args{
      query.const_data_ptr(),
      static_cast<std::uint8_t const*>(packed_cache.const_data_ptr()),
      output.data_ptr(),
      temp_output.data_ptr(),
      nullptr,
      nullptr,
      block_table.const_data_ptr<int>(),
      seq_lens.const_data_ptr<int>(),
      block_to_slot.const_data_ptr<int>(),
      tail_key.const_data_ptr(),
      tail_value.const_data_ptr(),
      static_cast<int>(batch),
      static_cast<int>(max_seq_len),
      static_cast<int>(block_table.size(1)),
      num_kv_splits,
      static_cast<float>(softmax_scale),
      layout};

  args.exp_sums = exp_sums.data_ptr<float>();
  args.max_logits = max_logits.data_ptr<float>();
  auto& queue = c10::xpu::getCurrentXPUStream().queue();
  auto const* dpas_layout = std::getenv("KVARN_NATIVE_XPU_DPAS_LAYOUT");
  auto status = dpas_layout != nullptr && std::atoi(dpas_layout) == 1
                    ? KVarNDecodeD256G128DpasConfig::run(queue, args)
                    : KVarNDecodeD256G128Config::run(queue, args);
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "native KVarN decode rejected the validated problem");
}

void kvarn_decode_xe2(
    const at::Tensor& query,
    const at::Tensor& packed_cache,
    const at::Tensor& block_table,
    const at::Tensor& seq_lens,
    const at::Tensor& block_to_slot,
    const at::Tensor& tail_key,
    const at::Tensor& tail_value,
    at::Tensor& output,
    int64_t max_seq_len,
    double softmax_scale) {
  int const num_kv_splits = native_split_count(max_seq_len);
  int64_t const batch = query.size(0);
  auto temp_output = at::empty(
      {batch, kQueryHeads * num_kv_splits, kHeadDim}, query.options());
  auto scratch_options = query.options().dtype(at::kFloat);
  auto exp_sums =
      at::empty({batch, kQueryHeads, num_kv_splits}, scratch_options);
  auto max_logits =
      at::empty({batch, kQueryHeads, num_kv_splits}, scratch_options);
  kvarn_decode_with_scratch_xe2(
      query,
      packed_cache,
      block_table,
      seq_lens,
      block_to_slot,
      tail_key,
      tail_value,
      temp_output,
      exp_sums,
      max_logits,
      output,
      max_seq_len,
      softmax_scale);

  // Multi-split decode consumes these function-local tensors asynchronously
  // in both the main kernel and its reducer. Keep their allocations live on
  // the current stream after this wrapper returns; otherwise the caching
  // allocator may recycle them into the next layer while reduction is still
  // reading the partials and statistics.
  const auto current_stream =
      c10::xpu::getCurrentXPUStream(query.device().index());
  c10::xpu::XPUCachingAllocator::recordStream(
      temp_output.storage().data_ptr(), current_stream);
  c10::xpu::XPUCachingAllocator::recordStream(
      exp_sums.storage().data_ptr(), current_stream);
  c10::xpu::XPUCachingAllocator::recordStream(
      max_logits.storage().data_ptr(), current_stream);
}
