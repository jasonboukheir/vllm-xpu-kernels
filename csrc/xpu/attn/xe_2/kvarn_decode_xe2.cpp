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
  int splits = value == nullptr ? 16 : std::atoi(value);
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
    double softmax_scale,
    bool unrotate_output,
    bool write_bf16_output) {
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
      write_bf16_output ? output.scalar_type() == at::kBFloat16
                        : output.scalar_type() == at::kHalf,
      "output must have dtype ",
      write_bf16_output ? "bfloat16" : "float16");
  TORCH_CHECK(
      !write_bf16_output || unrotate_output,
      "write_bf16_output requires unrotate_output");
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
      !unrotate_output || num_kv_splits > 1,
      "unrotate_output requires a multi-split KVarN decode");
  // Multi-split reducers index caller-owned scratch with raw packed strides,
  // so an oversized split dimension would silently change every following
  // row's address. The one-split direct-output path does not consume scratch
  // and may safely reuse the service's larger persistent allocation.
  bool const temp_split_extent_valid =
      temp_output.dim() == 3 &&
      (num_kv_splits == 1 ? temp_output.size(1) >= kQueryHeads
                          : temp_output.size(1) == kQueryHeads * num_kv_splits);
  TORCH_CHECK(
      temp_output.scalar_type() == at::kHalf && temp_output.dim() == 3 &&
          temp_output.size(0) == batch && temp_split_extent_valid &&
          temp_output.size(2) == kHeadDim && temp_output.is_contiguous(),
      "temp_output must be contiguous fp16 [B, ",
      kQueryHeads * num_kv_splits,
      ", 256] for the configured split count");
  bool const stats_split_extent_valid =
      exp_sums.dim() == 3 &&
      (num_kv_splits == 1 ? exp_sums.size(2) >= 1
                          : exp_sums.size(2) == num_kv_splits);
  TORCH_CHECK(
      exp_sums.scalar_type() == at::kFloat &&
          max_logits.scalar_type() == at::kFloat && exp_sums.dim() == 3 &&
          exp_sums.sizes() == max_logits.sizes() && exp_sums.size(0) == batch &&
          exp_sums.size(1) == kQueryHeads && stats_split_extent_valid &&
          exp_sums.is_contiguous() && max_logits.is_contiguous(),
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
      unrotate_output,
      write_bf16_output,
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
    double softmax_scale,
    bool unrotate_output,
    bool write_bf16_output) {
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
      softmax_scale,
      unrotate_output,
      write_bf16_output);

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

void kvarn_materialize_packed_kv_xe2(
    const at::Tensor& packed_cache,
    const at::Tensor& block_table,
    const at::Tensor& seq_lens,
    const at::Tensor& cu_seqlens_k,
    const at::Tensor& block_to_slot,
    const at::Tensor& tail_key,
    const at::Tensor& tail_value,
    at::Tensor& key_output,
    at::Tensor& value_output,
    int64_t max_seq_len) {
  for (auto const& item :
       {std::pair<at::Tensor const*, char const*>{
            &packed_cache, "packed_cache"},
        {&block_table, "block_table"},
        {&seq_lens, "seq_lens"},
        {&cu_seqlens_k, "cu_seqlens_k"},
        {&block_to_slot, "block_to_slot"},
        {&tail_key, "tail_key"},
        {&tail_value, "tail_value"},
        {&key_output, "key_output"},
        {&value_output, "value_output"}}) {
    check_xpu(*item.first, item.second);
    TORCH_CHECK(
        item.first->device() == packed_cache.device(),
        item.second,
        " must be on the packed-cache device");
  }
  int64_t const batch = seq_lens.numel();
  TORCH_CHECK(
      packed_cache.scalar_type() == at::kByte && packed_cache.dim() == 3 &&
          packed_cache.size(1) == kKVHeads &&
          packed_cache.size(2) >= kRecordBytes && packed_cache.is_contiguous(),
      "packed_cache must be contiguous uint8 [num_blocks, 4, record_bytes]");
  TORCH_CHECK(
      block_table.scalar_type() == at::kInt && block_table.dim() == 2 &&
          block_table.size(0) == batch && block_table.is_contiguous(),
      "block_table must be contiguous int32 [B, max_blocks]");
  TORCH_CHECK(
      seq_lens.scalar_type() == at::kInt && seq_lens.dim() == 1 &&
          seq_lens.is_contiguous() && cu_seqlens_k.scalar_type() == at::kInt &&
          cu_seqlens_k.dim() == 1 && cu_seqlens_k.numel() == batch + 1 &&
          cu_seqlens_k.is_contiguous(),
      "seq_lens and cu_seqlens_k must be contiguous int32 [B] and [B+1]");
  TORCH_CHECK(
      block_to_slot.scalar_type() == at::kInt && block_to_slot.dim() == 1 &&
          block_to_slot.size(0) >= packed_cache.size(0) &&
          block_to_slot.is_contiguous(),
      "block_to_slot must cover every physical block");
  TORCH_CHECK(
      tail_key.scalar_type() == at::kHalf &&
          tail_value.sizes() == tail_key.sizes() && tail_key.dim() == 4 &&
          tail_key.size(1) == kGroup && tail_key.size(2) == kKVHeads &&
          tail_key.size(3) == kHeadDim && tail_key.is_contiguous() &&
          tail_value.is_contiguous(),
      "tail pools must be contiguous fp16 [slots, 128, 4, 256]");
  TORCH_CHECK(
      key_output.scalar_type() == at::kHalf &&
          value_output.sizes() == key_output.sizes() && key_output.dim() == 3 &&
          key_output.size(1) == kKVHeads && key_output.size(2) == kHeadDim &&
          key_output.is_contiguous() && value_output.is_contiguous(),
      "outputs must be contiguous fp16 [tokens, 4, 256]");
  TORCH_CHECK(
      max_seq_len >= 1 && max_seq_len <= block_table.size(1) * kGroup &&
          batch <= std::numeric_limits<int>::max() &&
          block_table.size(1) <= std::numeric_limits<int>::max(),
      "invalid materialization extent");

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
  cutlass::fmha::collective::KVarNHybridTailLayout tail{
      block_to_slot.const_data_ptr<int>(),
      static_cast<cutlass::half_t const*>(tail_key.const_data_ptr()),
      static_cast<cutlass::half_t const*>(tail_value.const_data_ptr()),
      static_cast<int>(tail_key.stride(0)),
      static_cast<int>(tail_key.stride(1)),
      static_cast<int>(tail_key.stride(2))};
  auto const* page_table = block_table.const_data_ptr<int>();
  auto const* lengths = seq_lens.const_data_ptr<int>();
  auto const* cumulative = cu_seqlens_k.const_data_ptr<int>();
  auto* key = static_cast<cutlass::half_t*>(key_output.data_ptr());
  auto* value = static_cast<cutlass::half_t*>(value_output.data_ptr());
  int const max_blocks = static_cast<int>((max_seq_len + kGroup - 1) / kGroup);
  int const table_stride = static_cast<int>(block_table.stride(0));
  bool const dpas = [] {
    auto const* text = std::getenv("KVARN_NATIVE_XPU_DPAS_LAYOUT");
    return text != nullptr && std::atoi(text) == 1;
  }();
  auto& queue = c10::xpu::getCurrentXPUStream().queue();
  queue.parallel_for(
      sycl::nd_range<1>(
          sycl::range<1>(
              static_cast<size_t>(batch) * max_blocks * kKVHeads * 256),
          sycl::range<1>(256)),
      [=](sycl::nd_item<1> item) {
        int const group_id = static_cast<int>(item.get_group_linear_id());
        int const kv_head = group_id % kKVHeads;
        int const logical_block = (group_id / kKVHeads) % max_blocks;
        int const request = group_id / (kKVHeads * max_blocks);
        int const seq_len = lengths[request];
        int const token_base = logical_block * kGroup;
        if (token_base >= seq_len) return;
        int const physical = page_table[request * table_stride + logical_block];
        int const slot = tail.block_to_slot[physical];
        auto const* rec =
            slot < 0
                ? layout.cache + std::int64_t(physical) * layout.block_stride +
                      std::int64_t(kv_head) * layout.head_stride
                : nullptr;
        cutlass::fmha::collective::KVarNK4V4FragmentLoader<> loader{
            layout, tail, page_table, table_stride};
        int const local_id = static_cast<int>(item.get_local_linear_id());
        for (int linear = local_id; linear < kGroup * kHeadDim; linear += 256) {
          int const token = linear / kHeadDim;
          int const dim = linear % kHeadDim;
          if (token_base + token >= seq_len) continue;
          float kval;
          float vval;
          if (slot < 0) {
            float const kq = dpas
                                 ? loader.load_k_dpas_quantized(rec, token, dim)
                                 : loader.load_k_quantized(rec, token, dim);
            float const vq = dpas
                                 ? loader.load_v_dpas_quantized(rec, token, dim)
                                 : loader.load_v_quantized(rec, token, dim);
            float const kcol = decltype(loader)::load_f16(
                rec + layout.k_s_col_offset + 2 * dim);
            float const kzp =
                decltype(loader)::load_f16(rec + layout.k_zp_offset + 2 * dim);
            float const krow = decltype(loader)::load_f16(
                rec + layout.k_s_row_offset + 2 * token);
            float const vcol = decltype(loader)::load_f16(
                rec + layout.v_s_col_offset + 2 * dim);
            float const vrow = decltype(loader)::load_f16(
                rec + layout.v_s_row_offset + 2 * token);
            float const vzp = decltype(loader)::load_f16(
                rec + layout.v_zp_offset + 2 * token);
            kval = (kq * kcol + kzp) * krow;
            vval = (vq * vrow + vzp) * vcol;
          } else {
            kval = loader.load_tail(tail.key, slot, token, kv_head, dim);
            vval = loader.load_tail(tail.value, slot, token, kv_head, dim);
          }
          std::int64_t const out_token =
              cumulative[request] + token_base + token;
          std::int64_t const out_index =
              (out_token * kKVHeads + kv_head) * kHeadDim + dim;
          key[out_index] = static_cast<cutlass::half_t>(kval);
          value[out_index] = static_cast<cutlass::half_t>(vval);
        }
      });
}
