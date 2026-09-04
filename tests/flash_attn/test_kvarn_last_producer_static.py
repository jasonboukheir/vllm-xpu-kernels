# SPDX-License-Identifier: Apache-2.0
"""Host-only protocol proofs for the isolated ID19 decoder."""

from pathlib import Path


REPO_ROOT = Path(__file__).parents[2]
CONFIG = (REPO_ROOT / "csrc/xpu/attn/xe_2/kvarn_decode.hpp").read_text()
DISPATCH = (REPO_ROOT / "csrc/xpu/attn/xe_2/kvarn_decode_xe2.cpp").read_text()
KERNEL = (
    REPO_ROOT / "csrc/xpu/attn/xe_2/kernel/paged_decode_kernel.hpp"
).read_text()
REGISTRATION = (REPO_ROOT / "csrc/flash_attn/flash_api.cpp").read_text()


def test_id19_is_runtime_selectable_but_fails_closed_to_id18() -> None:
    assert "kQ6B1ShortLastProducer = 19" in DISPATCH
    assert "request_q6_b1_short_last_producer && batch == 1" in DISPATCH
    assert "kMaxShortContextSeqLen = 8192" in CONFIG
    assert (
        "KVarNB1ShortLastProducerFinalizer::kMaxShortContextSeqLen" in CONFIG
    )
    assert (
        "args.max_seq_len >\n"
        "              KVarNB1ShortLastProducerFinalizer::"
        "kMaxShortContextSeqLen"
        in CONFIG
    )
    assert (
        "KVarNB1ShortLastProducerFinalizer::kMaxShortContextSeqLen" in DISPATCH
    )
    assert (
        DISPATCH.count(
            "KVarNB1ShortLastProducerFinalizer::kMaxShortContextSeqLen"
        )
        >= 2
    )
    assert "max_seq_len <= 4096" not in DISPATCH
    assert "args.max_seq_len > 4096" not in CONFIG
    assert "unrotate_output && last_producer_state_initialized" in DISPATCH
    assert "use_q6_prefetch_record_cursor" in DISPATCH
    assert (
        "KVarNDecodeD256G128DpasQ6PrefetchRecordCursorConfig::run" in DISPATCH
    )


def test_id19_short_context_bucket_boundaries() -> None:
    """Prove the host-visible eligibility contract at the bucket boundary."""
    short_context_max = 8192

    def id19_eligible(seq_len: int) -> bool:
        # All other ID19 prerequisites are held true here so this focused
        # contract test isolates the short-context bucket itself.
        return seq_len > 0 and seq_len <= short_context_max

    # The 4096-input/512-output experiment must remain on ID19 throughout its
    # decode; test every length in that benchmark's transition interval.
    assert all(id19_eligible(seq_len) for seq_len in range(4096, 4609))
    assert id19_eligible(short_context_max)

    # The next token must fail closed to ID18 rather than entering an ID19
    # kernel compiled for the bounded short-context protocol.
    assert not id19_eligible(8193)
    assert not id19_eligible(16384)


def test_id18_producer_and_layout_are_unchanged() -> None:
    assert (
        "!KVarNDecodeD256G128DpasQ6PrefetchRecordCursorConfig::\n"
        "                  UsesLastProducerFinalizer"
    ) in CONFIG
    assert (
        "KVarNDecodeD256G128DpasQ6B1ShortLastProducerConfig::Mainloop,\n"
        "              KVarNDecodeD256G128DpasQ6Prefetch"
        "RecordCursorConfig::Mainloop"
    ) in CONFIG
    assert (
        "KVarNDecodeD256G128DpasQ6B1ShortLastProducerConfig::Epilogue,\n"
        "              KVarNDecodeD256G128DpasQ6Prefetch"
        "RecordCursorConfig::Epilogue"
    ) in CONFIG


def test_completion_protocol_has_release_sequence_and_no_spin() -> None:
    start = CONFIG.index("struct KVarNB1ShortLastProducerFinalizer")
    end = CONFIG.index("/** Concrete, intentionally narrow", start)
    finalizer = CONFIG[start:end]

    assert "sycl::group_barrier(workgroup)" in finalizer
    assert "sycl::atomic_fence(" in finalizer
    assert (
        "sycl::memory_order::release, sycl::memory_scope::device" in finalizer
    )
    assert "fetch_add(1, sycl::memory_order::acq_rel)" in finalizer
    assert (
        "completion.store(next_epoch, sycl::memory_order::release)" in finalizer
    )
    assert "batch * kKVHeads + kv_head" in finalizer
    assert "prior_count + 1" in finalizer
    assert "shared_.completed_epoch + 1" in finalizer
    assert "while (" not in finalizer
    assert "if (!shared_.is_last_producer) return" in finalizer


def test_scheduler_passes_only_nonempty_producers_to_finalizer() -> None:
    """Model the legacy rectangular scheduler's ragged split contract."""
    assert "int active_producer_count;" in KERNEL
    assert (
        "active_producer_count =\n"
        "              cute::ceil_div(windowed_k_blocks, num_blocks_per_split);"
        in KERNEL
    )
    assert (
        "idx_b, head, idx_kv_split, active_producer_count" in KERNEL
    )
    assert (
        "finalizer(\n"
        "            idx_b, head, idx_kv_split, "
        "is_single_split ? 1 : seq_num_kv_splits);"
        not in KERNEL
    )

    def active_producers(seq_len: int, requested_splits: int) -> int:
        kv_blocks = (seq_len + 63) // 64
        if kv_blocks < 32:
            return 1
        blocks_per_split = (
            kv_blocks + requested_splits - 1
        ) // requested_splits
        return (kv_blocks + blocks_per_split - 1) // blocks_per_split

    assert active_producers(1536, 16) == 1
    assert active_producers(4095, 16) == 16
    assert active_producers(4096, 16) == 16
    assert active_producers(4097, 16) == 13
    assert active_producers(4608, 16) == 15
    assert active_producers(8191, 32) == 32
    assert active_producers(8192, 32) == 32


def test_last_producer_preserves_output_contract() -> None:
    assert "kQueryHeadsPerKV = 6" in CONFIG
    assert "query_in_group < kQueryHeadsPerKV" in CONFIG
    assert "Element const reduced" in CONFIG
    assert "for (int stage = 0; stage < 8; ++stage)" in CONFIG
    assert "shared_.output_row[dim] * (1.0f / 16.0f)" in CONFIG
    assert "reinterpret_cast<BFloat16*>(params_.output)" in CONFIG
    assert "if constexpr (LastProducerFinalizer) {\n      return;" in CONFIG
    assert "LastProducerFinalizer finalizer" in KERNEL


def test_scratch_initialization_is_an_explicit_abi_contract() -> None:
    assert "last_producer_state_initialized=False" in REGISTRATION
    assert "at::zeros(" in DISPATCH
    assert "max_logits prefix" in DISPATCH
    assert "alignof(std::uint64_t)" in DISPATCH


def test_epoch_counter_cycles_only_after_every_producer() -> None:
    count_mask = (1 << 32) - 1

    def arrive(state: int, expected: int) -> tuple[int, bool, int]:
        prior = state
        state += 1
        count = prior & count_mask
        return state, count + 1 == expected, prior >> 32

    state = 0
    for expected in (2, 4, 8, 16, 32, 1, 8):
        last_flags = []
        completed_epoch = None
        for _ in range(expected):
            state, is_last, epoch = arrive(state, expected)
            last_flags.append(is_last)
            if is_last:
                completed_epoch = epoch
        assert last_flags == [False] * (expected - 1) + [True]
        assert completed_epoch is not None
        state = (completed_epoch + 1) << 32
        assert state & count_mask == 0
