# SPDX-License-Identifier: Apache-2.0
"""CPU-only validation for ragged native KVarN benchmark accounting."""

import pytest

from benchmark.benchmark_kvarn_decode import (nonempty_split_workgroups,
                                              parse_seq_lens)


def test_parse_ragged_lengths() -> None:
    assert parse_seq_lens("128,6000,65,1", 4, 6000) == [128, 6000, 65, 1]


@pytest.mark.parametrize("value", ["1,2,3", "0,2,3,4", "1,2,3,6001"])
def test_invalid_ragged_lengths(value: str) -> None:
    with pytest.raises(ValueError):
        parse_seq_lens(value, 4, 6000)


def test_nonempty_workgroup_accounting_uses_grid_max_partition() -> None:
    # context=6000 => 94 KV tiles => six tiles per split. The four rows need
    # 1, 16, 1, and 1 splits respectively, each replicated across four KV heads.
    assert nonempty_split_workgroups([128, 6000, 65, 1], 6000) == 4 * 19
