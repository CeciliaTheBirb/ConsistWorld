"""CPU-only contract tests for the released ConsistWorld P-Mem path."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

import wan.modules.moba_attention as moba_attention
from convert_consistworld_checkpoint import convert
from consistworld_runtime.checkpoints import load_full_state_dict, load_consistworld_warm_start
from wan.modules.model_ar import WanModelAR


def _reference_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float())
    weights = torch.softmax(scores / (q.shape[-1] ** 0.5), dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", weights, v.float()).to(q.dtype)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.core = nn.Parameter(torch.ones(2))
        self.mem_key_marker = nn.Parameter(torch.ones(2, 1))


class _TinyWarmStart(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block(), _Block()])


class PMemReleaseTest(unittest.TestCase):
    def setUp(self) -> None:
        self._flash_attention = moba_attention.flash_attention
        moba_attention.flash_attention = _reference_attention

    def tearDown(self) -> None:
        moba_attention.flash_attention = self._flash_attention

    def test_marker_only_affects_memory_ranges(self) -> None:
        torch.manual_seed(7)
        q = torch.randn(1, 4, 2, 4)
        k = torch.randn(1, 4, 2, 4)
        v = torch.randn(1, 4, 2, 4)
        groups = [
            (0, 1, [(0, 1)], []),
            (1, 2, [(1, 2)], []),
            (2, 3, [(0, 1), (2, 3)], [(1, 2)]),
            (3, 4, [(0, 1), (3, 4)], [(1, 2)]),
        ]

        base = moba_attention.moba_self_attention_groups(q, k, v, groups)
        zero = moba_attention.moba_self_attention_groups(
            q, k, v, groups, torch.zeros(2, 4)
        )
        marked = moba_attention.moba_self_attention_groups(
            q, k, v, groups, torch.full((2, 4), 0.25)
        )

        self.assertTrue(torch.equal(base, zero))
        self.assertTrue(torch.equal(base[:, :2], marked[:, :2]))
        self.assertFalse(torch.equal(base[:, 2:], marked[:, 2:]))

    def test_layout_separates_retrieval_from_rolling_history(self) -> None:
        model_stub = object.__new__(WanModelAR)
        groups = model_stub._mv_self_groups(
            num_views=2,
            s_tok=4,
            hv_tok=12,
            tv_tok=12,
            chunk_tok=4,
            tgt_abs0=[0, 1, 2],
            use_bi=False,
            window_chunks=1,
            sink_chunks=0,
            mem_hist_sets=[[[], []], [[], []], [[(0, 0)], [(1, 0)]]],
        )
        self.assertTrue(any(memory for _, _, _, memory in groups))
        for _, _, ordinary, memory in groups:
            self.assertFalse(set(ordinary).intersection(memory))

        with self.assertRaisesRegex(ValueError, "P-Mem leak/range"):
            model_stub._mv_self_groups(
                num_views=1,
                s_tok=4,
                hv_tok=12,
                tv_tok=12,
                chunk_tok=4,
                tgt_abs0=[2],
                use_bi=False,
                window_chunks=1,
                sink_chunks=0,
                mem_hist_sets=[
                    [[(0, 1)]],
                ],
            )

    def test_warm_start_and_conversion_preserve_marker_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            warm_start = _TinyWarmStart()
            old_state = {
                key: value
                for key, value in warm_start.state_dict().items()
                if "mem_key_marker" not in key
            }
            old_path = root / "old_stage1.pt"
            torch.save(old_state, old_path)
            initialized = load_consistworld_warm_start(warm_start, str(old_path))
            self.assertEqual(len(initialized), 2)
            for name, parameter in warm_start.named_parameters():
                if "mem_key_marker" in name:
                    self.assertEqual(torch.count_nonzero(parameter).item(), 0)

            reference = {
                "blocks.0.self_attn.q.weight": torch.zeros(8, 8),
                "blocks.0.core": torch.ones(1),
                "blocks.1.self_attn.q.weight": torch.zeros(8, 8),
                "blocks.1.core": torch.ones(1),
            }
            source = dict(reference)
            source["blocks.0.mem_key_marker"] = torch.ones(2, 4)
            source["blocks.1.mem_key_marker"] = torch.ones(2, 4)
            source["blocks.0.mem_proj.weight"] = torch.zeros(8, 8)
            reference_path = root / "reference.pt"
            source_path = root / "source.pt"
            output_path = root / "converted.pt"
            torch.save(reference, reference_path)
            torch.save(source, source_path)
            convert(source_path, reference_path, output_path)
            converted = load_full_state_dict(str(output_path))

            self.assertEqual(
                set(converted),
                set(reference) | {"blocks.0.mem_key_marker", "blocks.1.mem_key_marker"},
            )


if __name__ == "__main__":
    unittest.main()
