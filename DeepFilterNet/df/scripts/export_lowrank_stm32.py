"""Export low-rank STM32-compatible DF decoder ONNX models.

This script is an experimental size-reduction path for the STM32 DF decoder
export. It keeps the STM32 one-frame/stateful interface from export.py, but
factorizes the manually expanded GRU gate matrices with SVD to reduce flash.

Example:
    python DeepFilterNet/df/scripts/export_lowrank_stm32.py tmp/export \
        --model-base-dir DeepFilterNet2 --rank 72 --rank 64
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from df.enhance import init_df
from df.scripts.export import (
    STM32DfDecoderStep4D,
    export_impl,
    fuse_static_convp_pad,
    set_static_onnx_shapes,
)


class LowRankSTM32DfDecoderStep4D(STM32DfDecoderStep4D):
    """STM32 DF decoder with low-rank GRU gate matrix approximations."""

    def __init__(self, df_dec: torch.nn.Module, rank: int):
        self.low_rank = int(rank)
        super().__init__(df_dec, gru_layers=df_dec.df_gru.gru.num_layers)
        self._register_low_rank_gate_buffers()

    def _register_lr(self, name: str, weight: Tensor) -> None:
        with torch.no_grad():
            u, s, vh = torch.linalg.svd(weight.float(), full_matrices=False)
            rank = min(self.low_rank, s.numel())
            a = (u[:, :rank] * s[:rank].sqrt().unsqueeze(0)).to(weight.dtype).contiguous()
            b = (s[:rank].sqrt().unsqueeze(1) * vh[:rank, :]).to(weight.dtype).contiguous()
        self.register_buffer(f"{name}_a", a.clone())
        self.register_buffer(f"{name}_b", b.clone())

    def _register_low_rank_gate_buffers(self) -> None:
        gru = self.df_dec.df_gru.gru
        for layer_idx in range(self.stm32_gru_layers):
            for prefix in ("ih", "hh"):
                weight = getattr(gru, f"weight_{prefix}_l{layer_idx}").detach()
                for gate, gate_weight in zip(("r", "z", "n"), weight.chunk(3, 0)):
                    self._register_lr(f"lr_l{layer_idx}_w_{prefix}_{gate}", gate_weight)

    def _lr_linear(self, x: Tensor, name: str, bias: Tensor) -> Tensor:
        x = torch.nn.functional.linear(x, getattr(self, f"{name}_b"))
        return torch.nn.functional.linear(x, getattr(self, f"{name}_a"), bias)

    def _manual_gru_layer(
        self, x: Tensor, layer_idx: int, h0: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        gru = self.df_dec.df_gru.gru
        if h0 is None:
            h = torch.zeros((1, 1, gru.hidden_size), dtype=x.dtype, device=x.device)
        else:
            h = h0

        outputs = []
        for frame_idx in range(x.shape[0]):
            x_t = x[frame_idx : frame_idx + 1]
            i_r = self._lr_linear(
                x_t, f"lr_l{layer_idx}_w_ih_r", getattr(self, f"gru_l{layer_idx}_b_ih_r")
            )
            i_z = self._lr_linear(
                x_t, f"lr_l{layer_idx}_w_ih_z", getattr(self, f"gru_l{layer_idx}_b_ih_z")
            )
            i_n = self._lr_linear(
                x_t, f"lr_l{layer_idx}_w_ih_n", getattr(self, f"gru_l{layer_idx}_b_ih_n")
            )
            h_r = self._lr_linear(
                h, f"lr_l{layer_idx}_w_hh_r", getattr(self, f"gru_l{layer_idx}_b_hh_r")
            )
            h_z = self._lr_linear(
                h, f"lr_l{layer_idx}_w_hh_z", getattr(self, f"gru_l{layer_idx}_b_hh_z")
            )
            h_n = self._lr_linear(
                h, f"lr_l{layer_idx}_w_hh_n", getattr(self, f"gru_l{layer_idx}_b_hh_n")
            )

            resetgate = torch.sigmoid(i_r + h_r)
            updategate = torch.sigmoid(i_z + h_z)
            newgate = torch.tanh(i_n + resetgate * h_n)
            h = newgate + updategate * (h - newgate)
            outputs.append(h)
        return torch.cat(outputs, 0), h


def freeze(module: torch.nn.Module) -> torch.nn.Module:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)
    return module


def create_validation_npz(export_dir: Path, rank: int) -> Path:
    inputs = np.load(export_dir / "df_dec_stm32_input.npz")
    outputs = np.load(export_dir / f"df_dec_stm32_lowrank_r{rank}_output.npz")
    path = export_dir / f"df_dec_stm32_lowrank_r{rank}_val_io.npz"
    np.savez(
        path,
        m_inputs_1=inputs["emb"].transpose(0, 1, 3, 2).astype("float32"),
        m_inputs_2=inputs["c0"].astype("float32"),
        m_inputs_3=inputs["h0"].astype("float32"),
        m_outputs_1=outputs["coefs"].astype("float32"),
        m_outputs_2=outputs["alpha"].astype("float32"),
        m_outputs_3=outputs["h1"].astype("float32"),
    )
    return path


def export_rank(
    model: torch.nn.Module,
    export_dir: Path,
    rank: int,
    opset: int,
    check: bool,
    simplify: bool,
) -> None:
    data = np.load(export_dir / "df_dec_stm32_input.npz")
    emb = torch.from_numpy(data["emb"]).float()
    c0 = torch.from_numpy(data["c0"]).float()
    h0 = torch.from_numpy(data["h0"]).float()

    wrapper = freeze(LowRankSTM32DfDecoderStep4D(model.df_dec, rank=rank))
    path = export_dir / f"df_dec_stm32_lowrank_r{rank}.onnx"
    coefs, alpha, h1 = export_impl(
        str(path),
        wrapper,
        inputs=(emb, c0, h0),
        input_names=["emb", "c0", "h0"],
        output_names=["coefs", "alpha", "h1"],
        dynamic_axes={},
        jit=False,
        check=check,
        simplify=simplify,
        opset_version=opset,
        print_graph=False,
    )
    fuse_static_convp_pad(str(path), int(wrapper.df_convp_tpad.shape[2]))
    set_static_onnx_shapes(
        str(path),
        (emb, c0, h0, coefs, alpha, h1),
        ["emb", "c0", "h0", "coefs", "alpha", "h1"],
    )
    np.savez_compressed(
        export_dir / f"df_dec_stm32_lowrank_r{rank}_output.npz",
        coefs=coefs.detach().cpu().numpy(),
        alpha=alpha.detach().cpu().numpy(),
        h1=h1.detach().cpu().numpy(),
    )
    val_path = create_validation_npz(export_dir, rank)
    print(f"Exported {path}")
    print(f"Saved validation data {val_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_dir", type=Path, help="Directory containing df_dec_stm32_input.npz")
    parser.add_argument("--model-base-dir", default="DeepFilterNet2")
    parser.add_argument("--rank", action="append", type=int)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--no-check", action="store_false", dest="check")
    parser.add_argument("--simplify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.export_dir.mkdir(parents=True, exist_ok=True)
    model, _, _, _ = init_df(args.model_base_dir, log_level="ERROR", log_file=None)
    for rank in args.rank or [72]:
        export_rank(model, args.export_dir, rank, args.opset, args.check, args.simplify)


if __name__ == "__main__":
    main()
