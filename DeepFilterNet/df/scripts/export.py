import os
import shutil
import tarfile
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import onnx
import onnx.checker
import onnx.helper
import onnxruntime as ort
import torch
from loguru import logger
from torch import Tensor

from df.enhance import (
    ModelParams,
    df_features,
    enhance,
    get_model_basedir,
    init_df,
    setup_df_argument_parser,
)
from df.io import get_test_sample, save_audio
from libdf import DF


def shapes_dict(
    tensors: Tuple[Tensor], names: Union[Tuple[str], List[str]]
) -> Dict[str, Tuple[int]]:
    if len(tensors) != len(names):
        logger.warning(
            f"  Number of tensors ({len(tensors)}) does not match provided names: {names}"
        )
    return {k: v.shape for (k, v) in zip(names, tensors)}


def onnx_simplify(
    path: str, input_data: Dict[str, Tensor], input_shapes: Dict[str, Iterable[int]]
) -> str:
    import onnxsim

    model = onnx.load(path)
    model_simp, check = onnxsim.simplify(
        model,
        input_data=input_data,
        test_input_shapes=input_shapes,
    )
    model_n = os.path.splitext(os.path.basename(path))[0]
    assert check, "Simplified ONNX model could not be validated"
    logger.debug(model_n + ": " + onnx.helper.printable_graph(model.graph))
    try:
        onnx.checker.check_model(model_simp, full_check=True)
    except Exception as e:
        logger.error(f"Failed to simplify model {model_n}. Skipping: {e}")
        return path
    # new_path = os.path.join(os.path.dirname(path), model_n + "_simplified.onnx")
    onnx.save_model(model_simp, path)
    return path


def onnx_check(path: str, input_dict: Dict[str, Tensor], output_names: Tuple[str]):
    model = onnx.load(path)
    logger.debug(os.path.basename(path) + ": " + onnx.helper.printable_graph(model.graph))
    onnx.checker.check_model(model, full_check=True)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return sess.run(output_names, {k: v.numpy() for (k, v) in input_dict.items()})


def export_impl(
    path: str,
    model: torch.nn.Module,
    inputs: Tuple[Tensor, ...],
    input_names: List[str],
    output_names: List[str],
    dynamic_axes: Dict[str, Dict[int, str]],
    jit: bool = True,
    opset_version=14,
    check: bool = True,
    simplify: bool = True,
    print_graph: bool = False,
):
    export_dir = os.path.dirname(path)
    if not os.path.isdir(export_dir):
        logger.info(f"Creating export directory: {export_dir}")
        os.makedirs(export_dir)
    model_name = os.path.splitext(os.path.basename(path))[0]
    logger.info(f"Exporting model '{model_name}' to {export_dir}")

    input_shapes = shapes_dict(inputs, input_names)
    logger.info(f"  Input shapes: {input_shapes}")

    outputs = model(*inputs)
    output_shapes = shapes_dict(outputs, output_names)
    logger.info(f"  Output shapes: {output_shapes}")

    if jit:
        model = torch.jit.script(model, example_inputs=[tuple(a for a in inputs)])

    logger.info(f"  Dynamic axis: {dynamic_axes}")
    torch.onnx.export(
        model=deepcopy(model),
        f=path,
        args=inputs,
        input_names=input_names,
        dynamic_axes=dynamic_axes,
        output_names=output_names,
        opset_version=opset_version,
        keep_initializers_as_inputs=False,
    )

    input_dict = {k: v for (k, v) in zip(input_names, inputs)}
    if check:
        onnx_outputs = onnx_check(path, input_dict, tuple(output_names))
        for name, out, onnx_out in zip(output_names, outputs, onnx_outputs):
            try:
                np.testing.assert_allclose(
                    out.numpy().squeeze(), onnx_out.squeeze(), rtol=1e-6, atol=1e-5
                )
            except AssertionError as e:
                logger.warning(f"  Elements not close for {name}: {e}")
    if simplify:
        path = onnx_simplify(path, input_dict, shapes_dict(inputs, input_names))
        logger.info(f"  Saved simplified model {path}")
    if print_graph:
        onnx.helper.printable_graph(onnx.load_model(path).graph)

    return outputs


def set_static_onnx_shapes(path: str, tensors: Tuple[Tensor, ...], names: List[str]) -> None:
    model = onnx.load(path)
    shapes = shapes_dict(tensors, names)
    values = {value.name: value for value in list(model.graph.input) + list(model.graph.output)}
    for name, shape in shapes.items():
        value = values.get(name)
        if value is None:
            continue
        dims = value.type.tensor_type.shape.dim
        for dim, size in zip(dims, shape):
            dim.ClearField("dim_param")
            dim.dim_value = int(size)
    onnx.save_model(model, path)


def fuse_static_convp_pad(path: str, pad_top: int) -> None:
    model = onnx.load(path)
    concat_outputs = {
        node.output[0]: node
        for node in model.graph.node
        if node.op_type == "Concat" and "c0" in node.input
    }
    fused_concat_output = None

    for node in model.graph.node:
        if node.op_type != "Conv" or node.input[0] not in concat_outputs:
            continue
        fused_concat_output = node.input[0]
        node.input[0] = "c0"
        attrs = [attr for attr in node.attribute if attr.name != "pads"]
        attrs.append(onnx.helper.make_attribute("pads", [pad_top, 0, 0, 0]))
        del node.attribute[:]
        node.attribute.extend(attrs)
        break

    if fused_concat_output is not None:
        keep_nodes = [node for node in model.graph.node if node.output[0] != fused_concat_output]
        del model.graph.node[:]
        model.graph.node.extend(keep_nodes)
        onnx.checker.check_model(model)
        onnx.save_model(model, path)


def _use_sequence_first_gru(module: torch.nn.Module) -> None:
    for child in module.modules():
        if hasattr(child, "batch_first"):
            child.batch_first = False


class STM32DfDecoder(torch.nn.Module):
    """
    STM32-compatible DF decoder wrapper.

    Input:
      emb_tbh: [T, B, H]
      c0:      [B, C, T, F]

    Output:
      coefs
      alpha:   [B, T, 1]
    """

    def __init__(self, df_dec: torch.nn.Module, gru_layers: Union[int, None] = None):
        super().__init__()
        self.df_dec = deepcopy(df_dec)
        _use_sequence_first_gru(self.df_dec.df_gru)
        self.stm32_gru_layers = (
            int(gru_layers) if gru_layers is not None else self.df_dec.df_gru.gru.num_layers
        )
        if hasattr(self.df_dec.df_gru, "gru") and hasattr(self.df_dec.df_gru, "linear_in"):
            self._register_gru_gate_buffers()
        self._register_static_convp_pad()

    def _register_static_convp_pad(self) -> None:
        convp_layers = list(self.df_dec.df_convp)
        if not convp_layers or not isinstance(convp_layers[0], torch.nn.ConstantPad2d):
            self.register_buffer("df_convp_tpad", torch.empty(0))
            return

        pad_left, pad_right, pad_top, pad_bottom = convp_layers[0].padding
        if pad_left != 0 or pad_right != 0 or pad_bottom != 0:
            raise NotImplementedError("STM32 DF decoder export only supports causal time padding")

        conv = next(layer for layer in convp_layers if isinstance(layer, torch.nn.Conv2d))
        self.register_buffer(
            "df_convp_tpad",
            torch.zeros((1, conv.in_channels, pad_top, self.df_dec.df_bins)),
        )

    def _register_gru_gate_buffers(self) -> None:
        gru = self.df_dec.df_gru.gru
        for layer_idx in range(self.stm32_gru_layers):
            for prefix in ("ih", "hh"):
                weight = getattr(gru, f"weight_{prefix}_l{layer_idx}").detach()
                bias = getattr(gru, f"bias_{prefix}_l{layer_idx}").detach()
                w_r, w_z, w_n = weight.chunk(3, 0)
                b_r, b_z, b_n = bias.chunk(3, 0)
                self.register_buffer(f"gru_l{layer_idx}_w_{prefix}_r", w_r.clone())
                self.register_buffer(f"gru_l{layer_idx}_w_{prefix}_z", w_z.clone())
                self.register_buffer(f"gru_l{layer_idx}_w_{prefix}_n", w_n.clone())
                self.register_buffer(f"gru_l{layer_idx}_b_{prefix}_r", b_r.clone())
                self.register_buffer(f"gru_l{layer_idx}_b_{prefix}_z", b_z.clone())
                self.register_buffer(f"gru_l{layer_idx}_b_{prefix}_n", b_n.clone())

    def _manual_gru_layer(
        self, x: Tensor, layer_idx: int, h0: Union[Tensor, None] = None
    ) -> Tuple[Tensor, Tensor]:
        gru = self.df_dec.df_gru.gru

        if h0 is None:
            h = torch.zeros((1, 1, gru.hidden_size), dtype=x.dtype, device=x.device)
        else:
            h = h0
        outputs = []
        for frame_idx in range(x.shape[0]):
            x_t = x[frame_idx : frame_idx + 1]
            i_r = torch.nn.functional.linear(
                x_t,
                getattr(self, f"gru_l{layer_idx}_w_ih_r"),
                getattr(self, f"gru_l{layer_idx}_b_ih_r"),
            )
            i_z = torch.nn.functional.linear(
                x_t,
                getattr(self, f"gru_l{layer_idx}_w_ih_z"),
                getattr(self, f"gru_l{layer_idx}_b_ih_z"),
            )
            i_n = torch.nn.functional.linear(
                x_t,
                getattr(self, f"gru_l{layer_idx}_w_ih_n"),
                getattr(self, f"gru_l{layer_idx}_b_ih_n"),
            )
            h_r = torch.nn.functional.linear(
                h,
                getattr(self, f"gru_l{layer_idx}_w_hh_r"),
                getattr(self, f"gru_l{layer_idx}_b_hh_r"),
            )
            h_z = torch.nn.functional.linear(
                h,
                getattr(self, f"gru_l{layer_idx}_w_hh_z"),
                getattr(self, f"gru_l{layer_idx}_b_hh_z"),
            )
            h_n = torch.nn.functional.linear(
                h,
                getattr(self, f"gru_l{layer_idx}_w_hh_n"),
                getattr(self, f"gru_l{layer_idx}_b_hh_n"),
            )

            resetgate = torch.sigmoid(i_r + h_r)
            updategate = torch.sigmoid(i_z + h_z)
            newgate = torch.tanh(i_n + resetgate * h_n)
            h = newgate + updategate * (h - newgate)
            outputs.append(h)
        return torch.cat(outputs, 0), h

    def _manual_squeezed_gru(self, emb_tbh: Tensor) -> Tensor:
        x = self._grouped_linear_static(
            self.df_dec.df_gru.linear_in[0],
            emb_tbh,
            emb_tbh.shape[0],
            emb_tbh.shape[1],
        )
        x = self.df_dec.df_gru.linear_in[1](x)
        gru_in = x
        for layer_idx in range(self.stm32_gru_layers):
            x, _ = self._manual_gru_layer(x, layer_idx)
        if self.df_dec.df_gru.gru_skip is not None:
            x = x + self.df_dec.df_gru.gru_skip(gru_in)
        if isinstance(self.df_dec.df_gru.linear_out, torch.nn.Identity):
            return x
        x = self._grouped_linear_static(
            self.df_dec.df_gru.linear_out[0],
            x,
            x.shape[0],
            x.shape[1],
        )
        return self.df_dec.df_gru.linear_out[1](x)

    def _grouped_linear_static(
        self, layer: torch.nn.Module, x: Tensor, dim0: int, dim1: int
    ) -> Tensor:
        x = x.reshape(dim0, dim1, layer.groups, layer.ws)
        x = torch.einsum("btgi,gih->btgh", x, layer.weight)
        return x.reshape(dim0, dim1, layer.hidden_size)

    def _static_df_convp(self, c0: Tensor) -> Tensor:
        convp_layers = list(self.df_dec.df_convp)
        if convp_layers and isinstance(convp_layers[0], torch.nn.ConstantPad2d):
            if self.df_convp_tpad.numel() > 0:
                c0 = torch.cat((self.df_convp_tpad, c0), dim=2)
            convp_layers = convp_layers[1:]

        for layer in convp_layers:
            c0 = layer(c0)
        return c0

    def forward(self, emb_tbh: Tensor, c0: Tensor) -> Tuple[Tensor, Tensor]:
        t, b, _ = emb_tbh.shape

        # Feed the recurrent path in [T, B, H] without ONNX batch/time Transpose.
        # ST Edge AI 4.0 rejects ONNX GRU fed by an internal rank-3 tensor, so
        # this export expands the GRU math with the original weights.
        if hasattr(self.df_dec.df_gru, "gru") and hasattr(self.df_dec.df_gru, "linear_in"):
            c = self._manual_squeezed_gru(emb_tbh)
        else:
            c, _ = self.df_dec.df_gru(emb_tbh)

        # Convert GRU output back to batch-first for the FC/output layers.
        # The STM32 export uses a fixed B=1 shape, so reshape preserves the
        # exact data order without introducing a batch/time Transpose.
        emb = emb_tbh.reshape(b, t, -1)
        c = c.reshape(b, t, -1)
        if getattr(self.df_dec, "df_skip", None) is not None:
            c = c + self.df_dec.df_skip(emb)

        alpha = self.df_dec.df_fc_a(c)

        if hasattr(self.df_dec, "df_out"):
            # DeepFilterNet2 grouped-linear decoder path.
            c0 = self._static_df_convp(c0).permute(0, 2, 3, 1)
            if hasattr(self.df_dec.df_out[0], "groups"):
                c = self._grouped_linear_static(self.df_dec.df_out[0], c, b, t)
                c = self.df_dec.df_out[1](c)
            else:
                c = self.df_dec.df_out(c)
            c = c.view(b, t, self.df_dec.df_bins, self.df_dec.df_out_ch)
            c = c + c0
        else:
            # Legacy/linear decoder path.
            c0 = self._static_df_convp(c0).transpose(1, 2)
            c = self.df_dec.df_fc_out(c)
            c = c.view(b, t, self.df_dec.df_order * 2, self.df_dec.df_bins)
            c = c.add(c0).view(
                b,
                t,
                self.df_dec.df_order,
                2,
                self.df_dec.df_bins,
            ).transpose(3, 4)

        return c, alpha


class STM32DfDecoderStep(STM32DfDecoder):
    """
    STM32-compatible one-frame DF decoder wrapper.

    Input:
      emb_tbh: [1, B, H]
      c0:      [B, C, 1, F]
      h0:      [L, B, H]

    Output:
      coefs:   [B, 1, F, O*2]
      alpha:   [B, 1, 1]
      h1:      [L, B, H]
    """

    def _manual_squeezed_gru_step(self, emb_tbh: Tensor, h0: Tensor) -> Tuple[Tensor, Tensor]:
        x = self._grouped_linear_static(
            self.df_dec.df_gru.linear_in[0],
            emb_tbh,
            emb_tbh.shape[0],
            emb_tbh.shape[1],
        )
        x = self.df_dec.df_gru.linear_in[1](x)
        gru_in = x
        states = []
        for layer_idx in range(self.stm32_gru_layers):
            x, h = self._manual_gru_layer(x, layer_idx, h0[layer_idx : layer_idx + 1])
            states.append(h)
        if self.df_dec.df_gru.gru_skip is not None:
            x = x + self.df_dec.df_gru.gru_skip(gru_in)
        if not isinstance(self.df_dec.df_gru.linear_out, torch.nn.Identity):
            x = self._grouped_linear_static(
                self.df_dec.df_gru.linear_out[0],
                x,
                x.shape[0],
                x.shape[1],
            )
            x = self.df_dec.df_gru.linear_out[1](x)
        return x, torch.cat(states, dim=0)

    def forward(self, emb_tbh: Tensor, c0: Tensor, h0: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        t, b, _ = emb_tbh.shape
        c, h1 = self._manual_squeezed_gru_step(emb_tbh, h0)

        emb = emb_tbh.reshape(b, t, -1)
        c = c.reshape(b, t, -1)
        if getattr(self.df_dec, "df_skip", None) is not None:
            c = c + self.df_dec.df_skip(emb)

        alpha = self.df_dec.df_fc_a(c)

        if hasattr(self.df_dec, "df_out"):
            c0 = self._static_df_convp(c0).permute(0, 2, 3, 1)
            if hasattr(self.df_dec.df_out[0], "groups"):
                c = self._grouped_linear_static(self.df_dec.df_out[0], c, b, t)
                c = self.df_dec.df_out[1](c)
            else:
                c = self.df_dec.df_out(c)
            c = c.view(b, t * self.df_dec.df_bins * self.df_dec.df_out_ch)
            c0 = c0.reshape(b, t * self.df_dec.df_bins * self.df_dec.df_out_ch)
            c = c + c0
            c = c.reshape(b, t, self.df_dec.df_bins, self.df_dec.df_out_ch)
        else:
            c0 = self._static_df_convp(c0).transpose(1, 2)
            c = self.df_dec.df_fc_out(c)
            c = c.view(b, t, self.df_dec.df_order * 2, self.df_dec.df_bins)
            c = c.add(c0).view(
                b,
                t,
                self.df_dec.df_order,
                2,
                self.df_dec.df_bins,
            ).transpose(3, 4)

        return c, alpha, h1


class STM32DfDecoderStep4D(STM32DfDecoderStep):
    """
    STM32-compatible one-frame DF decoder with rank-4 inputs.

    ST Edge AI 4.0 has trouble when one model mixes rank-3 recurrent
    inputs with rank-4 convolution inputs. This wrapper keeps all external
    inputs rank-4 and reshapes them internally to the decoder math layout.
    """

    def forward(self, emb: Tensor, c0: Tensor, h0: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        emb_tbh = emb.reshape(1, 1, self.df_dec.emb_dim)
        h0_tbh = h0.reshape(
            self.stm32_gru_layers,
            1,
            self.df_dec.df_gru.gru.hidden_size,
        )
        return super().forward(emb_tbh, c0, h0_tbh)


@torch.no_grad()
def export(
    model,
    export_dir: str,
    df_state: DF,
    check: bool = True,
    simplify: bool = True,
    opset=14,
    export_full: bool = False,
    print_graph: bool = False,
):
    model = deepcopy(model).to("cpu")
    model.eval()
    p = ModelParams()
    audio = torch.randn((1, 1 * p.sr))
    spec, feat_erb, feat_spec = df_features(audio, df_state, p.nb_df, device="cpu")

    # Export full model
    if export_full:
        path = os.path.join(export_dir, "deepfilternet2.onnx")
        input_names = ["spec", "feat_erb", "feat_spec"]
        dynamic_axes = {
            "spec": {2: "S"},
            "feat_erb": {2: "S"},
            "feat_spec": {2: "S"},
            "enh": {2: "S"},
            "m": {2: "S"},
            "lsnr": {1: "S"},
        }
        inputs = (spec, feat_erb, feat_spec)
        output_names = ["enh", "m", "lsnr", "coefs"]
        export_impl(
            path,
            model,
            inputs=inputs,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            jit=False,
            check=check,
            simplify=simplify,
            opset_version=opset,
            print_graph=print_graph,
        )

    # Export encoder
    feat_spec = feat_spec.transpose(1, 4).squeeze(4)  # re/im into channel axis
    path = os.path.join(export_dir, "enc.onnx")
    inputs = (feat_erb, feat_spec)
    input_names = ["feat_erb", "feat_spec"]
    dynamic_axes = {
        "feat_erb": {2: "S"},
        "feat_spec": {2: "S"},
        "e0": {2: "S"},
        "e1": {2: "S"},
        "e2": {2: "S"},
        "e3": {2: "S"},
        "emb": {1: "S"},
        "c0": {2: "S"},
        "lsnr": {1: "S"},
    }
    output_names = ["e0", "e1", "e2", "e3", "emb", "c0", "lsnr"]
    e0, e1, e2, e3, emb, c0, lsnr = export_impl(
        path,
        model.enc,
        inputs=inputs,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        jit=True,
        check=check,
        simplify=simplify,
        opset_version=opset,
        print_graph=print_graph,
    )
    np.savez_compressed(
        os.path.join(export_dir, "enc_input.npz"),
        feat_erb=feat_erb.numpy(),
        feat_spec=feat_spec.numpy(),
    )
    np.savez_compressed(
        os.path.join(export_dir, "enc_output.npz"),
        e0=e0.numpy(),
        e1=e1.numpy(),
        e2=e2.numpy(),
        e3=e3.numpy(),
        emb=emb.numpy(),
        c0=c0.numpy(),
        lsnr=lsnr.numpy(),
    )

    # Export erb decoder
    np.savez_compressed(
        os.path.join(export_dir, "erb_dec_input.npz"),
        emb=emb.numpy(),
        e0=e0.numpy(),
        e1=e1.numpy(),
        e2=e2.numpy(),
        e3=e3.numpy(),
    )
    inputs = (emb.clone(), e3, e2, e1, e0)
    input_names = ["emb", "e3", "e2", "e1", "e0"]
    output_names = ["m"]
    dynamic_axes = {
        "emb": {1: "S"},
        "e3": {2: "S"},
        "e2": {2: "S"},
        "e1": {2: "S"},
        "e0": {2: "S"},
        "m": {2: "S"},
    }
    path = os.path.join(export_dir, "erb_dec.onnx")
    m = export_impl(  # noqa
        path,
        model.erb_dec,
        inputs=inputs,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        jit=True,
        check=check,
        simplify=simplify,
        opset_version=opset,
        print_graph=print_graph,
    )
    np.savez_compressed(os.path.join(export_dir, "erb_dec_output.npz"), m=m.numpy())

    # Export df decoder
    np.savez_compressed(
        os.path.join(export_dir, "df_dec_input.npz"), emb=emb.numpy(), c0=c0.numpy()
    )
    inputs = (emb.clone(), c0)
    input_names = ["emb", "c0"]
    output_names = ["coefs"]
    dynamic_axes = {
        "emb": {1: "S"},
        "c0": {2: "S"},
        "coefs": {1: "S"},
    }
    path = os.path.join(export_dir, "df_dec.onnx")
    coefs = export_impl(  # noqa
        path,
        model.df_dec,
        inputs=inputs,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        jit=False,
        check=check,
        simplify=simplify,
        opset_version=opset,
        print_graph=print_graph,
    )
    if isinstance(coefs, tuple):
        coefs = coefs[0]
    np.savez_compressed(os.path.join(export_dir, "df_dec_output.npz"), coefs=coefs.numpy())

    # Export STM32-compatible one-frame df decoder.
    #
    # Keep the full trained GRU depth. Reducing this to fit internal flash is
    # destructive: the exported graph still validates, but audio quality drops
    # sharply. Flash placement/compression should be handled by ST Edge AI
    # generation options instead of changing the network topology.
    stm32_gru_layers = model.df_dec.df_gru.gru.num_layers
    emb_tbh = emb[:, :1, :].clone().transpose(0, 1).contiguous()
    emb_stm32 = emb_tbh.reshape(1, 1, 1, model.df_dec.emb_dim).contiguous()
    c0_step = c0[:, :, :1, :].contiguous()
    h0 = torch.zeros(
        (
            stm32_gru_layers,
            emb_tbh.shape[1],
            model.df_dec.df_gru.gru.hidden_size,
        ),
        dtype=emb_tbh.dtype,
        device=emb_tbh.device,
    )
    h0_stm32 = h0.reshape(1, stm32_gru_layers, 1, -1).contiguous()

    np.savez_compressed(
        os.path.join(export_dir, "df_dec_stm32_input.npz"),
        emb=emb_stm32.detach().cpu().numpy(),
        c0=c0_step.detach().cpu().numpy(),
        h0=h0_stm32.detach().cpu().numpy(),
    )

    inputs = (emb_stm32, c0_step, h0_stm32)
    input_names = ["emb", "c0", "h0"]
    output_names = ["coefs", "alpha", "h1"]
    dynamic_axes = {}

    path = os.path.join(export_dir, "df_dec_stm32.onnx")
    stm32_df_dec = STM32DfDecoderStep4D(model.df_dec, gru_layers=stm32_gru_layers)

    coefs, alpha, h1 = export_impl(
        path,
        stm32_df_dec,
        inputs=inputs,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        jit=False,
        check=check,
        simplify=simplify,
        opset_version=opset,
        print_graph=print_graph,
    )
    fuse_static_convp_pad(path, int(stm32_df_dec.df_convp_tpad.shape[2]))
    set_static_onnx_shapes(path, inputs + (coefs, alpha, h1), input_names + output_names)

    np.savez_compressed(
        os.path.join(export_dir, "df_dec_stm32_output.npz"),
        coefs=coefs.detach().cpu().numpy(),
        alpha=alpha.detach().cpu().numpy(),
        h1=h1.detach().cpu().numpy(),
    )


def main(args):
    try:
        import monkeytype  # noqa: F401
    except ImportError:
        print("Failed to import monkeytype. Please install it via")
        print("$ pip install MonkeyType")
        exit(1)

    print(args)
    model, df_state, _, epoch = init_df(
        args.model_base_dir,
        post_filter=args.pf,
        log_level=args.log_level,
        log_file="export.log",
        config_allow_defaults=True,
        epoch=args.epoch,
    )
    sample = get_test_sample(df_state.sr())
    enhanced = enhance(model, df_state, sample, True)
    out_dir = Path("out")
    if out_dir.is_dir():
        # attempt saving enhanced audio
        save_audio(os.path.join(out_dir, "enhanced.wav"), enhanced, df_state.sr())
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    export(
        model,
        export_dir,
        df_state=df_state,
        opset=args.opset,
        check=args.check,
        simplify=args.simplify,
    )
    model_base_dir = get_model_basedir(args.model_base_dir)
    if model_base_dir != args.export_dir:
        shutil.copyfile(
            os.path.join(model_base_dir, "config.ini"),
            os.path.join(args.export_dir, "config.ini"),
        )
    model_name = Path(model_base_dir).name
    version_file = os.path.join(args.export_dir, "version.txt")
    with open(version_file, "w") as f:
        f.write(f"{model_name}_epoch_{epoch}")
    tar_name = export_dir / (Path(model_base_dir).name + "_onnx.tar.gz")
    with tarfile.open(tar_name, mode="w:gz") as f:
        f.add(os.path.join(args.export_dir, "enc.onnx"))
        f.add(os.path.join(args.export_dir, "erb_dec.onnx"))
        f.add(os.path.join(args.export_dir, "df_dec.onnx"))
        f.add(os.path.join(args.export_dir, "config.ini"))
        f.add(os.path.join(args.export_dir, "version.txt"))


if __name__ == "__main__":
    parser = setup_df_argument_parser()
    parser.add_argument("export_dir", help="Directory for exporting the onnx model.")
    parser.add_argument(
        "--no-check",
        help="Don't check models with onnx checker.",
        action="store_false",
        dest="check",
    )
    parser.add_argument("--simplify", help="Simply onnx models using onnxsim.", action="store_true")
    parser.add_argument("--opset", help="ONNX opset version", type=int, default=12)
    args = parser.parse_args()
    main(args)
