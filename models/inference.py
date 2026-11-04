"""Optional inference transforms for the temporal SCN.

INT8 here is a weight-only quantize/dequantize sensitivity experiment. Kernels,
activations and membrane state still use floating-point arithmetic. It does not
claim native INT8 execution, fixed-point state or hardware energy savings.
"""
from __future__ import annotations

import copy

import torch
from torch import nn

from models.utils import ConvLIFMaxPoolBlock, SpikingLinearBlock


def _fold_batch_norm(layer: nn.Module, bn: nn.BatchNorm1d) -> None:
    """Absorb a frozen output-channel BN affine transform into Conv/Linear."""
    if bn.training or not bn.track_running_stats or bn.running_mean is None or bn.running_var is None:
        raise ValueError('BatchNorm fusion requires eval mode and saved running statistics.')
    if isinstance(layer.weight, nn.parameter.UninitializedParameter):
        raise ValueError('Initialize all layers before preparing inference.')
    if layer.weight.shape[0] != bn.num_features:
        raise ValueError('BatchNorm output channels do not match the preceding layer.')
    if (not torch.isfinite(bn.running_mean).all() or not torch.isfinite(bn.running_var).all()
            or torch.any(bn.running_var < 0)):
        raise ValueError('BatchNorm running statistics must be finite with nonnegative variance.')
    with torch.no_grad():
        gamma = bn.weight if bn.affine else torch.ones_like(bn.running_mean)
        beta = bn.bias if bn.affine else torch.zeros_like(bn.running_mean)
        scale = gamma * torch.rsqrt(bn.running_var + bn.eps)
        bias = layer.bias if layer.bias is not None else torch.zeros_like(bn.running_mean)
        fused_weight = layer.weight * scale.reshape((-1,) + (1,) * (layer.weight.ndim - 1))
        fused_bias = (bias - bn.running_mean) * scale + beta
    layer.weight = nn.Parameter(fused_weight, requires_grad=layer.weight.requires_grad)
    layer.bias = nn.Parameter(fused_bias, requires_grad=(layer.bias.requires_grad
                                                     if layer.bias is not None else layer.weight.requires_grad))


def _quantize_weight(layer: nn.Module) -> dict:
    """Symmetric signed INT8 lattice per output channel, dequantized to FP32."""
    if isinstance(layer.weight, nn.parameter.UninitializedParameter):
        raise ValueError('Initialize all layers before preparing inference.')
    if layer.weight.dtype != torch.float32:
        raise ValueError('INT8 weight simulation expects FP32 model parameters; call model.float().')
    with torch.no_grad():
        weight = layer.weight.detach()
        if not torch.isfinite(weight).all():
            raise ValueError('INT8 weight simulation requires finite weights.')
        flat = weight.reshape(weight.shape[0], -1)
        maximum = flat.abs().amax(1)
        scale = torch.where(maximum > 0, (maximum / 127.).clamp_min(torch.finfo(weight.dtype).tiny),
                            torch.ones_like(maximum))
        quantized = torch.round(flat / scale[:, None]).clamp(-127, 127).to(torch.int8)
        restored = (quantized.float() * scale[:, None]).reshape_as(weight)
        error = restored - weight
        report = dict(weight_count=weight.numel(), output_channels=weight.shape[0],
                      scales=scale.cpu().tolist(), max_abs_error=float(error.abs().max()),
                      mean_abs_error=float(error.abs().mean()))
        layer.weight.copy_(restored)
    return report


def prepare_temporal_inference(model: nn.Module, *, fuse_bn: bool = False,
                               weight_bits: int | None = None) -> tuple[nn.Module, dict]:
    """Return an optional transformed copy plus a JSON-serializable report.

    Call ``model.eval()`` before requesting either transform. The caller's model
    and original checkpoint remain unchanged. BN is fused before weight
    quantization when both options are enabled. No options returns the original
    model, avoiding an unnecessary copy for ordinary evaluation.
    """
    if weight_bits not in (None, 8):
        raise ValueError('Only 8-bit symmetric weight simulation is supported.')
    if (fuse_bn or weight_bits is not None) and model.training:
        raise ValueError('Inference transforms require model.eval(); never apply them during training.')
    parameters = sum(parameter.numel() for parameter in model.parameters())
    report = dict(bn_fused_layers=[], weight_bits=weight_bits,
                  weight_quantization=None, quantized_weight_count=0,
                  per_layer_quantization={}, native_int8=False,
                  state_precision='unchanged floating-point; no membrane or activation quantization',
                  parameters_before=parameters, parameters_after=parameters,
                  scope='inference copy; original parameters and checkpoint unchanged')
    if not fuse_bn and weight_bits is None:
        return model, report
    prepared = copy.deepcopy(model)
    if fuse_bn:
        for name, block in prepared.named_modules():
            if isinstance(block, (ConvLIFMaxPoolBlock, SpikingLinearBlock)) and isinstance(block.bn, nn.BatchNorm1d):
                layer = block.conv if isinstance(block, ConvLIFMaxPoolBlock) else block.fc
                _fold_batch_norm(layer, block.bn)
                block.bn = nn.Identity().eval()
                report['bn_fused_layers'].append(name)
    if weight_bits is not None:
        report['weight_quantization'] = 'symmetric_per_output_channel_qdq'
        for name, layer in prepared.named_modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                details = _quantize_weight(layer)
                report['per_layer_quantization'][name] = details
                report['quantized_weight_count'] += details['weight_count']
    report['parameters_after'] = sum(parameter.numel() for parameter in prepared.parameters())
    return prepared, report
