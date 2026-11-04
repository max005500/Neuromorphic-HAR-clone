"""Frozen BN folding and explicit weight-only INT8 sensitivity experiments."""
import copy
import unittest

import torch
from torch import nn

from models.inference import prepare_temporal_inference
from models.temporal import TemporalSCN


class CompactInferenceTests(unittest.TestCase):
    def model(self):
        torch.manual_seed(31)
        kwargs = dict(n_channels=6, conv_channels=(3, 4, 4), conv_kernel_sizes=(3, 3, 3),
                      conv_strides=(2, 2, 1), pool_kernel_sizes=(2, 2, 2),
                      pool_strides=(2, 2, 2), p_drop=0., offset_hidden=3)
        model = TemporalSCN(kwargs, hidden_size=4, boundary=True).eval()
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, nn.BatchNorm1d):
                    module.running_mean.copy_(torch.linspace(-.3, .2, module.num_features))
                    module.running_var.copy_(torch.linspace(.4, 1.7, module.num_features))
                    module.weight.copy_(torch.linspace(.6, 1.3, module.num_features))
                    module.bias.copy_(torch.linspace(-.2, .1, module.num_features))
        return model

    def test_no_options_returns_original_and_transforms_require_eval(self):
        model = self.model()
        original, report = prepare_temporal_inference(model)
        self.assertIs(model, original)
        self.assertEqual(report['quantized_weight_count'], 0)
        self.assertFalse(report['native_int8'])
        for options in ({'fuse_bn': True}, {'weight_bits': 8}, {'fuse_bn': True, 'weight_bits': 8}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, 'eval'):
                prepare_temporal_inference(model.train(), **options)
        with self.assertRaisesRegex(ValueError, '8-bit'):
            prepare_temporal_inference(model.eval(), weight_bits=4)

    def test_folded_currents_match_nontrivial_bn_and_original_is_unchanged(self):
        model = self.model()
        snapshot = copy.deepcopy(model.state_dict())
        prepared, report = prepare_temporal_inference(model, fuse_bn=True)
        self.assertEqual(len(report['bn_fused_layers']), 4)
        for name in ('conv_block1', 'conv_block2', 'conv_block3', 'offset_proj_spk'):
            before, after = getattr(model.backbone, name), getattr(prepared.backbone, name)
            self.assertIsInstance(after.bn, nn.Identity)
            if name.startswith('conv'):
                value = torch.randn(2, before.conv.in_channels, 32)
                actual, expected = after.conv(value), before.bn(before.conv(value))
            else:
                value = torch.randn(9, before.fc.in_features)
                actual, expected = after.fc(value), before.bn(before.fc(value))
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        self.assertEqual(model.state_dict().keys(), snapshot.keys())
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, snapshot[name], rtol=0, atol=0)

    def test_fusion_preserves_temporal_logits_membrane_and_padding(self):
        model = self.model()
        prepared, _ = prepare_temporal_inference(model, fuse_bn=True)
        events = torch.randint(0, 2, (2, 3, 64, 6, 2)).float()
        offsets = torch.randn(2, 3, 6)
        present = torch.tensor([[True, True, True], [True, True, False]])
        with torch.no_grad():
            expected, reference_state = model(events, offsets, valid=present)
            actual, state = prepared(events, offsets, valid=present)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=2e-5, atol=2e-6)
        for value, reference in zip(state, reference_state):
            torch.testing.assert_close(value, reference, rtol=2e-5, atol=2e-6)

    def test_quantization_lattice_errors_zero_channels_and_unchanged_float_states(self):
        model = self.model()
        with torch.no_grad():
            model.memory.recurrent.weight[0].zero_()
        snapshot = copy.deepcopy(model.state_dict())
        prepared, report = prepare_temporal_inference(model, weight_bits=8)
        self.assertEqual(report['weight_bits'], 8)
        self.assertEqual(report['weight_quantization'], 'symmetric_per_output_channel_qdq')
        self.assertIn('no membrane', report['state_precision'])
        count = 0
        for name, layer in prepared.named_modules():
            if not isinstance(layer, (nn.Conv1d, nn.Linear)):
                continue
            details = report['per_layer_quantization'][name]
            scales = torch.tensor(details['scales'])
            normalized = layer.weight.detach().reshape(len(scales), -1) / scales[:, None]
            torch.testing.assert_close(normalized, normalized.round(), rtol=0, atol=1e-5)
            self.assertLessEqual(float(normalized.abs().max()), 127.00001)
            error = (layer.weight.detach() - snapshot[name + '.weight']).reshape(len(scales), -1).abs().amax(1)
            self.assertTrue(torch.all(error <= scales / 2 + 1e-7))
            self.assertTrue(torch.isfinite(layer.weight).all())
            self.assertEqual(layer.weight.dtype, torch.float32)
            count += layer.weight.numel()
            if layer.bias is not None:
                torch.testing.assert_close(layer.bias, snapshot[name + '.bias'], rtol=0, atol=0)
        self.assertEqual(report['quantized_weight_count'], count)
        torch.testing.assert_close(prepared.memory.recurrent.weight[0], torch.zeros(4), rtol=0, atol=0)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, snapshot[name], rtol=0, atol=0)
        for name, module in prepared.named_modules():
            if isinstance(module, nn.BatchNorm1d):
                torch.testing.assert_close(module.weight, snapshot[name + '.weight'], rtol=0, atol=0)
        with torch.no_grad():
            outputs, state = prepared(torch.zeros(2, 2, 64, 6, 2), torch.zeros(2, 2, 6))
        self.assertEqual(state[0].dtype, torch.float32)
        self.assertTrue(torch.isfinite(outputs['logits']).all())

    def test_both_transforms_quantize_fused_weights_after_bn_is_removed(self):
        model = self.model()
        fused, _ = prepare_temporal_inference(model, fuse_bn=True)
        combined, report = prepare_temporal_inference(model, fuse_bn=True, weight_bits=8)
        self.assertFalse(any(isinstance(module, nn.BatchNorm1d) for module in combined.modules()))
        self.assertEqual(len(report['bn_fused_layers']), 4)
        for name, layer in combined.named_modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                scales = torch.tensor(report['per_layer_quantization'][name]['scales'])
                original = dict(fused.named_modules())[name].weight.detach()
                error = (layer.weight.detach() - original).reshape(len(scales), -1).abs().amax(1)
                self.assertTrue(torch.all(error <= scales / 2 + 1e-7))

    def test_batchnorm_without_saved_running_statistics_rejected(self):
        model = self.model()
        model.backbone.conv_block1.bn = nn.BatchNorm1d(3, track_running_stats=False).eval()
        with self.assertRaisesRegex(ValueError, 'running statistics'):
            prepare_temporal_inference(model, fuse_bn=True)
