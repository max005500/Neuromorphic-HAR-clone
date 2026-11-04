"""Masked classification, soft transitions and boundary supervision for episodes."""
from __future__ import annotations
import math
import time
import numpy as np
from tqdm.auto import tqdm
import torch
from torch.nn import functional as F
from models.temporal import detach_state
from train.utils import make_autocast


def temporal_loss(outputs, target, valid, previous_target, previous_valid, *, costs,
                  class_weight=None, boundary_pos_weight=1., previous_probability=None,
                  lambda_transition=0., lambda_boundary=0., label_smoothing=0.):
    valid = valid & (target >= 0) & (target < 4)
    previous_valid = previous_valid & (previous_target >= 0) & (previous_target < 4)
    safe = target.clamp(0, 3)
    logits = outputs['logits'].float()
    if 'group' in outputs:
        group = ((safe == 0) | (safe == 3)).long()
        ce = F.cross_entropy(outputs['group'].float().flatten(0, 1), group.flatten(), reduction='none',
                             label_smoothing=label_smoothing).reshape_as(target)
        for key, child, mask in (('normal', (safe == 2).long(), group == 0),
                                  ('event', (safe == 3).long(), group == 1)):
            ce = ce + F.cross_entropy(outputs[key].float().flatten(0, 1), child.flatten(), reduction='none',
                                      label_smoothing=label_smoothing).reshape_as(target) * mask
    else:
        ce = F.cross_entropy(logits.flatten(0, 1), safe.flatten(), reduction='none',
                             label_smoothing=label_smoothing).reshape_as(target)
    weights = valid.to(ce.dtype)
    if class_weight is not None:
        weights = weights * class_weight[safe]
    ce_sum, ce_weight = (ce * weights).sum(), weights.sum()
    previous = torch.cat((previous_target[:, None], target[:, :-1]), 1)
    pair_valid = valid & torch.cat((previous_valid[:, None], valid[:, :-1]), 1)
    boundary_target = (previous != target).float()
    probability = torch.softmax(logits, -1)
    if previous_probability is None:
        previous_probability = torch.zeros_like(probability[:, 0])
    before = torch.cat((previous_probability[:, None], probability[:, :-1]), 1)
    pair_count = pair_valid.sum()
    # This is only a 4x4 expectation. Broadcasting avoids the batched outer-
    # product backward selected by einsum, which recent CUDA builds may route
    # through a Triton JIT kernel requiring a host C compiler.
    # Disabled auxiliary terms remain available for logging without a graph.
    gradients_enabled = torch.is_grad_enabled()
    with torch.set_grad_enabled(gradients_enabled and lambda_transition > 0):
        transition = (before.unsqueeze(-1) * costs[None, None, :, :]
                      * probability.unsqueeze(-2)).sum(dim=(-2, -1))
        transition_sum = (transition * pair_valid).sum()
    if 'boundary' in outputs:
        with torch.set_grad_enabled(gradients_enabled and lambda_boundary > 0):
            boundary = F.binary_cross_entropy_with_logits(outputs['boundary'].float(), boundary_target,
                reduction='none', pos_weight=logits.new_tensor(boundary_pos_weight))
            boundary_sum = (boundary * pair_valid).sum()
    else:
        boundary_sum = logits.new_zeros(())
    total = ce_sum / ce_weight.clamp_min(1e-12)
    if lambda_transition > 0:
        total = total + lambda_transition * transition_sum / pair_count.clamp_min(1)
    if lambda_boundary > 0:
        total = total + lambda_boundary * boundary_sum / pair_count.clamp_min(1)
    return total, dict(ce_sum=ce_sum, ce_weight=ce_weight, transition_sum=transition_sum,
                       boundary_sum=boundary_sum, pair_count=pair_count,
                       pair_valid=pair_valid, boundary_target=boundary_target, probability=probability)


def run_temporal_epoch(model, episodes, cfg, device, stats, *, optimizer=None, scaler=None,
                       epoch=0, collect=False, phase=None):
    model.train(optimizer is not None)
    costs = torch.tensor(stats['transition_costs'], dtype=torch.float32, device=device)
    weights = None
    if cfg.class_weight == 'balanced':
        counts = np.asarray(stats['class_counts'])
        if np.any(counts == 0):
            raise ValueError(f'Training lacks a class for balanced weights: {counts.tolist()}')
        weights = torch.tensor(counts.sum() / (4 * counts), dtype=torch.float32, device=device)
    sums = torch.zeros(9, dtype=torch.float64, device=device)
    predictions = []
    groups = list(episodes.groups(cfg.batch_size, shuffle=optimizer is not None, seed=cfg.seed + epoch))
    total_chunks = sum(math.ceil(max(len(ep.indices) for ep in group) / cfg.sequence_length) for group in groups)
    label = phase or ('Train' if optimizer is not None else 'Evaluación')
    if epoch:
        label = f'{label} e{epoch:03d}/{getattr(cfg, "epochs", epoch)}'
    progress = tqdm(total=total_chunks, desc=label, unit='bloque', dynamic_ncols=True,
                    mininterval=.5, disable=True if getattr(cfg, 'no_progress', False) else None)
    processed_windows = 0
    last_metrics = time.monotonic()
    live_metrics = {}
    try:
        for group_number, group in enumerate(groups, 1):
            state = previous_probability = None  # reset between independent recordings
            for chunk in episodes.chunks(group, cfg.sequence_length):
                events = chunk['events'].to(device, non_blocking=True)
                offsets = chunk['offsets'].to(device, non_blocking=True)
                target, valid = (chunk[key].to(device) for key in ('targets', 'valid'))
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(optimizer is not None), make_autocast(bool(cfg.amp and device.type == 'cuda')):
                    outputs, state = model(events, offsets, state, valid=chunk['present'])
                    loss, parts = temporal_loss(outputs, target, valid,
                        chunk['previous_target'].to(device), chunk['previous_valid'].to(device), costs=costs,
                        class_weight=weights, boundary_pos_weight=stats['boundary_pos_weight'],
                        previous_probability=previous_probability,
                        lambda_transition=cfg.lambda_transition, lambda_boundary=cfg.lambda_boundary,
                        label_smoothing=cfg.label_smoothing)
                    if optimizer is not None:
                        scaler.scale(loss).backward()
                        if cfg.grad_clip > 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                prediction = model.predict(outputs)
                sums[0].add_(parts['ce_sum'].detach().double())
                sums[1].add_(parts['ce_weight'].detach().double())
                sums[2].add_(parts['transition_sum'].detach().double())
                sums[3].add_(parts['boundary_sum'].detach().double())
                sums[4].add_(parts['pair_count'].detach().double())
                sums[5].add_(((prediction == target) & valid).sum())
                sums[6].add_(valid.sum())
                if 'boundary' in outputs:
                    guess = outputs['boundary'] >= 0
                    truth = parts['boundary_target'].bool()
                    sums[7].add_((guess & truth & parts['pair_valid']).sum())
                    sums[8].add_(((guess | truth) & parts['pair_valid']).sum())
                if collect:
                    present = chunk['present'].numpy()
                    row = dict(indices=chunk['indices'][present],
                        target=target.detach().cpu().numpy()[present],
                        predicted=prediction.detach().cpu().numpy()[present],
                        probability=parts['probability'].detach().cpu().numpy()[present],
                        valid=chunk['valid'].numpy()[present],
                        memory_spikes=outputs['memory_spikes'].detach().cpu().numpy()[present]
                                      if 'memory_spikes' in outputs else np.zeros(int(present.sum())))
                    for key in ('group', 'normal', 'event'):
                        if key in outputs:
                            row[key + '_probability'] = torch.softmax(outputs[key].detach().float(), -1).cpu().numpy()[present]
                    predictions.append(row)
                state = detach_state(state)
                previous_probability = parts['probability'][:, -1].detach()
                processed_windows += int(chunk['present'].sum())
                if not progress.disable:
                    postfix = dict(ventanas=processed_windows, grupo=f'{group_number}/{len(groups)}')
                    if time.monotonic() - last_metrics >= 5.:
                        # Read GPU totals occasionally, never at every chunk.
                        values = sums.detach().cpu().tolist()
                        live_loss = (values[0] / max(values[1], 1e-12)
                                     + cfg.lambda_transition * values[2] / max(values[4], 1)
                                     + cfg.lambda_boundary * values[3] / max(values[4], 1))
                        postfix.update(loss=f'{live_loss:.4f}',
                                       acc=f'{values[5] / max(values[6], 1):.1%}')
                        last_metrics = time.monotonic()
                        live_metrics = {key: postfix[key] for key in ('loss', 'acc')}
                    else:
                        postfix.update(live_metrics)
                    progress.set_postfix(postfix, refresh=False)
                progress.update(1)
    finally:
        progress.close()
    ce, denominator, transition, boundary, pairs, correct, count, btp, union = sums.cpu().tolist()
    result = dict(loss_ce=ce / max(denominator, 1e-12), loss_transition=transition / max(pairs, 1),
                  loss_boundary=boundary / max(pairs, 1), acc=correct / max(count, 1), n=int(count),
                  boundary_f1=2 * btp / (union + btp) if union + btp else None)
    result['loss'] = result['loss_ce'] + cfg.lambda_transition * result['loss_transition'] + cfg.lambda_boundary * result['loss_boundary']
    if collect:
        result['predictions'] = {key: np.concatenate([row[key] for row in predictions]) for key in predictions[0]}
        order = np.argsort(result['predictions']['indices'])
        result['predictions'] = {key: value[order] for key, value in result['predictions'].items()}
    return result
