"""Direct transcription checks against the frozen trainer source.

These tests import ``training/arm_a_2p5b_trainer.py`` @ 0dcbb87 and compare our
reference operators with the trainer's own functions. They are the strongest
available proof that the Akasha reference is source-faithful even though the
trained checkpoint is not local.
"""

from __future__ import annotations

import torch

from akasha.bench.correctness import canonical_init
from akasha.models.arma.ops import (
    apply_rope,
    layer_norm,
    rope_pair_freq,
    rope_phase,
)
from akasha.models.arma.reference_full import full_forward, scan_attention
from akasha.models.arma.reference_recurrent import create_state, prefill_all_logits


def _tiny_trainer_cfg(mod):
    return mod.ArmAConfig(T=16, V=32, D=8, N=32, H=2, L=2, HIDDEN=12,
                          SCAN_BLOCK=4)


def _load_trainer_weights(module, cfg, akasha_weights):
    model = module.OptArmA(cfg, torch.device("cpu"), scan_block=cfg.SCAN_BLOCK)
    with torch.no_grad():
        model = model.double()
        model.embedding.weight.copy_(akasha_weights.embedding.double())
        model.encoder.copy_(akasha_weights.encoder.double())
        model.decoder_x.copy_(akasha_weights.decoder_x.double())
        model.decoder_y.copy_(akasha_weights.decoder_y.double())
        model.readout.copy_(akasha_weights.readout.double())
        model.coordinator.Wc.copy_(akasha_weights.coord_Wc.double())
        model.coordinator.bc.copy_(akasha_weights.coord_bc.double())
        model.coordinator.alpha.copy_(akasha_weights.coord_alpha.double())
        model.writer.W1.copy_(akasha_weights.writer_W1.double())
        model.writer.W2.copy_(akasha_weights.writer_W2.double())
    return model


def test_rope_matches_trainer_bitwise(trainer_module, tiny_cfg):
    freq_ours = rope_pair_freq(tiny_cfg, torch.device("cpu"))
    freq_theirs = trainer_module.rope_pair_freq(tiny_cfg, torch.device("cpu"))
    assert torch.equal(freq_ours, freq_theirs)

    positions = torch.arange(tiny_cfg.T, dtype=torch.long)
    cos_o, sin_o = rope_phase(positions, freq_ours)
    cos_t, sin_t = trainer_module.rope_phase(positions, freq_theirs)
    assert torch.equal(cos_o, cos_t)
    assert torch.equal(sin_o, sin_t)


def test_attention_scan_matches_trainer_scan(trainer_module, tiny_cfg):
    torch.manual_seed(23)
    cfg = tiny_cfg
    qh = torch.randn(1, cfg.H, cfg.T, cfg.K, dtype=torch.float64)
    vh = torch.randn(1, cfg.H, cfg.T, cfg.D, dtype=torch.float64)
    segment_start = torch.zeros(1, cfg.T, dtype=torch.long)
    segment_start[0, 5:] = 5
    segment_start[0, 11:] = 11

    ours = scan_attention(qh, vh, segment_start, block=4)
    theirs = trainer_module.scan_chunkwise_candidate(
        qh, vh, segment_start, block=4, zero_carry=True
    )
    assert torch.equal(ours, theirs)


def test_layernorm_matches_trainer_module(trainer_module, tiny_cfg):
    torch.manual_seed(29)
    x = torch.randn(3, 5, tiny_cfg.D, dtype=torch.float64)
    trainer_ln = trainer_module.OptArmA(tiny_cfg, torch.device("cpu")).ln.double()
    ours = layer_norm(x)
    theirs = trainer_ln(x)
    assert torch.equal(ours, theirs)


def test_forward_packed_matches_full_and_recurrent(trainer_module, tiny_cfg):
    cfg = _tiny_trainer_cfg(trainer_module)
    akasha_cfg = tiny_cfg.with_overrides(T=16, V=32, D=8, N=32, H=2, L=2, HIDDEN=12)
    weights = canonical_init(akasha_cfg).to(dtype=torch.float64)
    model = _load_trainer_weights(trainer_module, cfg, weights)

    torch.manual_seed(31)
    t = cfg.T
    ids = torch.randint(0, cfg.V, (1, t))
    pos = torch.arange(t, dtype=torch.int32).unsqueeze(0)
    segpos = torch.zeros(1, t, dtype=torch.int32)
    start = torch.zeros(1, t, dtype=torch.int32)
    start[0, 6:] = 6
    start[0, 12:] = 12
    for i in range(1, t):
        segpos[0, i] = i - int(start[0, i])
    causal = torch.ones(t, t, dtype=torch.bool).tril(diagonal=-1)
    full_mask = (
        (start[:, :, None] == start[:, None, :])
        & causal.unsqueeze(0)
    )
    with torch.no_grad():
        trainer_logits = model.forward_packed(ids, pos, segpos, full_mask, start)

    akasha_logits = full_forward(
        weights,
        akasha_cfg,
        ids[0],
        positions=pos[0].long(),
        segment_ids=start[0].long(),
    ).logits[0]

    state = create_state(weights, akasha_cfg)
    rec_logits = prefill_all_logits(
        weights,
        akasha_cfg,
        state,
        ids[0],
        positions=pos[0].long(),
        segment_ids=start[0].long(),
    )

    err_full = (trainer_logits[0].double() - akasha_logits).abs().max().item()
    err_rec = (trainer_logits[0].double() - rec_logits).abs().max().item()
    assert err_full <= 1e-10, err_full
    assert err_rec <= 1e-10, err_rec
