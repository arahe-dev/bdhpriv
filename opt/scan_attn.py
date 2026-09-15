"""Exact GPU-friendly scan / segmented-prefix candidates for Arm-A.

Mathematical scope: Q=K, strict-past, same-document, no softmax, no scale.
Reference oracles (immutable) live in reference/scan_coordinator_oracle.py:
  dense_strict_past_attention <-> scan_cumsum_attention <-> scan_chunked_attention
  dense_coordinator_value      <-> segmented_prefix_coordinator_value

Conventions:
  q: [B,H,T,K]  (post-RoPE queries, also used as keys)
  v: [B,T,Dv]
  segment_start: [B,T] long, each entry = first-token column of its document
  full_mask: [B,T,T] bool, allowed (same-document & strict-past) per canonical harness
"""

import torch


# Module-level cache of strict-lower causal masks keyed by (device, W).
# Avoids rebuilding the mask per chunk per level per update.
CAUSAL_CACHE = {}


def _causal_mask(dev, w):
    key = (str(dev), w)
    m = CAUSAL_CACHE.get(key)
    if m is None or m.device != dev:
        m = torch.ones((w, w), dtype=torch.bool, device=dev).tril(diagonal=-1)
        CAUSAL_CACHE[key] = m
    return m


def dense_fullmask_attention(q, v, full_mask):
    """Canonical-shape baseline: explicit score matrix masked by full_mask."""
    scores = q @ q.transpose(-1, -2)
    scores = scores.masked_fill(~full_mask.unsqueeze(1), 0.0)
    vh = v.unsqueeze(1).expand(-1, q.shape[1], -1, -1)
    return scores @ vh


def dense_segstart_attention(q, v, segment_start):
    """Dense oracle keyed on segment_start (same math as full_mask form)."""
    b, h, t, _ = q.shape
    same = segment_start[:, :, None] == segment_start[:, None, :]
    strict = torch.ones((t, t), dtype=torch.bool, device=q.device).tril(diagonal=-1)
    allowed = (same & strict.unsqueeze(0)).unsqueeze(1)
    scores = q @ q.transpose(-1, -2)
    scores = scores.masked_fill(~allowed, 0.0)
    vh = v.unsqueeze(1).expand(-1, h, -1, -1)
    return scores @ vh


def scan_cumsum_attention(q, v, segment_start):
    """Exact single-pass vectorized scan.

    S_t = sum_{s in [seg_start(t), t-1]} q_s^T v_s ;  out_t = q_t @ S_t.
    Uses one global cumsum over flattened outer products. O(T) activation
    memory instead of O(T^2) scores.CPU/GPU exact up to fp associativity.
    NOTE: materializes [B,H,T,K*Dv]; the chunked form below is the
    production prototype that avoids this.
    """
    b, h, t, k = q.shape
    dv = v.shape[-1]
    a = torch.einsum("bhtk,btd->bhtkd", q, v).reshape(b, h, t, k * dv)
    p = torch.cumsum(a, dim=2)
    before = torch.cat((torch.zeros_like(p[:, :, :1]), p[:, :, :-1]), dim=2)
    base_idx = (segment_start - 1).clamp_min(0).view(b, 1, t, 1).expand(b, h, t, k * dv)
    base = p.gather(2, base_idx)
    fresh = (segment_start == 0).view(b, 1, t, 1).expand_as(base)
    base = torch.where(fresh, torch.zeros_like(base), base)
    s = (before - base).reshape(b, h, t, k, dv)
    return torch.einsum("bhtk,bhtkd->bhtd", q, s)


def scan_chunked_attention(q, v, segment_start, block=8):
    """Exact block-streaming scan: production-math prototype.

    Carries only state S [B,H,K,Dv] across blocks (SRAM-resident in a fused
    kernel); block-local terms come from a short cumsum. A materialization
    is per-block only. Proves the chunked recurrence equals dense math,
    including document resets inside a block and docs spanning blocks.
    """
    b, h, t, k = q.shape
    dv = v.shape[-1]
    a = torch.einsum("bhtk,btd->bhtkd", q, v).reshape(b, h, t, k * dv)
    kd = k * dv
    state = torch.zeros((b, h, kd), dtype=q.dtype, device=q.device)
    outs = []
    for t0 in range(0, t, block):
        t1 = min(t0 + block, t)
        w = t1 - t0
        ab = a[:, :, t0:t1]
        loc = torch.cumsum(ab, dim=2)
        loc_shift = torch.cat(
            (torch.zeros_like(loc[:, :, :1]), loc[:, :, :-1]), dim=2
        )  # L_{i-1}, L_{-1}:=0
        seg = segment_start[:, t0:t1]  # [B,W]
        cont = seg < t0  # continuing doc from before this block
        # Local base L_{seg-1-t0} for fresh docs (L_{-1} := 0 when seg == t0).
        j = (seg - 1 - t0).clamp_min(0).view(b, 1, w, 1).expand(b, h, w, kd)
        lbase = loc.gather(2, j)
        at_t0 = (seg == t0).view(b, 1, w, 1).expand_as(lbase)
        lbase = torch.where(at_t0, torch.zeros_like(lbase), lbase)
        s_carry = state.unsqueeze(2).expand(b, h, w, kd)
        s_t = torch.where(
            cont.view(b, 1, w, 1).expand_as(loc_shift),
            s_carry + loc_shift,
            loc_shift - lbase,
        )
        s_t = s_t.reshape(b, h, w, k, dv)
        outs.append(torch.einsum("bhtk,bhtkd->bhtd", q[:, :, t0:t1], s_t))
        # Next-block state via S_{t+1} = S_t + A_t, reset to 0 at doc starts.
        last = s_t[:, :, -1].reshape(b, h, kd)
        last_a = ab[:, :, -1]
        s_next = last + last_a
        if t1 < t:
            new_doc = segment_start[:, t1] != segment_start[:, t1 - 1]
            s_next = torch.where(
                new_doc.view(b, 1, 1).expand(b, h, kd),
                torch.zeros_like(s_next),
                s_next,
            )
        state = s_next
    return torch.cat(outs, dim=2)


def scan_chunkwise_bthk(qh, vh, segment_start, block=128, single_doc=False,
                        skip_zero_carry=False):
    """Exact chunkwise scan: production execution form (branch-free packed).

    Per block [t0,t1): out = carry_term + local_term, where
      carry_term_t = [seg(t)<t0] * (q_t @ S)            # pre-block state
      local_term   = masked(q_b @ q_b^T) @ v_b          # intra-block, [W,W] scores
    State update is the exact branch-free recurrence (no full-bmm, no
    where-select, no host sync):
      keep = rows >= max(seg(t1)-t0, 0); fresh = (keep*q_b)^T v_b
      S = S * contb + fresh,  contb = [seg(t1) < t0]
    which covers both cases exactly:
      contb=1 -> S + full block sum        (continuing document)
      contb=0 -> sum over rows >= seg(t1)  (document starts inside block)
    Dead final-chunk state updates are skipped (state never used after t1==T).

    skip_zero_carry: the chunk at t0==0 has S=0, so its carry term is
    identically zero for both single-doc and packed (seg<0 is never true).
    Skipping that bmm is exact and removes a dependency edge.
    """
    b, h, t, k = qh.shape
    dv = vh.shape[-1]
    dev = qh.device
    state = torch.zeros((b, h, k, dv), dtype=qh.dtype, device=dev)
    outs = []
    for t0 in range(0, t, block):
        t1 = min(t0 + block, t)
        w = t1 - t0
        qb = qh[:, :, t0:t1]
        vb = vh[:, :, t0:t1]
        # Cached strict-lower mask (avoids rebuilding per chunk per level).
        causal = _causal_mask(dev, w)
        scores = qb @ qb.transpose(-1, -2)
        if single_doc:
            local = scores.masked_fill(~causal, 0.0) @ vb
            if skip_zero_carry and t0 == 0:
                outs.append(local)
            else:
                carry = torch.einsum("bhwk,bhkd->bhwd", qb, state)
                outs.append(local + carry)
            if t1 < t:
                state = state + torch.einsum("bhwk,bhwd->bhkd", qb, vb)
            continue
        seg = segment_start[:, t0:t1]
        samedoc = seg[:, :, None] == seg[:, None, :]
        local = scores.masked_fill(~(samedoc.unsqueeze(1) & causal), 0.0) @ vb
        if skip_zero_carry and t0 == 0:
            # cont = (seg < 0) is all-False, so carry is exactly zero.
            outs.append(local)
        else:
            cont = (seg < t0).to(qb.dtype).view(b, 1, w, 1)
            carry = torch.einsum("bhwk,bhkd->bhwd", qb, state) * cont
            outs.append(local + carry)
        if t1 < t:
            segb = segment_start[:, t1]
            contb = (segb < t0).to(qb.dtype).view(b, 1, 1, 1)
            j = (segb - t0).clamp_min(0)
            keep = (
                (torch.arange(w, device=dev).unsqueeze(0) >= j.unsqueeze(1))
                .to(qb.dtype)
                .view(b, 1, w, 1)
            )
            fresh = torch.einsum("bhwk,bhwd->bhkd", qb * keep, vb)
            state = state * contb + fresh
    return torch.cat(outs, dim=2)


def scan_chunkwise_where_bthk(qh, vh, segment_start, block=128,
                              single_doc=False):
    """LEGACY certified packed update (A/B control only; not production).

    Exactly the G4-certified opt3c recurrence: computes the full block sum
    inside the state update and selects continue/reset with torch.where.
    Kept only so the branch-free implementation can be A/B-measured against
    the certified behavior in the same session. Do not use for new work.
    """
    b, h, t, k = qh.shape
    dv = vh.shape[-1]
    dev = qh.device
    state = torch.zeros((b, h, k, dv), dtype=qh.dtype, device=dev)
    outs = []
    for t0 in range(0, t, block):
        t1 = min(t0 + block, t)
        w = t1 - t0
        qb = qh[:, :, t0:t1]
        vb = vh[:, :, t0:t1]
        seg = segment_start[:, t0:t1]
        scores = qb @ qb.transpose(-1, -2)
        causal = _causal_mask(dev, w)
        if single_doc:
            local = scores.masked_fill(~causal, 0.0) @ vb
            carry = torch.einsum("bhwk,bhkd->bhwd", qb, state)
            outs.append(local + carry)
            state = state + torch.einsum("bhwk,bhwd->bhkd", qb, vb)
            continue
        samedoc = seg[:, :, None] == seg[:, None, :]
        local = scores.masked_fill(~(samedoc.unsqueeze(1) & causal), 0.0) @ vb
        cont = (seg < t0).to(qb.dtype).view(b, 1, w, 1)
        carry = torch.einsum("bhwk,bhkd->bhwd", qb, state) * cont
        outs.append(local + carry)
        full = state + torch.einsum("bhwk,bhwd->bhkd", qb, vb)
        if t1 < t:
            segb = segment_start[:, t1]
            contb = segb < t0
            j = (segb - t0).clamp_min(0)
            keep = (
                (torch.arange(w, device=dev).unsqueeze(0) >= j.unsqueeze(1))
                .to(qb.dtype)
                .view(b, 1, w, 1)
            )
            fresh = torch.einsum("bhwk,bhwd->bhkd", qb * keep, vb)
            state = torch.where(contb.view(b, 1, 1, 1), full, fresh)
    return torch.cat(outs, dim=2)


def scan_hybrid_bthk(qh, vh, block=512):
    """Single-doc hybrid: batched local over ALL chunks, sequential carry.

    Same FLOPs as chunkwise, different launch geometry: the local QK and
    local scorexV become ONE batched bmm each (over flat=B*H*Cn) instead of
    Cn separate calls; only the carry/state chain stays sequential (Cn small
    einsums). No G materialization (unlike parallel). Padded zero rows
    contribute nothing (exact); outputs sliced to T.
    """
    b, h, t, k = qh.shape
    dv = vh.shape[-1]
    dev = qh.device
    cn = (t + block - 1) // block
    pad = cn * block - t
    if pad:
        qh = torch.cat((qh, torch.zeros((b, h, pad, k), dtype=qh.dtype, device=dev)), dim=2)
        vh = torch.cat((vh, torch.zeros((b, h, pad, dv), dtype=vh.dtype, device=dev)), dim=2)
    w = block
    qb = qh.reshape(b, h, cn, w, k)
    vb = vh.reshape(b, h, cn, w, dv)
    flat = b * h * cn
    scores = qb.reshape(flat, w, k) @ qb.reshape(flat, w, k).transpose(-1, -2)
    causal = torch.ones((w, w), dtype=torch.bool, device=dev).tril(diagonal=-1)
    local = (scores.masked_fill(~causal, 0.0) @ vb.reshape(flat, w, dv)).reshape(b, h, cn, w, dv)
    state = torch.zeros((b, h, k, dv), dtype=qh.dtype, device=dev)
    outs = []
    for c in range(cn):
        qbc = qb[:, :, c]
        outs.append(local[:, :, c] + torch.einsum("bhwk,bhkd->bhwd", qbc, state))
        state = state + torch.einsum("bhwk,bhwd->bhkd", qbc, vb[:, :, c])
    out = torch.cat(outs, dim=2)
    return out[:, :, :t] if pad else out


def scan_static4_bthk(qh, vh, causal):
    """Single-doc static four-stage scan (production specialization).

    Exact chunkwise math with T == 4*W, four EXPLICIT stages, no Python loop
    over chunks, no dynamic bounds, no per-chunk mask construction: the
    (W,W) strict-lower causal mask is passed in (module-cached). Lets
    torch.compile see a completely static graph. Single-doc only.
    """
    b, h, t, k = qh.shape
    dv = vh.shape[-1]
    w = causal.shape[0]
    assert t == 4 * w, f"static4 needs T==4W, got T={t} W={w}"
    q0, q1, q2, q3 = qh.chunk(4, dim=2)
    v0, v1, v2, v3 = vh.chunk(4, dim=2)
    s0 = q0 @ q0.transpose(-1, -2)
    l0 = s0.masked_fill(~causal, 0.0) @ v0
    o0 = l0
    a0 = q0.transpose(-1, -2) @ v0
    s1m = q1 @ q1.transpose(-1, -2)
    l1 = s1m.masked_fill(~causal, 0.0) @ v1
    o1 = l1 + q1 @ a0
    a1 = a0 + q1.transpose(-1, -2) @ v1
    s2m = q2 @ q2.transpose(-1, -2)
    l2 = s2m.masked_fill(~causal, 0.0) @ v2
    o2 = l2 + q2 @ a1
    a2 = a1 + q2.transpose(-1, -2) @ v2
    s3m = q3 @ q3.transpose(-1, -2)
    l3 = s3m.masked_fill(~causal, 0.0) @ v3
    o3 = l3 + q3 @ a2
    return torch.cat([o0, o1, o2, o3], dim=2)


def scan_parallel_bthk(qh, vh, block=256):
    """Exact parallel-scan for the single-document case (no resets).

    Chunks are independent given chunk-entry states G_c = sum_{s<c0} A_s:
      A_c = q_c^T @ v_c            (one batched bmm over all chunks)
      G_{c+1} = G_c + A_c          (tiny sequential prefix over Cn states;
                                    a plain loop, compile-safe, negligible)
      out_c = masked(q_c@q_c^T)@v_c + q_c @ G_c   (three batched bmms)
    ~5 launches/level, zero Python T-loop, no cumsum/gather/scan ops.
    Peak transient O(B*H*Cn*K*Dv) for chunk states (64 MB @B1/T2048/W256).
    Pads T up to a multiple of block; exact on the unpadded prefix.
    """
    b, h, t, k = qh.shape
    dv = vh.shape[-1]
    dev = qh.device
    cn = (t + block - 1) // block
    pad = cn * block - t
    if pad:
        qh = torch.cat((qh, torch.zeros((b, h, pad, k), dtype=qh.dtype, device=dev)), dim=2)
        vh = torch.cat((vh, torch.zeros((b, h, pad, dv), dtype=vh.dtype, device=dev)), dim=2)
    w = block
    qb = qh.reshape(b, h, cn, w, k)
    vb = vh.reshape(b, h, cn, w, dv)
    flat = b * h * cn
    # Chunk outer sums: A_c = q_c^T @ v_c.
    a = (qb.reshape(flat, w, k).transpose(-1, -2) @ vb.reshape(flat, w, dv))
    a = a.reshape(b, h, cn, k, dv)
    # Entry states G_0 = 0, G_{c+1} = G_c + A_c.
    g_states = []
    g = torch.zeros((b, h, k, dv), dtype=qh.dtype, device=dev)
    for c in range(cn):
        g_states.append(g)
        g = g + a[:, :, c]
    g = torch.stack(g_states, dim=2)  # [B,H,Cn,K,Dv]
    # Intra-chunk strict-past terms.
    scores = qb.reshape(flat, w, k) @ qb.reshape(flat, w, k).transpose(-1, -2)
    causal = torch.ones((w, w), dtype=torch.bool, device=dev).tril(diagonal=-1)
    local = (scores.masked_fill(~causal, 0.0) @ vb.reshape(flat, w, dv)).reshape(b, h, cn, w, dv)
    # Carry terms q_c @ G_c.
    carry = (qb.reshape(flat, w, k) @ g.reshape(flat, k, dv)).reshape(b, h, cn, w, dv)
    out = (local + carry).reshape(b, h, cn * block, dv)
    return out[:, :, :t] if pad else out


def dense_coordinator_fullmask(z, segpos, full_mask):
    """Canonical coordinator formula: mask-BMM strict-past same-document mean."""
    prev = torch.bmm(full_mask.to(z.dtype), z)
    den = segpos.clamp_min(1).to(z.dtype).unsqueeze(-1)
    return prev / den - z


def segmented_prefix_coordinator(z, segpos, segment_start, single_doc=False):
    """Exact segmented-prefix form of the coordinator value.

    NOTE: cumsum runs over the last dim on a transposed view ([B,W,T]).
    Identical to cumsum(z, dim=1) but keeps the scan dim contiguous, which
    inductor's scan codegen handles robustly (non-contiguous SplitScan over
    [B,T,D] hit an inductor codegen bug on torch 2.11).
    """
    b, t, w = z.shape
    incl = z.transpose(1, 2).cumsum(-1).transpose(1, 2)
    before = torch.cat((torch.zeros_like(z[:, :1]), incl[:, :-1]), dim=1)
    if single_doc:
        prev = before
    else:
        idx = segment_start.to(torch.long).clamp(0, t - 1).unsqueeze(-1).expand(b, t, w)
        prev = before - before.gather(1, idx)
    den = segpos.clamp_min(1).to(z.dtype).unsqueeze(-1)
    return prev / den - z
