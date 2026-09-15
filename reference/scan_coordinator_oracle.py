# Arm-A scan + coordinator reference cell (correctness-only)
# This deliberately uses tiny CPU tensors. It is NOT the 3+10 benchmark cell:
# the Python recurrence below is a readable reference, not a production GPU kernel.
# Keep T small; autograd retains the per-token state history.

import torch

MAX_REFERENCE_T = 128


def dense_strict_past_attention(q, v, segment_start):
    """Dense oracle: q is also k; no softmax or attention scaling."""
    batch, heads, time, key_dim = q.shape
    if time > MAX_REFERENCE_T:
        raise ValueError(f"Reference-only guard: T must be <= {MAX_REFERENCE_T}.")
    if v.shape[:2] != (batch, time):
        raise ValueError("v must have shape [B,T,Dv].")

    same_document = segment_start[:, :, None] == segment_start[:, None, :]
    strict_past = torch.ones(
        (time, time), dtype=torch.bool, device=q.device
    ).tril(diagonal=-1)
    allowed = (same_document & strict_past.unsqueeze(0)).unsqueeze(1)

    scores = q @ q.transpose(-1, -2)
    scores = scores.masked_fill(~allowed, 0.0)
    values_by_head = v.unsqueeze(1).expand(-1, heads, -1, -1)
    return scores @ values_by_head


def strict_past_state_scan_reference(q, v, segment_start):
    """
    Exact recurrence reference:
      output_t = q_t @ S_t
      S_{t+1} = S_t + q_t^T v_t
    Reset S at each document start. This uses only a rolling [B,H,K,Dv]
    state explicitly, but autograd still retains intermediates; tiny T only.
    """
    batch, heads, time, key_dim = q.shape
    if time > MAX_REFERENCE_T:
        raise ValueError(f"Reference-only guard: T must be <= {MAX_REFERENCE_T}.")
    if segment_start.shape != (batch, time):
        raise ValueError("segment_start must have shape [B,T].")
    if v.shape[:2] != (batch, time):
        raise ValueError("v must have shape [B,T,Dv].")

    value_dim = v.shape[-1]
    state = torch.zeros(
        (batch, heads, key_dim, value_dim), dtype=q.dtype, device=q.device
    )
    outputs = []

    for t in range(time):
        if t == 0:
            state = torch.zeros_like(state)
        else:
            reset = segment_start[:, t] != segment_start[:, t - 1]
            state = torch.where(
                reset[:, None, None, None], torch.zeros_like(state), state
            )

        q_t = q[:, :, t, :]
        out_t = torch.einsum("bhk,bhkd->bhd", q_t, state)
        v_t = v[:, t, :][:, None, None, :]
        state = state + q_t.unsqueeze(-1) * v_t
        outputs.append(out_t)

    return torch.stack(outputs, dim=2)


def dense_coordinator_value(z, segpos, segment_start):
    """Reference for the strict-past same-document mask-BMM formula."""
    batch, time, width = z.shape
    same_document = segment_start[:, :, None] == segment_start[:, None, :]
    strict_past = torch.ones(
        (time, time), dtype=torch.bool, device=z.device
    ).tril(diagonal=-1)
    mask = same_document & strict_past.unsqueeze(0)
    previous_sum = torch.bmm(mask.to(z.dtype), z)
    denominator = segpos.clamp_min(1).to(z.dtype).unsqueeze(-1)
    return previous_sum / denominator - z


def segmented_prefix_coordinator_value(z, segpos, segment_start):
    """
    Same coordinator value using a segmented prefix sum.
    segment_start stores each token's segment-start column index.
    """
    batch, time, width = z.shape
    inclusive = torch.cumsum(z, dim=1)
    before = torch.cat((torch.zeros_like(z[:, :1, :]), inclusive[:, :-1, :]), dim=1)

    start_index = segment_start.to(torch.long).clamp(0, time - 1)
    baseline = before.gather(
        1, start_index.unsqueeze(-1).expand(batch, time, width)
    )
    previous_sum = before - baseline
    denominator = segpos.clamp_min(1).to(z.dtype).unsqueeze(-1)
    return previous_sum / denominator - z


# Packed-document example: each segment ID is its first token's column.
torch.manual_seed(17)
device = torch.device("cpu")
dtype = torch.float64
batch, heads, time, key_dim, value_dim = 2, 2, 11, 7, 5

segment_start = torch.tensor(
    [
        [0, 0, 0, 0, 4, 4, 4, 7, 7, 7, 7],
        [0, 0, 2, 2, 2, 5, 5, 5, 5, 9, 9],
    ],
    dtype=torch.long,
    device=device,
)
positions = torch.arange(time, device=device).expand(batch, -1)
segpos = positions - segment_start

q0 = torch.randn(batch, heads, time, key_dim, dtype=dtype, device=device)
v0 = torch.randn(batch, time, value_dim, dtype=dtype, device=device)
probe = torch.randn(batch, heads, time, value_dim, dtype=dtype, device=device)

q_dense = q0.clone().requires_grad_(True)
v_dense = v0.clone().requires_grad_(True)
out_dense = dense_strict_past_attention(q_dense, v_dense, segment_start)
grad_q_dense, grad_v_dense = torch.autograd.grad(
    (out_dense * probe).sum(), (q_dense, v_dense)
)

q_scan = q0.clone().requires_grad_(True)
v_scan = v0.clone().requires_grad_(True)
out_scan = strict_past_state_scan_reference(q_scan, v_scan, segment_start)
grad_q_scan, grad_v_scan = torch.autograd.grad(
    (out_scan * probe).sum(), (q_scan, v_scan)
)

attention_errors = {
    "forward_max_abs": float((out_scan - out_dense).abs().max()),
    "q_gradient_max_abs": float((grad_q_scan - grad_q_dense).abs().max()),
    "v_gradient_max_abs": float((grad_v_scan - grad_v_dense).abs().max()),
}
torch.testing.assert_close(out_scan, out_dense, rtol=1e-10, atol=1e-10)
torch.testing.assert_close(grad_q_scan, grad_q_dense, rtol=1e-10, atol=1e-10)
torch.testing.assert_close(grad_v_scan, grad_v_dense, rtol=1e-10, atol=1e-10)

z0 = torch.randn(batch, time, value_dim, dtype=dtype, device=device)
coord_probe = torch.randn_like(z0)

z_dense = z0.clone().requires_grad_(True)
coord_dense = dense_coordinator_value(z_dense, segpos, segment_start)
grad_z_dense = torch.autograd.grad((coord_dense * coord_probe).sum(), z_dense)[0]

z_prefix = z0.clone().requires_grad_(True)
coord_prefix = segmented_prefix_coordinator_value(z_prefix, segpos, segment_start)
grad_z_prefix = torch.autograd.grad((coord_prefix * coord_probe).sum(), z_prefix)[0]

coordinator_errors = {
    "value_max_abs": float((coord_prefix - coord_dense).abs().max()),
    "z_gradient_max_abs": float((grad_z_prefix - grad_z_dense).abs().max()),
}
torch.testing.assert_close(coord_prefix, coord_dense, rtol=1e-10, atol=1e-10)
torch.testing.assert_close(grad_z_prefix, grad_z_dense, rtol=1e-10, atol=1e-10)

print({
    "status": "REFERENCE_CHECKS_PASSED",
    "scope": "tiny CPU correctness only; not a GPU timing implementation",
    "attention": attention_errors,
    "coordinator": coordinator_errors,
})
