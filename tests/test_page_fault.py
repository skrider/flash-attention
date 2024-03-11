import pytest
import math
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from flash_attn import flash_attn_with_kvcache

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=np.inf)

# @pytest.mark.parametrize("dtype", ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
@pytest.mark.parametrize("dtype", [torch.float16])
# @pytest.mark.parametrize("num_splits", [1, 0])
@pytest.mark.parametrize("num_splits", [1])
# @pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("mha_type", ["mha"])
# @pytest.mark.parametrize("new_kv", [False, True])
@pytest.mark.parametrize("new_kv", [True])
# @pytest.mark.parametrize("alibi", [False, True])
@pytest.mark.parametrize("alibi", [False])
# @pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("local", [False])
# @pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("causal", [True])
# @pytest.mark.parametrize("seqlen_new_eq_seqlen_q", [True, False])
@pytest.mark.parametrize("seqlen_new_eq_seqlen_q", [True])
# @pytest.mark.parametrize("rotary_interleaved", [False, True])
@pytest.mark.parametrize("rotary_interleaved", [False])
# @pytest.mark.parametrize("rotary_fraction", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("rotary_fraction", [0.0])
# @pytest.mark.parametrize("paged_kv_block_size", [None, 256, 512])
@pytest.mark.parametrize("paged_kv_block_size", [16])
# @pytest.mark.parametrize("has_batch_idx", [False, True])
@pytest.mark.parametrize("has_batch_idx", [False])
@pytest.mark.parametrize("mark_keys", [False])
@pytest.mark.parametrize("interleave_kv", [False])
@pytest.mark.parametrize("niter", [1])
# @pytest.mark.parametrize("d", [32, 59, 64, 80, 128, 256])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [56, 80])
@pytest.mark.parametrize("d", [64])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        # (1, 128),
        # (1, 339),
        # (3, 1024),
        # (64, 800),
        # (64, 256),
        # (3, 799),
        # (64, 2048),
        # (16, 20000),
        # (1, 128 * 1024),
        # (16, 128 * 1024),
        (2, 128),
    ],
)
# @pytest.mark.parametrize('seqlen_q,seqlen_k', [ (2, 128) ])
@pytest.mark.parametrize('seqlen_new', [ 64 ])
def test_flash_attn_page_fault(
    seqlen_q,
    seqlen_k,
    seqlen_new,
    d,
    has_batch_idx,
    paged_kv_block_size,
    rotary_fraction,
    rotary_interleaved,
    seqlen_new_eq_seqlen_q,
    causal,
    local,
    alibi,
    new_kv,
    mha_type,
    num_splits,
    interleave_kv,
    mark_keys,
    niter,
    dtype,
):
    if not new_kv and niter > 1:
        pytest.skip()
    if seqlen_q > seqlen_k and new_kv:
        pytest.skip()
    if not new_kv and rotary_fraction > 0.0:
        pytest.skip()
    if has_batch_idx and paged_kv_block_size is not None:
        pytest.skip()
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 1
    batch_size_cache = batch_size if not has_batch_idx else batch_size * 2
    nheads = 1
    # rotary_dim must be a multiple of 16, and must be <= d
    rotary_dim = math.floor(int(rotary_fraction * d) / 16) * 16
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)
    assert nheads % nheads_k == 0
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype) * d
    if new_kv:
        k = torch.randn(batch_size, seqlen_new, nheads_k, d, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen_new, nheads_k, d, device=device, dtype=dtype)
        if mark_keys:
            for i in range(seqlen_new):
                k[:, i].fill_(-1.0 * i)
                v[:, i].fill_(-1.0 * i)
    else:
        k, v = None, None
    if paged_kv_block_size is None:
        k_cache = torch.randn(batch_size_cache, seqlen_k, nheads_k, d, device=device, dtype=dtype)
        v_cache = torch.randn(batch_size_cache, seqlen_k, nheads_k, d, device=device, dtype=dtype)
        block_table = None
    else:

        num_blocks = math.ceil(seqlen_k / paged_kv_block_size) * batch_size * 3
        if interleave_kv:
            kv_cache_paged = torch.randn(
                num_blocks, 2, paged_kv_block_size, nheads_k, d, device=device, dtype=dtype
            )
            k_cache_paged = kv_cache_paged[:, 0] * d
            v_cache_paged = kv_cache_paged[:, 1]
            pass
        else:
            k_cache_paged = torch.randn(
                num_blocks, paged_kv_block_size, nheads_k, d, device=device, dtype=dtype
            ) * d
            v_cache_paged = torch.randn(
                num_blocks, paged_kv_block_size, nheads_k, d, device=device, dtype=dtype
            )
        block_table = rearrange(
            torch.randperm(num_blocks, dtype=torch.int32, device=device),
            "(b nblocks) -> b nblocks",
            b=batch_size,
        )
        k_cache = rearrange(
            k_cache_paged[block_table.flatten()],
            "(b nblocks) block_size ... -> b (nblocks block_size) ...",
            b=batch_size,
        )[:, :seqlen_k]
        v_cache = rearrange(
            v_cache_paged[block_table.flatten()],
            "(b nblocks) block_size ... -> b (nblocks block_size) ...",
            b=batch_size,
        )[:, :seqlen_k]
        if mark_keys:
            for i in range(batch_size):
                for bn in range(num_blocks):
                    block_idx = block_table[i, bn]
                    for j in range(paged_kv_block_size):
                        k_cache_paged[block_idx, j].fill_(1.0 * (bn * paged_kv_block_size + j))
                        v_cache_paged[block_idx, j].fill_(1.0 * (bn * paged_kv_block_size + j))

    # seq_ids = torch.arange(batch_size, device=device, dtype=torch.int32) + 112342
    seq_ids = torch.ones(batch_size, device=device, dtype=torch.int32) * 0xefffffff
    page_fault_mask = torch.zeros_like(block_table).to(torch.int32)

    # cache_seqlens = torch.randint(
    #     0 if new_kv else 1,
    #     # If we don't use seqlen_q in the case of causal and rotary, cos/sin won't be long enough
    #     (seqlen_k - (seqlen_q if (causal or local) and rotary_dim > 1 else seqlen_new) + 1)
    #     if new_kv
    #     else (seqlen_k + 1),
    #     (batch_size,),
    #     dtype=torch.int32,
    #     device=device,
    # )
    cache_seqlens = torch.ones((batch_size,), dtype=torch.int32, device=device) * (seqlen_k - seqlen_new)
    arange = rearrange(torch.arange(seqlen_k, device=device), "s -> 1 s")
    cache_seqlens_expanded = rearrange(cache_seqlens, "b -> b 1")
    key_padding_mask = arange < cache_seqlens_expanded + (seqlen_new if new_kv else 0)
    if has_batch_idx:
        cache_batch_idx = torch.randperm(batch_size_cache, dtype=torch.int32, device=device)[
            :batch_size
        ]
    else:
        cache_batch_idx = None
    if alibi:
        alibi_slopes = torch.rand(batch_size, nheads, device=device, dtype=torch.float32) * 0.3
        attn_bias = attn_bias_from_alibi_slopes(
            alibi_slopes, seqlen_q, seqlen_k, None, key_padding_mask, causal=causal
        )
    else:
        alibi_slopes, attn_bias = None, None
    # cache_seqlens = torch.tensor([32], dtype=torch.int32, device=device)
    cos, sin = None, None
    q_ro, k_ro = q, k
    k_cache_ref = (k_cache if not has_batch_idx else k_cache[cache_batch_idx]).clone()
    v_cache_ref = (v_cache if not has_batch_idx else v_cache[cache_batch_idx]).clone()
    if new_kv:
        update_mask = torch.logical_and(
            cache_seqlens_expanded <= arange, arange < cache_seqlens_expanded + seqlen_new
        )
        k_cache_ref[update_mask] = rearrange(k_ro, "b s ... -> (b s) ...")
        v_cache_ref[update_mask] = rearrange(v, "b s ... -> (b s) ...")
    k_cache_rep = repeat(k_cache_ref, "b s h d -> b s (h g) d", g=nheads // nheads_k)
    v_cache_rep = repeat(v_cache_ref, "b s h d -> b s (h g) d", g=nheads // nheads_k)
    assert seqlen_new % niter == 0
    for i in range(niter):
        k_new_start = (seqlen_new // niter) * i
        k_new_len = seqlen_new // niter
        print("RUN", i)
        out = flash_attn_with_kvcache(
            q,
            k_cache if paged_kv_block_size is None else k_cache_paged,
            v_cache if paged_kv_block_size is None else v_cache_paged,
            k[:, k_new_start:k_new_start + k_new_len],
            v[:, k_new_start:k_new_start + k_new_len],
            rotary_cos=cos,
            rotary_sin=sin,
            cache_seqlens=cache_seqlens,
            cache_batch_idx=cache_batch_idx,
            block_table=block_table,
            causal=causal,
            window_size=window_size,
            rotary_interleaved=rotary_interleaved,
            alibi_slopes=alibi_slopes,
            num_splits=num_splits,
            seq_ids=seq_ids,
            page_fault_mask=page_fault_mask,
        )
        torch.cuda.synchronize()
        page_faults_ref = (seqlen_k - seqlen_new + paged_kv_block_size - 1) // paged_kv_block_size

        assert torch.all(page_fault_mask[:, :page_faults_ref] == 1)
        assert torch.all(page_fault_mask[:, page_faults_ref:] == 0)

        cache_seqlens += k_new_len
    
    out_ref, _ = attention_ref(
        q_ro,
        k_cache_rep,
        v_cache_rep,
        None,
        key_padding_mask,
        attn_bias,
        0.0,
        None,
        causal=causal,
        window_size=window_size,
    )
    # take one bit out of every 8 bytes
    mask = ~np.array([0x0, 0x0, 0x0, 0x1], dtype=np.uint16)
    out_pt, _ = attention_ref(
        q_ro,
        k_cache_rep,
        v_cache_rep,
        None,
        key_padding_mask,
        attn_bias,
        0.0,
        None,
        causal=causal,
        window_size=window_size,
        upcast=False,
        reorder_ops=True,
        mask=mask,
    )
    print(out)
    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
    print(f"Output/Pytorch max diff: {(out_pt - out).abs().max().item()}")
    print(f"Output/Pytorch mean diff: {(out_pt - out).abs().mean().item()}")

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    if new_kv:
        if paged_kv_block_size is None:
            k_cache_select = k_cache if not has_batch_idx else k_cache[cache_batch_idx]
            v_cache_select = v_cache if not has_batch_idx else v_cache[cache_batch_idx]
        else:
            k_cache_select = rearrange(
                k_cache_paged[block_table.flatten()],
                "(b nblocks) block_size ... -> b (nblocks block_size) ...",
                b=batch_size,
            )[:, :seqlen_k]
            v_cache_select = rearrange(
                v_cache_paged[block_table.flatten()],
                "(b nblocks) block_size ... -> b (nblocks block_size) ...",
                b=batch_size,
            )[:, :seqlen_k]
        assert torch.allclose(k_cache_select, k_cache_ref, rtol=1e-3, atol=1e-3)
    mult = 3 if not alibi else 5
    assert (out - out_ref).abs().max().item() <= mult * (out_pt - out_ref).abs().max().item() + 1e-5

def bitwise_mask_half(v, mask: np.ndarray) -> torch.Tensor:
    assert mask[0].nbytes == 2
    assert v.dtype == torch.float16
    v_cpu = v.flatten().reshape(-1, mask.size).cpu().numpy()
    v_masked = np.bitwise_and(v_cpu.view(np.uint16), mask).view(np.half)
    v_masked_t = torch.from_numpy(v_masked).reshape(v.shape).cuda()
    return v_masked_t

def attn_bias_from_alibi_slopes(
    slopes, seqlen_q, seqlen_k, query_padding_mask=None, key_padding_mask=None, causal=False
):
    batch, nheads = slopes.shape
    device = slopes.device
    slopes = rearrange(slopes, "b h -> b h 1 1")
    if causal:
        return torch.arange(-seqlen_k + 1, 1, device=device, dtype=torch.float32) * slopes
    else:
        row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
        col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
        sk = (
            seqlen_k
            if key_padding_mask is None
            else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
        )
        sq = (
            seqlen_q
            if query_padding_mask is None
            else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
        )
        relative_pos = torch.abs(row_idx + sk - sq - col_idx)
        return -slopes * relative_pos.to(dtype=slopes.dtype)
    
def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
):
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )

def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    upcast=True,
    reorder_ops=False,
    mask=None,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling k, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if mask is not None:
        assert q.dtype == torch.half
        # q = bitwise_mask_half(q, mask)
        k = bitwise_mask_half(k, mask)
        v = bitwise_mask_half(v, mask)
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)
