import torch
from torch import Tensor
import math
import triton
import triton.language as tl
from einops import rearrange, repeat, einsum
from jaxtyping import Float

class FlashAttention2_torch(torch.autograd.Function):
    """
    FlashAttention2_torch is a pure pytorch implementation of FlashAttention2.
    """

    @staticmethod
    def forward(
        ctx, 
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys    d_k"],
        V: Float[Tensor, " ... keys    d_v"],
        is_causal=False
    ) -> Float[Tensor, " ... queries d_v"]:
        """
        Forward pass of FlashAttention2_torch.
        """
        # Save original shapes
        original_shape = Q.shape
        seq_len, head_dim = Q.shape[-2:]

        q_tile_size = min(128, seq_len)
        kv_tile_size = min(128, seq_len)

        # tile Q, K, V
        Q_tiled = rearrange(Q, "... (tq bq) d -> (...) tq bq d", bq=q_tile_size)
        K_tiled = rearrange(K, "... (tk bk) d -> (...) tk bk d", bk=kv_tile_size)
        V_tiled = rearrange(V, "... (tv bv) d -> (...) tv bv d", bv=kv_tile_size)

        # output tensor
        Out = torch.zeros(*original_shape[:-2], seq_len, head_dim, device=Q.device)
        L = torch.zeros(*original_shape[:-2], seq_len, device=Q.device)

        num_q_tiles =  Q_tiled.shape[1]
        num_k_tiles = K_tiled.shape[1]

        for i in range(num_q_tiles):
            Q_i = Q_tiled[:,i]
            # output tile
            O_i = torch.zeros(q_tile_size, head_dim, device=Q.device)
            # running proxy for the softmax denominator
            l_i = torch.zeros(q_tile_size, device=Q.device)
            # runing maximun for denominator
            m_i = torch.full((q_tile_size,), -torch.inf, device=Q.device)
            for j in range(num_k_tiles):
                K_j = K_tiled[:,j]
                V_j = V_tiled[:,j]
                # pre softmax attention scores tile
                # S_ij = Q_i @ K_j.T / math.sqrt(d)
                S_ij = einsum(Q_i, K_j, "b q d, b k d -> b q k") / math.sqrt(head_dim)
                old_m_i = m_i.clone() # need to save this for l_i
                m_i = torch.max(m_i, S_ij.max(dim=-1).values)
                P_ij = torch.exp(S_ij - m_i.unsqueeze(-1)) 
                exp_max_diff = torch.exp(old_m_i - m_i)
                l_i = exp_max_diff * l_i + P_ij.sum(dim=-1)
                weight_values_tile = einsum(P_ij, V_j, "b q k, b k d -> b q d")
                O_i = torch.diag_embed(exp_max_diff) @ O_i + weight_values_tile
            O_i = torch.diag_embed(1.0 / l_i) @ O_i # final normalization
            L_i = m_i + torch.log(l_i)

            Out[i * q_tile_size:(i + 1) * q_tile_size] = O_i
            L[i * q_tile_size:(i + 1) * q_tile_size] = L_i
        
        ctx.save_for_backward(Q, K, V, L, Out)

        return Out

    @staticmethod
    def backward(ctx, grad_out):
        """
        Backward pass of FlashAttention2_torch.
        """
        raise NotImplementedError("Backward pass is not implemented")

class FlashAttention2_triton(torch.autograd.Function):
    """
    FlashAttention2_triton is a triton implementation of FlashAttention2.
    """

    @staticmethod
    def forward(
        ctx, 
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys    d_k"],
        V: Float[Tensor, " ... keys    d_v"],
        is_causal=False
    ) -> Float[Tensor, " ... queries d_v"]:
        # Save original shapes
        original_shape = Q.shape
        seq_len, head_dim = Q.shape[-2:]
        B = math.prod(original_shape[:-2])

        assert Q.shape[-2] == K.shape[-2] == V.shape[-2]
        q_tile_size = min(64, seq_len)
        kv_tile_size = min(64, seq_len)
        q_tile_num = triton.cdiv(seq_len, q_tile_size)

        # Flatten QKV
        Qf = rearrange(Q, "... s h -> (...) s h")
        Kf = rearrange(K, "... s h -> (...) s h")
        Vf = rearrange(V, "... s h -> (...) s h")

        # Allocate output
        Out = torch.empty(*original_shape[:-2], seq_len, head_dim, device=Q.device)
        L = torch.empty(*original_shape[:-2], seq_len, device=Q.device)
        Outf = rearrange(Out, "... s h -> (...) s h")
        Lf = rearrange(L, "... s -> (...) s")

        scale = 1.0/math.sqrt(head_dim)

        # launch flash attention kernel
        flash_fwd_kernel[(q_tile_num, B)](
            Qf, Kf, Vf,
            Outf, Lf,
            *Qf.stride(),
            *Kf.stride(),
            *Vf.stride(),
            *Outf.stride(),
            *Lf.stride(),
            N_QUERIES=seq_len,
            N_KEYS=seq_len,
            scale=scale,
            D=head_dim,
            Q_TILE_SIZE=q_tile_size,
            K_TILE_SIZE=kv_tile_size,
        )

        ctx.save_for_backward(Q, K, V, L, Out)

        return Out

    @staticmethod
    def backward(ctx, grad_out):
        ...


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        # eache thread access one Q tile
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0), # each thread need access to all K tiles
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0), # each thread need access to all V tiles
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Q_tile = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero")
        
    # Initialize online-softmax state
    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)

    for i in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_tile = tl.load(K_block_ptr, boundary_check=(0,1), padding_option="zero")
        V_tile = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero")

        S = tl.dot(Q_tile, tl.trans(K_tile)) * scale
        new_m_i = tl.maximum(m_i, tl.max(S, axis=-1))
        P = tl.exp(S - new_m_i[:,None])
        exp_max_diff = tl.exp(m_i - new_m_i)
        l_i = exp_max_diff * l_i + tl.sum(P, axis=-1)
        weight_values_tile = tl.dot(P.to(V_tile.dtype), V_tile)
        O_i = exp_max_diff[:,None] * O_i + weight_values_tile

        # Advance pointers
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
        m_i = new_m_i

    # Final normalization
    O_i = (O_i / l_i[:,None]).to(O_block_ptr.type.element_ty)
    L_i = (m_i + tl.log(l_i)).to(tl.float32)

    # Store output
    tl.store(O_block_ptr, O_i, boundary_check=(0,1))
    tl.store(L_block_ptr, L_i, boundary_check=(0,))
        


