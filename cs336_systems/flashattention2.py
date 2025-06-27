import torch
import torch.nn as nn
from torch import Tensor
import math
import triton.language as tl
from einops import rearrange, repeat, einsum
from typing import Optional, Tuple
from jaxtyping import Float, Bool

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
        batch_size = original_shape[0] if len(original_shape) == 3 else 1
        
        # reshape Q, K, V to 2D
        Q_2d = rearrange(Q, "... d_k -> (...) d_k") # (N_q, d)
        K_2d = rearrange(K, "... d_k -> (...) d_k") # (N_k, d)
        V_2d = rearrange(V, "... d_v -> (...) d_v") # (N_k, d)

        N_q, d = Q_2d.shape
        N_k, _ = K_2d.shape
        q_tile_size = min(128, N_q)
        kv_tile_size = min(128, N_k)
        T_q = math.ceil(N_q / q_tile_size)
        T_k = math.ceil(N_k / kv_tile_size)
        assert Q_2d.shape[1] == K_2d.shape[1] == V_2d.shape[1], "dimension must be the same for Q, K, V"
        assert K_2d.shape[0] == V_2d.shape[0], "K and V must have the same number of rows"
        assert N_q % q_tile_size == 0, "N_q must be divisible by q_tile_size"
        assert N_k % kv_tile_size == 0, "N_k must be divisible by kv_tile_size"
        # ignore padding first

        # tile Q, K, V
        Q_tiled = rearrange(Q_2d, "(tq bq) d -> tq bq d", tq=T_q)
        K_tiled = rearrange(K_2d, "(tk bk) d -> tk bk d", tk=T_k)
        V_tiled = rearrange(V_2d, "(tk bk) d -> tk bk d", tk=T_k)

        # output tensor
        O = torch.zeros(N_q, d)
        L = torch.zeros(N_q)

        O_tiled = rearrange(O, "(tq bq) d -> tq bq d", tq=T_q)
        L_tiled = rearrange(L, "(tq bq) -> tq bq", tq=T_q)

        for i in range(T_q):
            Q_i = Q_tiled[i]
            # output tile
            O_i = torch.zeros(q_tile_size, d)
            # running proxy for the softmax denominator
            l_i = torch.zeros(q_tile_size)
            # runing maximun for denominator
            m_i = torch.full((q_tile_size,), -torch.inf)
            for j in range(T_k):
                K_j = K_tiled[j]
                V_j = V_tiled[j]
                # pre softmax attention scores tile
                # S_ij = Q_i @ K_j.T / math.sqrt(d)
                S_ij = einsum(Q_i, K_j, "q d, k d -> q k") / math.sqrt(d)
                assert S_ij.shape == (q_tile_size, kv_tile_size), "S_ij must be of shape (q_tile_size, kv_tile_size)"
                old_m_i = m_i.clone() # need to save this for l_i
                m_i = torch.max(m_i, S_ij.max(dim=-1).values)
                P_ij = torch.exp(S_ij - m_i.unsqueeze(-1)) 
                exp_max_diff = torch.exp(old_m_i - m_i)
                assert P_ij.shape == (q_tile_size, kv_tile_size), "P_ij must be of shape (q_tile_size, kv_tile_size)"
                l_i = exp_max_diff * l_i + P_ij.sum(dim=-1)
                assert l_i.shape == (q_tile_size,), "l_i must be of shape (q_tile_size,)"
                weight_values_tile = einsum(P_ij, V_j, "q k, k d -> q d")
                O_i = torch.diag(exp_max_diff) @ O_i + weight_values_tile
            O_i = torch.diag(1.0 / l_i) @ O_i # final normalization
            L_i = m_i + torch.log(l_i)

            O[i * q_tile_size:(i + 1) * q_tile_size] = O_i
            L[i * q_tile_size:(i + 1) * q_tile_size] = L_i
        
        # Reshape L to match expected shape (batch_size, n_queries)
        L_reshaped = rearrange(L, "(b n) -> b n", b=batch_size)
        
        # Reshape O to match original input shape
        O_reshaped = rearrange(O, "(b n) d -> b n d", b=batch_size)
        assert O_reshaped.shape == original_shape, "O_reshaped must be of shape " + str(original_shape)
        
        ctx.save_for_backward(Q, K, V, L_reshaped, O_reshaped)

        return O_reshaped

    @staticmethod
    def backward(ctx, grad_out):
        """
        Backward pass of FlashAttention2_torch.
        """
        raise NotImplementedError("Backward pass is not implemented")