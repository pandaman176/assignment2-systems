import torch
import pandas as pd
from itertools import product
from triton import testing as ttesting
from cs336_basics.model import scaled_dot_product_attention
from cs336_systems.flashattention2 import FlashAttention2_triton

def main():
    # Initialize benchmark parameters
    seq_lens = [2**i for i in range(7, 16)] # 128 -> 65536
    d_models = [2**i for i in range(4, 7)] # 16 -> 128
    percisions = [torch.bfloat16, torch.float32]
    percision_names = {torch.bfloat16: "bf16", torch.float32: "f32"}
    # Do benchmark for all combinotions of these parameters
    data = []
    for seq_len, d_model, precision in product(seq_lens, d_models, percisions):
        print(f"seq_len={seq_len}, d_model={d_model}, precision={percision_names[precision]}")
        ft_torch = None
        bt_torch = None
        fbt_torch = None
        ft_triton = None
        bt_triton = None
        fbt_triton = None

        ft_torch, bt_torch, fbt_torch, ft_triton, bt_triton, fbt_triton = run_benchmark(seq_len, d_model, precision)

        result = {
            "seq_len": seq_len,
            "d_model": d_model,
            "precision": percision_names[precision],    
            "ft_torch": ft_torch,
            "bt_torch": bt_torch,
            "fbt_torch": fbt_torch,
            "ft_triton": ft_triton,
            "bt_triton": bt_triton,
            "fbt_triton": fbt_triton,
        }

        data.append(result)
    
    df = pd.DataFrame(data)
    print(df)

def run_benchmark(seq_len, d_model, precision):
    Q = torch.randn(1, seq_len, d_model, device="cuda", dtype=precision)
    K = torch.randn(1, seq_len, d_model, device="cuda", dtype=precision)
    V = torch.randn(1, seq_len, d_model, device="cuda", dtype=precision)

    mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=Q.device))

    def _torch_fwd(_Q, _K, _V):
        return scaled_dot_product_attention(_Q, _K, _V, mask)
    
    def _triton_fwd(_Q, _K, _V):
        return FlashAttention2_triton.apply(_Q, _K, _V, is_causal=True)

    ft_p, bt_p, fbt_p = "OOM", "OOM", "OOM"
    ft_t, bt_t, fbt_t = "OOM", "OOM", "OOM"
    try:
        ft_p = test_forward(_torch_fwd, Q, K, V)
        # Clone tensors to clear gradients in compute graph
        bt_p = test_backward(_torch_fwd, Q.clone(), K.clone(), V.clone())
        fbt_p = test_tot(_torch_fwd, Q.clone(), K.clone(), V.clone())
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError as e:
        print(f"OOM(torch): {e} {seq_len=} {d_model=} {precision=}")
    finally:
        torch.cuda.empty_cache()
    
    try:
        ft_t = test_forward(_triton_fwd, Q, K, V)
        bt_t = test_backward(_triton_fwd, Q.clone(), K.clone(), V.clone())
        fbt_t = test_tot(_triton_fwd, Q.clone(), K.clone(), V.clone())
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError as e:
        print(f"OOM(triton): {e} {seq_len=} {d_model=} {precision=}")
    finally:
        torch.cuda.empty_cache()
    
    return ft_p, bt_p, fbt_p, ft_t, bt_t, fbt_t


WARMUP = 25
REP = 50

def test_forward(func, *args):
    return ttesting.do_bench(lambda: func(*args), warmup=WARMUP, rep=REP)

def test_backward(func, *args):
    # Open gradient
    for arg in args:
        arg.requires_grad_(True)
    # Run forward
    out = func(*args)
    grad_out = torch.randn_like(out)
    def _backward():
        out.backward(grad_out,retain_graph=True)
        for arg in args:
            arg.grad = None # Clear gradients for next recomputation
    
    return ttesting.do_bench(_backward, warmup=WARMUP, rep=REP)

def test_tot(func, *args):
    # Open gradient
    for arg in args:
        arg.requires_grad_(True)
    
    def _tot():
        out = func(*args)
        out.sum().backward()
        for arg in args:
            arg.grad = None # Clear gradients for next recomputation

    return ttesting.do_bench(_tot, warmup=WARMUP, rep=REP)

if __name__ == "__main__":
    main()