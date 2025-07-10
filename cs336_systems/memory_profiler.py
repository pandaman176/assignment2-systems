#!/usr/bin/env python3
"""
简化的内存分析脚本：比较使用和不使用优化器状态分片的内存使用情况
适用于单GPU环境
"""

import torch
import torch.nn as nn
import os
import gc
import psutil
import numpy as np
from typing import Dict, List, Tuple, Any
import argparse
from dataclasses import dataclass
import json
from contextlib import nullcontext

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW


@dataclass
class MemorySnapshot:
    """内存快照数据结构"""
    phase: str
    total_memory_mb: float
    allocated_memory_mb: float
    cached_memory_mb: float
    parameters_memory_mb: float
    optimizer_states_memory_mb: float
    gradients_memory_mb: float
    activations_memory_mb: float


def get_memory_info() -> dict[str, float]:
    """获取当前内存使用情况"""
    if torch.cuda.is_available():
        # GPU内存信息
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        cached = torch.cuda.memory_reserved() / 1024**2  # MB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
    else:
        # CPU内存信息
        process = psutil.Process()
        memory_info = process.memory_info()
        allocated = memory_info.rss / 1024**2  # MB
        cached = allocated
        total = psutil.virtual_memory().total / 1024**2  # MB
    
    return {
        "total_memory_mb": total,
        "allocated_memory_mb": allocated,
        "cached_memory_mb": cached
    }


def estimate_model_memory_breakdown(model: nn.Module, optimizer: torch.optim.Optimizer = None) -> dict[str, float]:
    """估算模型内存分解"""
    device = next(model.parameters()).device
    
    # 参数内存
    parameters_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    
    # 优化器状态内存
    optimizer_states_memory = 0.0
    if optimizer is not None:
        for group in optimizer.param_groups:
            for p in group['params']:
                if p in optimizer.state:
                    state = optimizer.state[p]
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor):
                            optimizer_states_memory += value.numel() * value.element_size()
        optimizer_states_memory /= 1024**2
    
    # 梯度内存
    gradients_memory = sum(p.numel() * p.element_size() for p in model.parameters() if p.grad is not None) / 1024**2
    
    # 激活内存（粗略估算）
    parameters_memory_mb = parameters_memory
    activations_memory = parameters_memory_mb * 2.0  # 假设激活内存约为参数内存的2倍
    
    return {
        "parameters_memory_mb": parameters_memory,
        "optimizer_states_memory_mb": optimizer_states_memory,
        "gradients_memory_mb": gradients_memory,
        "activations_memory_mb": activations_memory
    }


def take_memory_snapshot(phase: str, model: nn.Module, optimizer: torch.optim.Optimizer = None) -> MemorySnapshot:
    """获取内存快照"""
    # 强制垃圾回收
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 获取基本内存信息
    memory_info = get_memory_info()
    
    # 估算内存分解
    if model is not None:
        breakdown = estimate_model_memory_breakdown(model, optimizer)
    else:
        breakdown = {
            "parameters_memory_mb": 0.0,
            "optimizer_states_memory_mb": 0.0,
            "gradients_memory_mb": 0.0,
            "activations_memory_mb": 0.0
        }
    
    return MemorySnapshot(
        phase=phase,
        total_memory_mb=memory_info["total_memory_mb"],
        allocated_memory_mb=memory_info["allocated_memory_mb"],
        cached_memory_mb=memory_info["cached_memory_mb"],
        parameters_memory_mb=breakdown["parameters_memory_mb"],
        optimizer_states_memory_mb=breakdown["optimizer_states_memory_mb"],
        gradients_memory_mb=breakdown["gradients_memory_mb"],
        activations_memory_mb=breakdown["activations_memory_mb"]
    )


def print_memory_snapshot(snapshot: MemorySnapshot):
    """打印内存快照"""
    print(f"\n=== {snapshot.phase} ===")
    print(f"总内存: {snapshot.total_memory_mb:.2f} MB")
    print(f"已分配内存: {snapshot.allocated_memory_mb:.2f} MB")
    print(f"缓存内存: {snapshot.cached_memory_mb:.2f} MB")
    print("内存分解:")
    print(f"  - 参数: {snapshot.parameters_memory_mb:.2f} MB")
    print(f"  - 优化器状态: {snapshot.optimizer_states_memory_mb:.2f} MB")
    print(f"  - 梯度: {snapshot.gradients_memory_mb:.2f} MB")
    print(f"  - 激活: {snapshot.activations_memory_mb:.2f} MB")


def run_memory_profiling_experiment(
    model_config: dict[str, int],
    vocab_size: int = 10000,
    batch_size: int = 1,
    seq_len: int = 1024,
    device: torch.device = None
) -> list[MemorySnapshot]:
    """运行内存分析实验"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    snapshots = []
    
    print("开始内存分析实验...")
    print(f"模型配置: {model_config}")
    print(f"设备: {device}")
    
    # 1. 模型初始化前
    snapshot = take_memory_snapshot("模型初始化前", None)
    snapshots.append(snapshot)
    print_memory_snapshot(snapshot)
    
    # 2. 创建模型
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=seq_len,
        rope_theta=10000.0,
        **model_config
    ).to(device)
    
    snapshot = take_memory_snapshot("模型初始化后", model)
    snapshots.append(snapshot)
    print_memory_snapshot(snapshot)
    
    # 3. 创建优化器
    optimizer = AdamW(model.parameters(), lr=1e-4)
    
    snapshot = take_memory_snapshot("优化器创建后", model, optimizer)
    snapshots.append(snapshot)
    print_memory_snapshot(snapshot)
    
    # 4. 前向传播
    model.train()
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    outputs = model(inputs)
    
    snapshot = take_memory_snapshot("前向传播后", model, optimizer)
    snapshots.append(snapshot)
    print_memory_snapshot(snapshot)
    
    # 5. 反向传播
    loss = outputs.mean()
    loss.backward()
    
    snapshot = take_memory_snapshot("反向传播后", model, optimizer)
    snapshots.append(snapshot)
    print_memory_snapshot(snapshot)
    
    # 6. 优化器步骤前
    snapshot = take_memory_snapshot("优化器步骤前", model, optimizer)
    snapshots.append(snapshot)
    print_memory_snapshot(snapshot)
    
    # 7. 优化器步骤
    optimizer.step()
    
    snapshot = take_memory_snapshot("优化器步骤后", model, optimizer)
    snapshots.append(snapshot)
    print_memory_snapshot(snapshot)
    
    return snapshots


def simulate_sharded_memory_usage(snapshots: list[MemorySnapshot], world_size: int = 2) -> list[MemorySnapshot]:
    """模拟分片后的内存使用情况"""
    sharded_snapshots = []
    
    for snapshot in snapshots:
        # 创建分片后的快照
        sharded_snapshot = MemorySnapshot(
            phase=snapshot.phase,
            total_memory_mb=snapshot.total_memory_mb,
            allocated_memory_mb=snapshot.allocated_memory_mb,
            cached_memory_mb=snapshot.cached_memory_mb,
            parameters_memory_mb=snapshot.parameters_memory_mb,  # 参数内存不变
            optimizer_states_memory_mb=snapshot.optimizer_states_memory_mb / world_size,  # 优化器状态分片
            gradients_memory_mb=snapshot.gradients_memory_mb,  # 梯度内存不变
            activations_memory_mb=snapshot.activations_memory_mb  # 激活内存不变
        )
        sharded_snapshots.append(sharded_snapshot)
    
    return sharded_snapshots


def compare_memory_usage(snapshots_no_sharding: list[MemorySnapshot], 
                        snapshots_with_sharding: list[MemorySnapshot]) -> dict[str, Any]:
    """比较内存使用情况"""
    comparison = {
        "model_initialization": {},
        "optimizer_creation": {},
        "forward_pass": {},
        "backward_pass": {},
        "before_optimizer_step": {},
        "after_optimizer_step": {}
    }
    
    phases = ["模型初始化后", "优化器创建后", "前向传播后", "反向传播后", "优化器步骤前", "优化器步骤后"]
    
    for i, phase in enumerate(phases):
        no_shard = snapshots_no_sharding[i+1]  # +1 因为第一个是初始化前
        with_shard = snapshots_with_sharding[i+1]
        
        comparison[list(comparison.keys())[i]] = {
            "no_sharding_allocated_mb": no_shard.allocated_memory_mb,
            "with_sharding_allocated_mb": with_shard.allocated_memory_mb,
            "memory_saved_mb": no_shard.allocated_memory_mb - with_shard.allocated_memory_mb,
            "memory_saved_percent": ((no_shard.allocated_memory_mb - with_shard.allocated_memory_mb) / no_shard.allocated_memory_mb) * 100,
            "optimizer_states_no_sharding_mb": no_shard.optimizer_states_memory_mb,
            "optimizer_states_with_sharding_mb": with_shard.optimizer_states_memory_mb,
            "optimizer_states_saved_mb": no_shard.optimizer_states_memory_mb - with_shard.optimizer_states_memory_mb,
            "optimizer_states_saved_percent": ((no_shard.optimizer_states_memory_mb - with_shard.optimizer_states_memory_mb) / no_shard.optimizer_states_memory_mb) * 100 if no_shard.optimizer_states_memory_mb > 0 else 0
        }
    
    return comparison


def print_comparison_results(comparison: dict[str, Any]):
    """打印比较结果"""
    print("\n" + "="*80)
    print("内存使用比较结果")
    print("="*80)
    
    for phase, data in comparison.items():
        print(f"\n{phase}:")
        print(f"  无分片已分配内存: {data['no_sharding_allocated_mb']:.2f} MB")
        print(f"  有分片已分配内存: {data['with_sharding_allocated_mb']:.2f} MB")
        print(f"  节省内存: {data['memory_saved_mb']:.2f} MB ({data['memory_saved_percent']:.1f}%)")
        print(f"  优化器状态无分片: {data['optimizer_states_no_sharding_mb']:.2f} MB")
        print(f"  优化器状态有分片: {data['optimizer_states_with_sharding_mb']:.2f} MB")
        print(f"  优化器状态节省: {data['optimizer_states_saved_mb']:.2f} MB ({data['optimizer_states_saved_percent']:.1f}%)")


def save_results_to_json(snapshots_no_sharding: list[MemorySnapshot], 
                        snapshots_with_sharding: list[MemorySnapshot],
                        comparison: dict[str, Any],
                        filename: str = "memory_profiling_results.json"):
    """保存结果到JSON文件"""
    def snapshot_to_dict(snapshot):
        return {
            "phase": snapshot.phase,
            "total_memory_mb": snapshot.total_memory_mb,
            "allocated_memory_mb": snapshot.allocated_memory_mb,
            "cached_memory_mb": snapshot.cached_memory_mb,
            "parameters_memory_mb": snapshot.parameters_memory_mb,
            "optimizer_states_memory_mb": snapshot.optimizer_states_memory_mb,
            "gradients_memory_mb": snapshot.gradients_memory_mb,
            "activations_memory_mb": snapshot.activations_memory_mb
        }
    
    results = {
        "experiment_info": {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "world_size": 2  # 模拟2个GPU
        },
        "snapshots_no_sharding": [snapshot_to_dict(s) for s in snapshots_no_sharding],
        "snapshots_with_sharding": [snapshot_to_dict(s) for s in snapshots_with_sharding],
        "comparison": comparison
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到 {filename}")


def main():
    parser = argparse.ArgumentParser(description="简化的内存分析脚本")
    parser.add_argument("--model-size", type=str, default="xl", 
                       choices=["sm", "md", "lg", "xl", "2.7b"],
                       help="模型大小预设")
    parser.add_argument("--vocab-size", type=int, default=10000,
                       help="词汇表大小")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="批次大小")
    parser.add_argument("--seq-len", type=int, default=1024,
                       help="序列长度")
    parser.add_argument("--output", type=str, default="memory_profiling_results.json",
                       help="输出文件名")
    parser.add_argument("--world-size", type=int, default=2,
                       help="模拟的GPU数量")
    
    args = parser.parse_args()
    
    # 模型配置预设
    model_configs = {
        "sm": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
        "md": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
        "lg": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
        "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
        "2.7b": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    }
    
    model_config = model_configs[args.model_size]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("开始内存分析实验")
    print(f"模型大小: {args.model_size}")
    print(f"模型配置: {model_config}")
    print(f"设备: {device}")
    print(f"模拟GPU数量: {args.world_size}")
    
    # 运行实际实验（无分片）
    print("\n" + "="*50)
    print("运行实际实验（无分片）")
    print("="*50)
    snapshots_no_sharding = run_memory_profiling_experiment(
        model_config, args.vocab_size, args.batch_size, args.seq_len, device
    )
    
    # 模拟分片后的内存使用
    print("\n" + "="*50)
    print("模拟分片后的内存使用")
    print("="*50)
    snapshots_with_sharding = simulate_sharded_memory_usage(snapshots_no_sharding, args.world_size)
    
    # 打印分片后的结果
    for snapshot in snapshots_with_sharding:
        print_memory_snapshot(snapshot)
    
    # 比较结果
    comparison = compare_memory_usage(snapshots_no_sharding, snapshots_with_sharding)
    print_comparison_results(comparison)
    
    # 保存结果
    save_results_to_json(snapshots_no_sharding, snapshots_with_sharding, comparison, args.output)


if __name__ == "__main__":
    main() 