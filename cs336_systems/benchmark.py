import timeit
import torch
import torch.nn as nn
import os
import numpy as np
from loguru import logger
from typing import Any, Dict
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from cs336_basics.optimizer import AdamW
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark runner")
    parser.add_argument("--use-backward", action="store_true",
                        help="Whether to include backward pass in benchmark")
    parser.add_argument("--num", type=int, default=3,
                        help="Number of benchmark(default: 3)")
    parser.add_argument("--warm-steps", type=int, default=5,
                        help="Number of warm steps(default: 5)")
    return parser.parse_args()

_PRESETS: dict[str, dict[str, int]] = {
    "sm": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "md": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "lg": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7b": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

class BenchmarkAnalyzer:
    def __init__(self):
        self.results_list = []
        self.df = None
    
    def add_result(self, results: Dict[str, Any]):
        """添加单次benchmark结果"""
        # 提取主要统计信息，避免存储列表数据
        summary_result = {
            "model_name": results["model_name"],
            "avg_forward_time": results["avg_forward_time"],
            "total_forward_time": results["total_forward_time"],
            "std_forward_time": results["std_forward_time"],
            
        }
        
        # 如果包含backward信息
        if "avg_backward_time" in results:
            summary_result.update({
                "avg_backward_time": results["avg_backward_time"],
                "total_backward_time": results["total_backward_time"],
                "avg_total_time": results["avg_total_time"],
                "std_backward_time": results["std_backward_time"],
                "std_total_time": results["std_total_time"],
            })
        
        self.results_list.append(summary_result)
        self._update_dataframe()
    
    def _update_dataframe(self):
        """更新DataFrame"""
        self.df = pd.DataFrame(self.results_list)
    
    def get_summary_table(self):
        """获取汇总表格"""
        if self.df is None or self.df.empty:
            return None
        
        # 按模型分组统计
        agg_dict = {
            'avg_forward_time': ['mean', 'std', 'min', 'max'],
            'total_forward_time': ['mean', 'std'],
        }

        if 'avg_backward_time' in self.df.columns:
            agg_dict['avg_backward_time'] = ['mean', 'std']
        if 'avg_total_time' in self.df.columns:
            agg_dict['avg_total_time'] = ['mean', 'std']

        summary = self.df.groupby('model_name').agg(agg_dict).round(4)
        
        # 展平多级列名
        summary.columns = [f"{col[0]}_{col[1]}" for col in summary.columns]
        return summary
    
    def plot_performance_comparison(self, figsize=(12, 8)):
        """绘制性能对比图"""
        if self.df is None or self.df.empty:
            print("没有数据可绘制")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Forward time comparison
        self.df.boxplot(column='avg_forward_time', by='model_name', ax=axes[0,0])
        axes[0,0].set_title('Average Forward Time by Model')
        axes[0,0].set_xlabel('Model')
        axes[0,0].set_ylabel('Time (s)')
        
        # Forward time trend
        for model in self.df['model_name'].unique():
            model_data = self.df[self.df['model_name'] == model]
            axes[0,1].plot(range(len(model_data)), model_data['avg_forward_time'], 
                          marker='o', label=model)
        axes[0,1].set_title('Forward Time Trend')
        axes[0,1].set_xlabel('Benchmark Run')
        axes[0,1].set_ylabel('Time (s)')
        axes[0,1].legend()
        
        # Backward time (if available)
        if 'avg_backward_time' in self.df.columns:
            self.df.boxplot(column='avg_backward_time', by='model_name', ax=axes[1,0])
            axes[1,0].set_title('Average Backward Time by Model')
            axes[1,0].set_xlabel('Model')
            axes[1,0].set_ylabel('Time (s)')
            
            # Total time comparison
            for model in self.df['model_name'].unique():
                model_data = self.df[self.df['model_name'] == model]
                axes[1,1].plot(range(len(model_data)), model_data['avg_total_time'], 
                              marker='s', label=model)
            axes[1,1].set_title('Total Time Trend')
            axes[1,1].set_xlabel('Benchmark Run')
            axes[1,1].set_ylabel('Time (s)')
            axes[1,1].legend()
        else:
            # 如果没有backward数据，显示总的forward时间
            axes[1,0].bar(range(len(self.df)), self.df['total_forward_time'])
            axes[1,0].set_title('Total Forward Time')
            axes[1,0].set_xlabel('Benchmark Run')
            axes[1,0].set_ylabel('Time (s)')
            
            axes[1,1].axis('off')  # 隐藏第四个子图
        
        plt.tight_layout()
        plt.show()
    
    def export_to_markdown(self, filename="benchmark_results.md"):
        """导出结果到markdown文件"""
        if self.df is not None:
            self.df.to_markdown(filename, index=False)
            print(f"结果已导出到 {filename}")
        else:
            print("没有数据可导出")
    
    def get_best_performance(self):
        """获取最佳性能结果"""
        if self.df is None or self.df.empty:
            return None
        
        best_forward = self.df.loc[self.df['avg_forward_time'].idxmin()]
        result = {"best_forward": best_forward}
        
        if 'avg_backward_time' in self.df.columns:
            best_backward = self.df.loc[self.df['avg_backward_time'].idxmin()]
            best_total = self.df.loc[self.df['avg_total_time'].idxmin()]
            result.update({
                "best_backward": best_backward,
                "best_total": best_total
            })
        
        return result

def run_benchmark(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    warmup_steps=10,
    total_steps=100,
    include_backward=False,
    vocab_size=10000,
    batch_size=32,
    seq_len=1024,
    model_name="basics",
    device: torch.device | None = None,
) -> dict:
    logger.info(f"Warmup steps: {warmup_steps} Total steps: {total_steps}")
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    outputs = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    forward_times = []
    if include_backward:
        backward_times = []
    model.train(include_backward)
    for step in range(total_steps):
        torch.cuda.synchronize() # 保证计时从这开始
        t0 = timeit.default_timer()
        logits = model(inputs)
        loss = cross_entropy(logits, outputs)
        torch.cuda.synchronize()
        forward_time = timeit.default_timer() - t0
        if include_backward:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            backward_time = timeit.default_timer() - forward_time - t0
            logger.info(f"Forward time: {forward_time:.3f}s, Backward time: {backward_time:.3f}s")
        else:
            logger.info(f"Forward time: {forward_time:.3f}s")
        
        if step >= warmup_steps:
            forward_times.append(forward_time)
            if include_backward:
                backward_times.append(backward_time)
    
    forward_times = torch.tensor(forward_times)
    total_times = forward_times
    if include_backward:
        backward_times = torch.tensor(backward_times)
        total_times += backward_times
    results = {
        "model_name": model_name,
        "forward_times": forward_times,
        "total_forward_time": sum(forward_times).item(),
        "avg_forward_time": (sum(forward_times) / len(forward_times)).item(),
        "std_forward_time": np.std(forward_times).item(),
    }
    if include_backward:
        results["backward_time"] = backward_times
        results["total_time"] = total_times
        results["total_backward_time"] = sum(backward_times).item()
        results["avg_backward_time"] = (sum(backward_times) / len(backward_times)).item()
        results["avg_total_time"] = (sum(total_times) / len(total_times)).item()
        results["std_backward_time"] = np.std(backward_times).item()
        results["std_total_time"] = np.std(total_times).item()

    return results
    

def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    logger.add("benchmark_{time}.log", rotation="10 MB", retention="7 days", encoding="utf-8")

    analyzer = BenchmarkAnalyzer()

    device = torch.device("cuda:0")
    logger.info(f"Using device: {device}")

    curr_benchmark_num = 0
    for k, v in _PRESETS.items():
        if curr_benchmark_num >= args.num:
            break
        curr_benchmark_num += 1
        logger.info(f"Running benchmark for {k}")
        cfg = v.copy()
        cfg["model_name"] = k
        cfg["vocab_size"] = 10000
        cfg["context_length"] = 512
        cfg["rope_theta"] = 10000.0
        cfg["lr"] = 3e-4
        cfg["beta1"] = 0.9
        cfg["beta2"] = 0.999
        cfg["eps"] = 1e-8
        cfg["weight_decay"] = 0.01

        benchmark_cfg = {
            "warmup_steps": args.warm_steps,
            "total_steps": 10 + args.warm_steps,
            "include_backward": args.use_backward,
            "batch_size": 2,
        }

        # create model
        model = BasicsTransformerLM(
            vocab_size=cfg["vocab_size"],
            context_length=cfg["context_length"],
            d_model=cfg["d_model"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            d_ff=cfg["d_ff"],
            rope_theta=cfg["rope_theta"],
        )
        model.to(device)

        # create optimizer
        optimizer = AdamW(
            model.parameters(), 
            lr=cfg["lr"],
            betas=(cfg["beta1"], cfg["beta2"]),
            eps=cfg["eps"],
            weight_decay=cfg["weight_decay"],
        )

        # run benchmark
        results = run_benchmark(
            model=model,
            optimizer=optimizer,
            warmup_steps=benchmark_cfg["warmup_steps"],
            total_steps=benchmark_cfg["total_steps"],
            include_backward=benchmark_cfg["include_backward"],
            vocab_size=cfg["vocab_size"],
            batch_size=benchmark_cfg["batch_size"],
            seq_len=cfg["context_length"],
            model_name=cfg["model_name"],
            device=device,
        )

        logger.info(f"Results: \n{results}")
        analyzer.add_result(results)


    # 显示DataFrame
    print("Benchmark Results DataFrame:")
    print(analyzer.df)
    print("\n" + "="*50 + "\n")
    
    # 显示汇总统计
    print("Summary Statistics:")
    summary = analyzer.get_summary_table()
    print(summary)
    print("\n" + "="*50 + "\n")
    
    # 显示最佳性能
    print("Best Performance:")
    best = analyzer.get_best_performance()
    for key, value in best.items():
        print(f"{key}:")
        print(value)
        print()
        
        # 绘制图表
        analyzer.plot_performance_comparison()
    
    # 导出结果到markdown文件
    analyzer.export_to_markdown()



if __name__ == "__main__":
    main()