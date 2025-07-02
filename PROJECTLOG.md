# Profile 

| model_name   |   avg_forward_time |   total_forward_time |
|:-------------|-------------------:|---------------------:|
| sm           |          0.0227895 |             0.227895 |
| md           |          0.0454014 |             0.454014 |
| lg           |          0.0859365 |             0.859365 |

| model_name   |   avg_forward_time |   total_forward_time |   avg_backward_time |   total_backward_time |   avg_total_time |
|:-------------|-------------------:|---------------------:|--------------------:|----------------------:|-----------------:|
| sm           |          0.0645101 |             0.645101 |           0.0420522 |              0.420522 |        0.0645101 |
| md           |          0.141717  |             1.41717  |           0.0964797 |              0.964797 |        0.141717  |

Due to computation resource limitation(CUDA out of memory), we only run the first 10 steps of the benchmark.

# benchmark_attn
```bash
d_model seq_len fwd(ms) bwd(ms) mem_before(MB)
16      256     0.12    0.90    20
16      1024    0.31    1.46    53
16      4096    7.20    24.24   563
16      8192    28.61   94.97   2182
16      16384   OOM     OOM     OOM
32      256     0.11    1.06    20
32      1024    0.44    1.46    55
32      4096    7.30    24.44   571
32      8192    28.79   95.34   2199
32      16384   OOM     OOM     OOM
64      256     0.11    0.93    21
64      1024    0.33    1.43    59
64      4096    7.36    24.52   588
64      8192    29.10   95.94   2232
64      16384   OOM     OOM     OOM
128     256     0.14    1.02    23
128     1024    0.37    1.55    67
128     4096    7.81    25.30   621
128     8192    31.01   99.37   2300
128     16384   OOM     OOM     OOM
```

# 