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

# flash attention benchmark

|    |   seq_len |   d_model | precision   |   ft_torch | bt_torch   | fbt_torch   |   ft_triton |   bt_triton |   fbt_triton | speedup_fwd   | speedup_bwd   | speedup_tot   |
|---:|----------:|----------:|:------------|-----------:|:-----------|:------------|------------:|------------:|-------------:|:--------------|:--------------|:--------------|
|  0 |       128 |        16 | bf16        |      0.033 | 0.435      | 0.707       |       0.005 |       0.311 |        0.477 | 6.0x          | 1.4x          | 1.5x          |
|  1 |       128 |        16 | f32         |      0.034 | 0.207      | 0.919       |       0.006 |       0.215 |        0.394 | 5.5x          | 1.0x          | 2.3x          |
|  2 |       128 |        32 | bf16        |      0.03  | 0.266      | 0.893       |       0.006 |       0.29  |        0.801 | 5.2x          | 0.9x          | 1.1x          |
|  3 |       128 |        32 | f32         |      0.061 | 0.401      | 0.786       |       0.008 |       0.176 |        0.55  | 8.1x          | 2.3x          | 1.4x          |
|  4 |       128 |        64 | bf16        |      0.031 | 0.224      | 0.669       |       0.043 |       0.256 |        0.524 | 0.7x          | 0.9x          | 1.3x          |
|  5 |       128 |        64 | f32         |      0.142 | 0.391      | 0.861       |       0.01  |       0.165 |        0.721 | 13.8x         | 2.4x          | 1.2x          |
|  6 |       256 |        16 | bf16        |      0.031 | 0.203      | 0.622       |       0.006 |       0.277 |        0.42  | 5.0x          | 0.7x          | 1.5x          |
|  7 |       256 |        16 | f32         |      0.034 | 0.310      | 0.667       |       0.007 |       0.188 |        0.292 | 4.6x          | 1.7x          | 2.3x          |
|  8 |       256 |        32 | bf16        |      0.033 | 0.250      | 0.613       |       0.007 |       0.18  |        0.63  | 4.7x          | 1.4x          | 1.0x          |
|  9 |       256 |        32 | f32         |      0.097 | 0.344      | 0.963       |       0.01  |       0.137 |        0.541 | 10.1x         | 2.5x          | 1.8x          |
| 10 |       256 |        64 | bf16        |      0.028 | 0.297      | 1.056       |       0.009 |       0.353 |        0.443 | 3.2x          | 0.8x          | 2.4x          |
| 11 |       256 |        64 | f32         |      0.165 | 0.538      | 1.112       |       0.015 |       0.178 |        0.664 | 10.8x         | 3.0x          | 1.7x          |
| 12 |       512 |        16 | bf16        |      0.14  | 0.675      | 0.983       |       0.008 |       0.269 |        0.631 | 18.1x         | 2.5x          | 1.6x          |
| 13 |       512 |        16 | f32         |      0.051 | 0.357      | 0.707       |       0.01  |       0.225 |        0.387 | 5.1x          | 1.6x          | 1.8x          |
| 14 |       512 |        32 | bf16        |      0.032 | 0.216      | 0.763       |       0.009 |       0.19  |        0.434 | 3.4x          | 1.1x          | 1.8x          |
| 15 |       512 |        32 | f32         |      0.036 | 0.211      | 0.778       |       0.014 |       0.182 |        0.397 | 2.6x          | 1.2x          | 2.0x          |
| 16 |       512 |        64 | bf16        |      0.033 | 0.281      | 0.839       |       0.013 |       0.267 |        0.602 | 2.6x          | 1.1x          | 1.4x          |
| 17 |       512 |        64 | f32         |      0.037 | 0.342      | 0.835       |       0.024 |       0.49  |        0.611 | 1.5x          | 0.7x          | 1.4x          |
| 18 |      1024 |        16 | bf16        |      0.047 | 0.338      | 0.690       |       0.012 |       0.258 |        0.527 | 4.0x          | 1.3x          | 1.3x          |
| 19 |      1024 |        16 | f32         |      0.109 | 0.412      | 1.045       |       0.016 |       0.333 |        0.621 | 6.7x          | 1.2x          | 1.7x          |
| 20 |      1024 |        32 | bf16        |      0.047 | 0.217      | 0.931       |       0.014 |       0.211 |        0.66  | 3.4x          | 1.0x          | 1.4x          |
| 21 |      1024 |        32 | f32         |      0.058 | 0.231      | 0.872       |       0.024 |       0.197 |        0.563 | 2.5x          | 1.2x          | 1.5x          |
| 22 |      1024 |        64 | bf16        |      0.049 | 0.251      | 0.924       |       0.021 |       0.355 |        0.48  | 2.3x          | 0.7x          | 1.9x          |
| 23 |      1024 |        64 | f32         |      0.119 | 0.282      | 0.613       |       0.043 |       0.211 |        0.461 | 2.7x          | 1.3x          | 1.3x          |
| 24 |      2048 |        16 | bf16        |      0.122 | 0.231      | 0.711       |       0.018 |       0.341 |        0.641 | 6.8x          | 0.7x          | 1.1x          |
| 25 |      2048 |        16 | f32         |      0.15  | 0.325      | 0.680       |       0.029 |       0.188 |        0.757 | 5.2x          | 1.7x          | 0.9x          |
| 26 |      2048 |        32 | bf16        |      0.099 | 0.283      | 0.739       |       0.023 |       0.385 |        0.52  | 4.2x          | 0.7x          | 1.4x          |
| 27 |      2048 |        32 | f32         |      0.157 | 0.331      | 0.734       |       0.042 |       0.192 |        0.56  | 3.7x          | 1.7x          | 1.3x          |
| 28 |      2048 |        64 | bf16        |      0.1   | 0.482      | 1.082       |       0.039 |       0.308 |        0.706 | 2.6x          | 1.6x          | 1.5x          |
| 29 |      2048 |        64 | f32         |      0.204 | 0.355      | 0.945       |       0.083 |       0.204 |        0.447 | 2.5x          | 1.7x          | 2.1x          |
| 30 |      4096 |        16 | bf16        |      0.343 | 0.870      | 1.178       |       0.033 |       0.62  |        0.93  | 10.5x         | 1.4x          | 1.3x          |
| 31 |      4096 |        16 | f32         |      0.854 | 2.205      | 3.002       |       0.054 |       0.67  |        0.745 | 15.9x         | 3.3x          | 4.0x          |
| 32 |      4096 |        32 | bf16        |      0.344 | 0.845      | 1.388       |       0.043 |       0.609 |        0.651 | 8.1x          | 1.4x          | 2.1x          |
| 33 |      4096 |        32 | f32         |      0.856 | 2.212      | 3.012       |       0.081 |       0.736 |        0.969 | 10.6x         | 3.0x          | 3.1x          |
| 34 |      4096 |        64 | bf16        |      0.347 | 0.846      | 1.159       |       0.078 |       0.687 |        0.969 | 4.5x          | 1.2x          | 1.2x          |
| 35 |      4096 |        64 | f32         |      0.915 | 2.300      | 3.169       |       0.167 |       0.775 |        0.929 | 5.5x          | 3.0x          | 3.4x          |
| 36 |      8192 |        16 | bf16        |      2.186 | 4.822      | 6.927       |       0.063 |       2.46  |        2.526 | 34.7x         | 2.0x          | 2.7x          |
| 37 |      8192 |        16 | f32         |      4.09  | 9.327      | 13.355      |       0.105 |       2.759 |        2.858 | 39.1x         | 3.4x          | 4.7x          |
| 38 |      8192 |        32 | bf16        |      2.185 | 4.826      | 6.921       |       0.083 |       2.46  |        2.534 | 26.4x         | 2.0x          | 2.7x          |
| 39 |      8192 |        32 | f32         |      4.079 | 9.331      | 13.343      |       0.158 |       2.752 |        2.897 | 25.8x         | 3.4x          | 4.6x          |
| 40 |      8192 |        64 | bf16        |      2.19  | 4.834      | 6.945       |       0.144 |       2.527 |        2.661 | 15.2x         | 1.9x          | 2.6x          |
| 41 |      8192 |        64 | f32         |      4.14  | 9.430      | 13.499      |       0.317 |       2.823 |        3.129 | 13.1x         | 3.3x          | 4.3x          |
| 42 |     16384 |        16 | bf16        |      8.574 | 19.053     | 27.574      |       0.198 |       9.688 |        9.885 | 43.2x         | 2.0x          | 2.8x          |
| 43 |     16384 |        16 | f32         |     16.271 | 37.072     | 53.286      |       0.347 |      10.845 |       11.185 | 46.9x         | 3.4x          | 4.8x          |
| 44 |     16384 |        32 | bf16        |      8.508 | 19.050     | 27.512      |       0.291 |       9.693 |        9.973 | 29.2x         | 2.0x          | 2.8x          |
| 45 |     16384 |        32 | f32         |     16.31  | 37.164     | 53.423      |       0.613 |      10.922 |       11.514 | 26.6x         | 3.4x          | 4.6x          |
| 46 |     16384 |        64 | bf16        |      8.532 | 19.072     | 27.554      |       0.51  |       9.933 |       10.398 | 16.7x         | 1.9x          | 2.6x          |
| 47 |     16384 |        64 | f32         |     16.412 | 37.389     | 53.749      |       1.236 |      11.235 |       12.473 | 13.3x         | 3.3x          | 4.3x          |
| 48 |     32768 |        16 | bf16        |     33.81  | 75.721     | 109.476     |       0.666 |      38.015 |       38.713 | 50.7x         | 2.0x          | 2.8x          |
| 49 |     32768 |        16 | f32         |     64.637 | OOM        | OOM         |       1.347 |      42.887 |       44.257 | 48.0x         | N/A           | N/A           |
| 50 |     32768 |        32 | bf16        |     33.831 | 75.870     | 109.643     |       1.146 |      38.485 |       39.607 | 29.5x         | 2.0x          | 2.8x          |
| 51 |     32768 |        32 | f32         |     65.034 | OOM        | OOM         |       2.436 |      43.682 |       46.077 | 26.7x         | N/A           | N/A           |
| 52 |     32768 |        64 | bf16        |     33.957 | 76.090     | 109.990     |       1.991 |      38.795 |       40.726 | 17.1x         | 2.0x          | 2.7x          |
| 53 |     32768 |        64 | f32         |     65.043 | OOM        | OOM         |       4.933 |      43.865 |       48.948 | 13.2x         | N/A           | N/A           |

# Benchmark on all reduce

GLOO | cpu  | 2 proc |    1 MB -> min   0.68 mean    0.88 max   1.19 MB/s
NCCL | cuda | 2 proc |    1 MB -> min   0.17 mean    0.18 max   0.23 MB/s
GLOO | cpu  | 4 proc |    1 MB -> min   1.45 mean    1.72 max   2.22 MB/s
[WARNING] NCCL with world_size=4 requires >=4 GPUs, but only 2 GPUs are available.
GLOO | cpu  | 6 proc |    1 MB -> min   1.89 mean    2.17 max   2.57 MB/s
[WARNING] NCCL with world_size=6 requires >=6 GPUs, but only 2 GPUs are available.
GLOO | cpu  | 2 proc |   10 MB -> min   4.65 mean    5.53 max   8.16 MB/s
NCCL | cuda | 2 proc |   10 MB -> min   1.34 mean    1.36 max   1.40 MB/s
GLOO | cpu  | 4 proc |   10 MB -> min   6.63 mean    7.42 max   8.65 MB/s
[WARNING] NCCL with world_size=4 requires >=4 GPUs, but only 2 GPUs are available.
GLOO | cpu  | 6 proc |   10 MB -> min   7.23 mean    8.29 max   9.00 MB/s
[WARNING] NCCL with world_size=6 requires >=6 GPUs, but only 2 GPUs are available.
GLOO | cpu  | 2 proc |  100 MB -> min  48.28 mean   54.38 max  64.34 MB/s
NCCL | cuda | 2 proc |  100 MB -> min  12.95 mean   13.07 max  13.15 MB/s
GLOO | cpu  | 4 proc |  100 MB -> min  65.56 mean   75.38 max  97.51 MB/s
[WARNING] NCCL with world_size=4 requires >=4 GPUs, but only 2 GPUs are available.
GLOO | cpu  | 6 proc |  100 MB -> min 106.05 mean  117.75 max 126.51 MB/s
[WARNING] NCCL with world_size=6 requires >=6 GPUs, but only 2 GPUs are available.
GLOO | cpu  | 2 proc | 1024 MB -> min 694.55 mean 1060.76 max 1268.18 MB/s
NCCL | cuda | 2 proc | 1024 MB -> min 130.23 mean  131.34 max 132.71 MB/s
GLOO | cpu  | 4 proc | 1024 MB -> min 1111.45 mean 1311.16 max 1643.89 MB/s
[WARNING] NCCL with world_size=4 requires >=4 GPUs, but only 2 GPUs are available.
GLOO | cpu  | 6 proc | 1024 MB -> min 1424.33 mean 1759.10 max 2168.13 MB/s
[WARNING] NCCL with world_size=6 requires >=6 GPUs, but only 2 GPUs are available.

# DDP Benchmark

root@autodl-container-7e674e82f3-4ace97e9:~/assignment2-systems# uv run -m -- cs336_systems.ddp_bench
Benchmarking on NCCL
--------------------------------------------------------
Enter the size of the model (default 10): 50
NaiveDDPIndividualParameters   | bkw pass and communication time (ms) |min 0.95 | mean 1.01 | max 1.16
FlatDDPParameters              | bkw pass and communication time (ms) |min 0.94 | mean 0.97 | max 1.03
DDPIndividualParameters        | bkw pass and communication time (ms) |min 0.94 | mean 0.98 | max 1.02
DDPBucketParameters(1 MB)      | bkw pass and communication time (ms) |min 0.85 | mean 1.12 | max 4.51
--------------------------------------------------------
NaiveDDPIndividualParameters   | bkw pass and communication time (ms) |min 0.95 | mean 1.25 | max 3.30
FlatDDPParameters              | bkw pass and communication time (ms) |min 0.88 | mean 1.09 | max 2.25
DDPIndividualParameters        | bkw pass and communication time (ms) |min 0.92 | mean 1.36 | max 4.95
DDPBucketParameters(10 MB)     | bkw pass and communication time (ms) |min 0.85 | mean 1.30 | max 3.91
--------------------------------------------------------
NaiveDDPIndividualParameters   | bkw pass and communication time (ms) |min 1.01 | mean 1.64 | max 5.11
FlatDDPParameters              | bkw pass and communication time (ms) |min 0.92 | mean 1.35 | max 5.92
DDPIndividualParameters        | bkw pass and communication time (ms) |min 0.96 | mean 1.14 | max 1.35
DDPBucketParameters(100 MB)    | bkw pass and communication time (ms) |min 0.87 | mean 0.96 | max 1.20
--------------------------------------------------------
NaiveDDPIndividualParameters   | bkw pass and communication time (ms) |min 0.86 | mean 0.90 | max 0.98
FlatDDPParameters              | bkw pass and communication time (ms) |min 0.84 | mean 0.88 | max 0.95
DDPIndividualParameters        | bkw pass and communication time (ms) |min 0.90 | mean 1.06 | max 2.94
DDPBucketParameters(1000 MB)   | bkw pass and communication time (ms) |min 0.79 | mean 0.82 | max 0.91
--------------------------------------------------------
root@autodl-container-7e674e82f3-4ace97e9:~/assignment2-systems# uv run -m -- cs336_systems.ddp_bench
Benchmarking on NCCL
--------------------------------------------------------
Enter the size of the model (default 10): 1000
NaiveDDPIndividualParameters   | bkw pass and communication time (ms) |min 2.34 | mean 2.52 | max 2.75
FlatDDPParameters              | bkw pass and communication time (ms) |min 2.28 | mean 2.75 | max 7.28
DDPIndividualParameters        | bkw pass and communication time (ms) |min 2.10 | mean 2.30 | max 3.30
DDPBucketParameters(1 MB)      | bkw pass and communication time (ms) |min 2.13 | mean 2.28 | max 2.46
--------------------------------------------------------
NaiveDDPIndividualParameters   | bkw pass and communication time (ms) |min 2.39 | mean 2.50 | max 2.66
FlatDDPParameters              | bkw pass and communication time (ms) |min 2.25 | mean 2.74 | max 12.19
DDPIndividualParameters        | bkw pass and communication time (ms) |min 2.12 | mean 2.34 | max 3.20
DDPBucketParameters(10 MB)     | bkw pass and communication time (ms) |min 2.12 | mean 2.47 | max 6.75
--------------------------------------------------------
NaiveDDPIndividualParameters   | bkw pass and communication time (ms) |min 2.37 | mean 2.69 | max 3.75
FlatDDPParameters              | bkw pass and communication time (ms) |min 2.25 | mean 2.35 | max 2.71
DDPIndividualParameters        | bkw pass and communication time (ms) |min 2.13 | mean 2.22 | max 3.18
DDPBucketParameters(100 MB)    | bkw pass and communication time (ms) |min 2.21 | mean 2.41 | max 4.22
--------------------------------------------------------
NaiveDDPIndividualParameters   | bkw pass and communication time (ms) |min 2.38 | mean 2.62 | max 5.29
FlatDDPParameters              | bkw pass and communication time (ms) |min 2.27 | mean 2.31 | max 2.40
DDPIndividualParameters        | bkw pass and communication time (ms) |min 2.12 | mean 2.15 | max 2.23
DDPBucketParameters(1000 MB)   | bkw pass and communication time (ms) |min 2.15 | mean 2.21 | max 2.40
--------------------------------------------------------
root@autodl-container-7e674e82f3-4ace97e9:~/assignment2-systems# uv run -m -- cs336_systems.ddp_bench
Benchmarking on NCCL
--------------------------------------------------------
Enter the size of the model (default 10): 5000
NaiveDDPIndividualParameters   | bkw pass and communication time (ms) |min 39.99 | mean 41.44 | max 42.71
FlatDDPParameters              | bkw pass and communication time (ms) |min 41.81 | mean 42.97 | max 46.46
DDPIndividualParameters        | bkw pass and communication time (ms) |min 40.02 | mean 42.17 | max 46.20
DDPBucketParameters(1 MB)      | bkw pass and communication time (ms) |min 39.55 | mean 41.36 | max 42.14
--------------------------------------------------------
NaiveDDPIndividualParameters   | bkw pass and communication time (ms) |min 39.56 | mean 40.53 | max 42.61
FlatDDPParameters              | bkw pass and communication time (ms) |min 41.37 | mean 42.51 | max 44.41
DDPIndividualParameters        | bkw pass and communication time (ms) |min 39.32 | mean 40.83 | max 42.15
DDPBucketParameters(10 MB)     | bkw pass and communication time (ms) |min 39.83 | mean 41.21 | max 43.05
--------------------------------------------------------
NaiveDDPIndividualParameters   | bkw pass and communication time (ms) |min 40.15 | mean 41.18 | max 42.69
FlatDDPParameters              | bkw pass and communication time (ms) |min 40.89 | mean 42.27 | max 45.44
DDPIndividualParameters        | bkw pass and communication time (ms) |min 39.77 | mean 41.16 | max 42.15
DDPBucketParameters(100 MB)    | bkw pass and communication time (ms) |min 39.43 | mean 40.77 | max 43.23
--------------------------------------------------------
NaiveDDPIndividualParameters   | bkw pass and communication time (ms) |min 39.38 | mean 40.36 | max 43.63
FlatDDPParameters              | bkw pass and communication time (ms) |min 41.22 | mean 43.07 | max 44.40
DDPIndividualParameters        | bkw pass and communication time (ms) |min 39.87 | mean 41.35 | max 42.31
DDPBucketParameters(1000 MB)   | bkw pass and communication time (ms) |min 41.05 | mean 42.51 | max 43.61
--------------------------------------------------------