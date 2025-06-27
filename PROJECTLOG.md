## 1.1 Profile 

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

## 1.2 FlashAttention

