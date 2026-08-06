# DLA-34 在 BW150 上的第一轮优化结果

## 测试环境

- 容器：`zk-dla-0806`
- 设备：BW150，物理 2 号空闲卡
- PyTorch：2.9.0
- HIP：6.3.26045
- 输入：随机 Tensor，`3×224×224`
- 测试：warmup 20，正式迭代 100 次
- 原始实现：`model.py` + `benchmark.py`
- 优化实现：`model.py` + `benchmark_optimized.py`

原始 `model.py` 和 `benchmark.py` 未修改。测试时使用：

```bash
HIP_VISIBLE_DEVICES=2 python3 benchmark.py ...
```

## 原始基线

| Batch | 精度 | Batch 延迟 | 吞吐 |
|---:|---|---:|---:|
| 16 | FP32 | 8.010 ms | 1997.5 images/s |
| 16 | FP16 | 6.165 ms | 2595.3 images/s |
| 128 | FP32 | 49.997 ms | 2560.1 images/s |
| 128 | FP16 | 29.640 ms | 4318.5 images/s |

## 单项与逐步组合结果

NHWC 均先用 `MIOPEN_FIND_MODE=1` 建立当前 batch/dtype/layout 的缓存，再以 mode 3 正式测试。

| Batch | 精度 | 配置 | Batch 延迟 | 吞吐 |
|---:|---|---|---:|---:|
| 16 | FP32 | NHWC + MIOpen | 7.081 ms | 2259.5 images/s |
| 16 | FP32 | NHWC + TF32 + MIOpen | 5.800 ms | 2758.7 images/s |
| 16 | FP32 | 最优组合 | 4.713 ms | 3394.7 images/s |
| 16 | FP16 | NHWC + MIOpen | 5.727 ms | 2794.0 images/s |
| 16 | FP16 | 最优组合 | 3.180 ms | 5031.4 images/s |
| 128 | FP32 | NHWC + MIOpen | 40.064 ms | 3194.9 images/s |
| 128 | FP32 | NHWC + TF32 + MIOpen | 31.880 ms | 4015.0 images/s |
| 128 | FP32 | 最优组合 | 25.955 ms | 4931.6 images/s |
| 128 | FP16 | NHWC + MIOpen | 22.863 ms | 5598.5 images/s |
| 128 | FP16 | 最优组合 | 16.464 ms | 7774.3 images/s |

最优组合为：

- FP32：NHWC/channels_last + TF32 + MIOpen mode 3 + `torch.compile(mode="reduce-overhead", fullgraph=False)`
- FP16：NHWC/channels_last + MIOpen mode 3 + `torch.compile(mode="reduce-overhead", fullgraph=False)`

## 最终收益

| Batch | 精度 | 基线延迟 | 优化延迟 | 延迟降低 | 基线吞吐 | 优化吞吐 | 吞吐提升 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 16 | FP32 | 8.010 ms | 4.713 ms | 41.16% | 1997.5 | 3394.7 | 69.94% |
| 16 | FP16 | 6.165 ms | 3.180 ms | 48.42% | 2595.3 | 5031.4 | 93.86% |
| 128 | FP32 | 49.997 ms | 25.955 ms | 48.09% | 2560.1 | 4931.6 | 92.63% |
| 128 | FP16 | 29.640 ms | 16.464 ms | 44.45% | 4318.5 | 7774.3 | 80.02% |

## 复现命令

MIOpen 缓存已经建立后，直接使用 `--best`：

```bash
cd /data1/zk/dla34-inceptionv3-mobilenetv2/dla34

HIP_VISIBLE_DEVICES=2 python3 benchmark_optimized.py \
  --best --device cuda:0 --dtype fp32 --batch-size 16

HIP_VISIBLE_DEVICES=2 python3 benchmark_optimized.py \
  --best --device cuda:0 --dtype fp16 --batch-size 16

HIP_VISIBLE_DEVICES=2 python3 benchmark_optimized.py \
  --best --device cuda:0 --dtype fp32 --batch-size 128

HIP_VISIBLE_DEVICES=2 python3 benchmark_optimized.py \
  --best --device cuda:0 --dtype fp16 --batch-size 128
```

如果更换 batch、精度、layout、硬件或 DTK 版本，应先去掉 compile，以 `--channels-last --miopen-find-mode 1 --warmup 1 --iterations 1` 为新组合建立 MIOpen 缓存，再使用 mode 3 正式测试。

## 说明

- `torch.compile` 三种模式中，batch 16/FP32 单项测试分别为：default 7.081 ms、reduce-overhead 6.817 ms、max-autotune 6.860 ms；因此采用 reduce-overhead。
- `fullgraph=True` 为 4.710 ms，与 `False` 的 4.713 ms 基本一致，因此保留兼容性更好的 `False`。
- TF32 使用 `MIOPEN_PRECISION_FP32_FP32_FP32_TF32_FP32=1` 与当前 PyTorch 提示要求的 `HIPBLASLT_ALLOW_TF32=1`。
- 尚未执行 ImageNet 精度回归。TF32/FP16 上生产前需用真实预训练权重和验证集检查 Top-1/Top-5。
- 本轮没有执行 rocBLAS Layer4 自动选优，因为它会改写容器全局 rocBLAS 配置，而且 DLA-34 主要是卷积模型；如需继续冲刺，应先备份全局配置并单独评估。
