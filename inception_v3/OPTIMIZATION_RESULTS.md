# Inception-v3 在 BW150 上的优化结果

## 测试环境

- 设备：Hygon BW150（物理卡 2，容器内为 `cuda:0`）
- 容器：`zk-dla-0806`
- PyTorch：2.9.0
- HIP：6.3.26045
- 输入尺寸：`299 × 299`
- 预热 / 正式迭代：20 / 100
- 模型权重：随机初始化；不影响纯推理性能测试

`model.py` 和 `benchmark.py` 始终保持原始状态。所有优化均由独立的
`benchmark_optimized.py` 控制，便于逐项关闭和复现。

## 最终结果

延迟均为整个 batch 的平均推理延迟，不包含首次图编译时间。

| Batch | 精度 | 原始延迟 (ms) | 优化后延迟 (ms) | 延迟下降 | 原始吞吐 (img/s) | 优化后吞吐 (img/s) | 峰值显存 (MB) |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 16 | FP32 | 15.018 | 8.144 | 45.8% | 1065.4 | 1964.5 | 237.3 |
| 16 | FP16 | 12.356 | 6.193 | 49.9% | 1294.9 | 2583.7 | 122.0 |
| 128 | FP32 | 85.918 | 45.621 | 46.9% | 1489.8 | 2805.7 | 1237.6 |
| 128 | FP16 | 55.024 | 31.803 | 42.2% | 2326.3 | 4024.7 | 619.1 |

最终组合：

- FP32：`channels_last + TF32 + MIOpen Find mode 3 + torch.compile(reduce-overhead)`
- FP16：`channels_last + MIOpen Find mode 3 + torch.compile(reduce-overhead)`

## 分步测试结论

下表用于区分各类优化的独立贡献；数据均来自 BW150 实测。

| 测试项 | Batch / 精度 | 平均延迟 (ms) | 结论 |
|:---|:---:|---:|:---|
| 原始基线 | 16 / FP32 | 15.018 | 基准 |
| channels_last + MIOpen mode 3 | 16 / FP32 | 12.681 | 有效 |
| channels_last + TF32 + MIOpen mode 3 | 16 / FP32 | 11.902 | TF32 继续有效 |
| compile default（NCHW） | 16 / FP32 | 15.057 | 单独使用无收益 |
| compile reduce-overhead（NCHW） | 16 / FP32 | 14.719 | 单独收益很小 |
| compile max-autotune（NCHW） | 16 / FP32 | 14.578 | 首次编译成本高 |
| 最终组合 | 16 / FP32 | 8.144 | 优化可以叠加，收益明显 |
| 原始基线 | 128 / FP32 | 85.918 | 基准 |
| channels_last + MIOpen mode 3 | 128 / FP32 | 75.074 | 有效 |
| channels_last + TF32 + MIOpen mode 3 | 128 / FP32 | 59.530 | 大 batch 下 TF32 收益明显 |
| 最终组合 | 128 / FP32 | 45.621 | 最终采用 |
| 原始基线 | 128 / FP16 | 55.024 | 基准 |
| channels_last + MIOpen mode 3 | 128 / FP16 | 46.142 | 有效 |
| 最终组合 | 128 / FP16 | 31.803 | 最终采用 |

`max-autotune` 在组合测试时首次编译超过 3 分钟，因此没有作为通用默认项。
它在 NCHW 单项测试中相对 `reduce-overhead` 仅快约 1%，不足以抵消明显更高的编译成本。
rocBLAS 全局调优没有执行：Inception-v3 以卷积为主，而且该操作会修改共享配置，
不适合在公共容器中作为模型默认设置。

## 复现方式

推荐直接使用脚本内的最终预设：

```bash
cd /data1/zk/dla34-inceptionv3-mobilenetv2/inception_v3

HIP_VISIBLE_DEVICES=2 python3 benchmark_optimized.py \
  --best --device cuda:0 --dtype fp32 --batch-size 16

HIP_VISIBLE_DEVICES=2 python3 benchmark_optimized.py \
  --best --device cuda:0 --dtype fp16 --batch-size 128
```

也可以显式展开各优化项：

```bash
# FP32
HIP_VISIBLE_DEVICES=2 python3 benchmark_optimized.py \
  --device cuda:0 --dtype fp32 --batch-size 128 \
  --channels-last --tf32 --miopen-find-mode 3 \
  --compile-mode reduce-overhead

# FP16（TF32 不适用于 FP16）
HIP_VISIBLE_DEVICES=2 python3 benchmark_optimized.py \
  --device cuda:0 --dtype fp16 --batch-size 128 \
  --channels-last --miopen-find-mode 3 \
  --compile-mode reduce-overhead
```

首次运行 `torch.compile` 会生成编译缓存，耗时可能较长；以上表格统计的是预热完成后的稳态推理延迟。
