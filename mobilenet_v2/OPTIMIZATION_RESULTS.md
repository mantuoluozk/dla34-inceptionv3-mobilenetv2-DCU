# MobileNet-v2 在 BW150 上的优化结果

## 测试环境

- 设备：Hygon BW150（物理卡 2，容器内为 `cuda:0`）
- 容器：`zk-dla-0806`
- PyTorch：2.9.0
- HIP：6.3.26045
- 输入尺寸：`224 × 224`
- 预热 / 正式迭代：20 / 100
- 模型权重：随机初始化；不影响纯推理性能测试

原始 `model.py` 和 `benchmark.py` 未修改。所有优化均放在独立的
`benchmark_optimized.py` 中，可以逐项启用或关闭。

## 最终结果

延迟为整个 batch 的平均稳态推理延迟，不包含 MIOpen 搜索和首次图编译时间。

| Batch | 精度 | 原始延迟 (ms) | 优化后延迟 (ms) | 延迟下降 | 原始吞吐 (img/s) | 优化后吞吐 (img/s) | 优化后峰值显存 (MB) |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 16 | FP32 | 6.300 | 2.686 | 57.4% | 2539.6 | 5955.7 | 115.6 |
| 16 | FP16 | 6.586 | 2.408 | 63.4% | 2429.4 | 6644.4 | 57.4 |
| 128 | FP32 | 22.504 | 15.443 | 31.4% | 5687.8 | 8288.6 | 822.6 |
| 128 | FP16 | 26.276 | 11.976 | 54.4% | 4871.3 | 10687.8 | 411.1 |

最终组合：

- FP32：`channels_last + TF32 + MIOpen Find mode 3 + torch.compile(reduce-overhead)`
- FP16：`channels_last + MIOpen Find mode 3 + torch.compile(reduce-overhead)`

## 分步测试结论

| 测试项 | Batch / 精度 | 平均延迟 (ms) | 结论 |
|:---|:---:|---:|:---|
| 原始基线 | 16 / FP32 | 6.300 | 基准 |
| channels_last + MIOpen mode 3 | 16 / FP32 | 6.355 | 小 batch 单独使用无收益 |
| channels_last + TF32 + MIOpen mode 3 | 16 / FP32 | 6.340 | TF32 单独收益不明显 |
| compile reduce-overhead（NCHW） | 16 / FP32 | 2.950 | 图融合收益明显 |
| 最终组合 | 16 / FP32 | 2.686 | 最终采用 |
| compile default（最终布局） | 16 / FP32 | 3.328 | 慢于 reduce-overhead |
| compile max-autotune（最终布局） | 16 / FP32 | 2.744 | 慢于 reduce-overhead，编译更久 |
| 原始基线 | 16 / FP16 | 6.586 | 基准 |
| channels_last + MIOpen mode 3 | 16 / FP16 | 6.436 | 单独收益很小 |
| compile reduce-overhead（NCHW） | 16 / FP16 | 1634.917 | 严重后端回退，禁止使用 |
| 最终组合（NHWC + compile） | 16 / FP16 | 2.408 | NHWC 避开回退并获得融合收益 |
| 原始基线 | 128 / FP32 | 22.504 | 基准 |
| channels_last + MIOpen mode 3 | 128 / FP32 | 20.019 | 有效 |
| channels_last + TF32 + MIOpen mode 3 | 128 / FP32 | 19.649 | TF32 有小幅增益 |
| 最终组合 | 128 / FP32 | 15.443 | 最终采用 |
| 原始基线 | 128 / FP16 | 26.276 | 基准，原生 FP16 反而慢于 FP32 |
| channels_last + MIOpen mode 3 | 128 / FP16 | 15.339 | depthwise convolution 对布局敏感 |
| 最终组合 | 128 / FP16 | 11.976 | 最终采用 |

## 关键观察

MobileNet-v2 包含大量 depthwise convolution 和小算子。原始 NCHW 下，FP16 在两个
batch size 上都慢于 FP32；`bs=128` 使用 NHWC 后，FP16 延迟从 26.276 ms 降到
15.339 ms，说明先前的 FP16 反常主要与算子布局/后端路径有关，而不是 FP16 硬件本身更慢。

另一个重要限制是：当前 PyTorch 2.9.0 + HIP 6.3 环境中，FP16 NCHW 与
`torch.compile` 组合会严重回退到约 1.63 秒/batch；转换为 NHWC 后恢复到 2.408 ms。
因此 FP16 图编译必须与 `channels_last` 一起使用，不能单独打开。

`max-autotune` 首次编译约两分钟，稳态速度仍慢于 `reduce-overhead`，所以没有采用。
rocBLAS 全局调优没有执行：本模型以卷积为主，而且该操作会修改共享配置，不适合作为
公共容器中的模型默认设置。

## 复现方式

推荐直接使用脚本内的最终预设：

```bash
cd /data1/zk/dla34-inceptionv3-mobilenetv2/mobilenet_v2

HIP_VISIBLE_DEVICES=2 python3 benchmark_optimized.py \
  --best --device cuda:0 --dtype fp32 --batch-size 16

HIP_VISIBLE_DEVICES=2 python3 benchmark_optimized.py \
  --best --device cuda:0 --dtype fp16 --batch-size 128
```

显式展开方式：

```bash
# FP32
HIP_VISIBLE_DEVICES=2 python3 benchmark_optimized.py \
  --device cuda:0 --dtype fp32 --batch-size 128 \
  --channels-last --tf32 --miopen-find-mode 3 \
  --compile-mode reduce-overhead

# FP16：不要移除 --channels-last
HIP_VISIBLE_DEVICES=2 python3 benchmark_optimized.py \
  --device cuda:0 --dtype fp16 --batch-size 128 \
  --channels-last --miopen-find-mode 3 \
  --compile-mode reduce-overhead
```

首次执行 `torch.compile` 会生成缓存，耗时明显高于后续运行；最终表格只统计预热后的稳态推理。
