# DLA-34、Inception-v3、MobileNet-v2 在海光 BW150 上的推理优化

本仓库提供三个图像分类模型的原始 PyTorch 性能基线、独立优化脚本，以及在海光 BW150 上的完整测试结果。优化脚本不会修改原始模型和基准代码，便于逐项验证优化效果。

## 项目结构

```text
.
├── dla34/
│   ├── model.py
│   ├── benchmark.py
│   ├── benchmark_optimized.py
│   ├── OPTIMIZATION_RESULTS.md
│   └── LICENSE
├── inception_v3/
│   ├── model.py
│   ├── benchmark.py
│   ├── benchmark_optimized.py
│   └── OPTIMIZATION_RESULTS.md
├── mobilenet_v2/
│   ├── model.py
│   ├── benchmark.py
│   ├── benchmark_optimized.py
│   └── OPTIMIZATION_RESULTS.md
├── docker_create.sh
└── README.md
```

## 模型来源

- `dla34`：根据 UC Berkeley 官方 [Deep Layer Aggregation](https://github.com/ucbdrive/dla) PyTorch 实现整理。
- `inception_v3`：调用 torchvision 官方 `inception_v3` 实现。
- `mobilenet_v2`：调用 torchvision 官方 `mobilenet_v2` 实现。

## 测试环境

| 项目 | 配置 |
|:---|:---|
| 加速卡 | Hygon BW150 |
| PyTorch | 2.9.0 |
| HIP | 6.3.26045 |
| Python | 3.10.12 |
| Docker 镜像 | `harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.9.0-ubuntu22.04-dtk26.04-py3.10` |
| 正式测试 | warmup 20，iterations 100 |

基础依赖：

```text
torch
torchvision
```

可使用仓库中的脚本创建或进入测试容器：

```bash
docker pull harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.9.0-ubuntu22.04-dtk26.04-py3.10
chmod +x docker_create.sh
./docker_create.sh
```

## 开箱运行：原始基线

原始脚本使用随机输入进行纯推理测速，无需下载数据集或预训练权重。设备在 PyTorch 中使用 `cuda:0` 兼容接口；如需指定物理卡，可设置 `HIP_VISIBLE_DEVICES`。

### DLA-34

```bash
cd dla34

HIP_VISIBLE_DEVICES=2 python3 benchmark.py \
  --device cuda:0 --dtype fp32 --batch-size 16

HIP_VISIBLE_DEVICES=2 python3 benchmark.py \
  --device cuda:0 --dtype fp16 --batch-size 128
```

### Inception-v3

```bash
cd inception_v3

HIP_VISIBLE_DEVICES=2 python3 benchmark.py \
  --device cuda:0 --dtype fp32 --batch-size 16

HIP_VISIBLE_DEVICES=2 python3 benchmark.py \
  --device cuda:0 --dtype fp16 --batch-size 128
```

### MobileNet-v2

```bash
cd mobilenet_v2

HIP_VISIBLE_DEVICES=2 python3 benchmark.py \
  --device cuda:0 --dtype fp32 --batch-size 16

HIP_VISIBLE_DEVICES=2 python3 benchmark.py \
  --device cuda:0 --dtype fp16 --batch-size 128
```

如需一次测试当前目录的四种目标组合：

```bash
for batch_size in 16 128; do
  for dtype in fp32 fp16; do
    HIP_VISIBLE_DEVICES=2 python3 benchmark.py \
      --device cuda:0 --dtype "$dtype" --batch-size "$batch_size"
  done
done
```

## 优化运行

三个 `benchmark_optimized.py` 使用相同的参数接口，并将当前 BW150 实测最优组合封装为 `--best`：

- FP32：NHWC/channels_last + TF32 + MIOpen Find mode 3 + `torch.compile(mode="reduce-overhead")`
- FP16：NHWC/channels_last + MIOpen Find mode 3 + `torch.compile(mode="reduce-overhead")`

### 使用最优预设

在对应模型目录执行：

```bash
# FP32，batch 16
HIP_VISIBLE_DEVICES=2 python3 benchmark_optimized.py \
  --best --device cuda:0 --dtype fp32 --batch-size 16

# FP16，batch 16
HIP_VISIBLE_DEVICES=2 python3 benchmark_optimized.py \
  --best --device cuda:0 --dtype fp16 --batch-size 16

# FP32，batch 128
HIP_VISIBLE_DEVICES=2 python3 benchmark_optimized.py \
  --best --device cuda:0 --dtype fp32 --batch-size 128

# FP16，batch 128
HIP_VISIBLE_DEVICES=2 python3 benchmark_optimized.py \
  --best --device cuda:0 --dtype fp16 --batch-size 128
```

### 手动组合优化项

```bash
# FP32
HIP_VISIBLE_DEVICES=2 python3 benchmark_optimized.py \
  --device cuda:0 --dtype fp32 --batch-size 128 \
  --channels-last --tf32 --miopen-find-mode 3 \
  --compile-mode reduce-overhead

# FP16
HIP_VISIBLE_DEVICES=2 python3 benchmark_optimized.py \
  --device cuda:0 --dtype fp16 --batch-size 128 \
  --channels-last --miopen-find-mode 3 \
  --compile-mode reduce-overhead
```

可独立控制的优化参数：

```text
--channels-last
--tf32                         # 仅适用于 FP32
--miopen-find-mode 1|3
--compile-mode none|default|reduce-overhead|max-autotune
--fullgraph
```

首次运行新 batch、精度或 layout 时，可以先使用 MIOpen mode 1 建立算法缓存：

```bash
HIP_VISIBLE_DEVICES=2 python3 benchmark_optimized.py \
  --device cuda:0 --dtype fp16 --batch-size 128 \
  --channels-last --miopen-find-mode 1 \
  --compile-mode none --warmup 1 --iterations 1
```

随后切换到 mode 3 进行正式测试。首次执行 `torch.compile` 也会生成编译缓存，其编译时间不计入表中的稳态推理延迟。

## BW150 性能结果

| 模型 | Batch | 精度 | 基线延迟 | 优化延迟 | 延迟降低 | 基线吞吐 | 优化吞吐 | 吞吐提升 |
|:---|---:|:---:|---:|---:|---:|---:|---:|---:|
| DLA-34 | 16 | FP32 | 8.010 ms | 4.713 ms | 41.16% | 1997.5 img/s | 3394.7 img/s | 69.94% |
| DLA-34 | 16 | FP16 | 6.165 ms | 3.180 ms | 48.42% | 2595.3 img/s | 5031.4 img/s | 93.86% |
| DLA-34 | 128 | FP32 | 49.997 ms | 25.955 ms | 48.09% | 2560.1 img/s | 4931.6 img/s | 92.63% |
| DLA-34 | 128 | FP16 | 29.640 ms | 16.464 ms | 44.45% | 4318.5 img/s | 7774.3 img/s | 80.02% |
| Inception-v3 | 16 | FP32 | 15.018 ms | 8.144 ms | 45.77% | 1065.4 img/s | 1964.5 img/s | 84.40% |
| Inception-v3 | 16 | FP16 | 12.356 ms | 6.193 ms | 49.88% | 1294.9 img/s | 2583.7 img/s | 99.53% |
| Inception-v3 | 128 | FP32 | 85.918 ms | 45.621 ms | 46.90% | 1489.8 img/s | 2805.7 img/s | 88.33% |
| Inception-v3 | 128 | FP16 | 55.024 ms | 31.803 ms | 42.20% | 2326.3 img/s | 4024.7 img/s | 73.01% |
| MobileNet-v2 | 16 | FP32 | 6.300 ms | 2.686 ms | 57.36% | 2539.6 img/s | 5955.7 img/s | 134.52% |
| MobileNet-v2 | 16 | FP16 | 6.586 ms | 2.408 ms | 63.44% | 2429.4 img/s | 6644.4 img/s | 173.50% |
| MobileNet-v2 | 128 | FP32 | 22.504 ms | 15.443 ms | 31.38% | 5687.8 img/s | 8288.6 img/s | 45.73% |
| MobileNet-v2 | 128 | FP16 | 26.276 ms | 11.976 ms | 54.42% | 4871.3 img/s | 10687.8 img/s | 119.40% |

详细的逐项测试数据、编译模式对比和分析见：

- [DLA-34 优化结果](dla34/OPTIMIZATION_RESULTS.md)
- [Inception-v3 优化结果](inception_v3/OPTIMIZATION_RESULTS.md)
- [MobileNet-v2 优化结果](mobilenet_v2/OPTIMIZATION_RESULTS.md)

## 输出指标

每次运行以 JSON 格式输出：

- 模型、设备、Python、PyTorch 和 HIP 信息
- 输入、输出形状与张量布局
- 参数量和模型权重大小
- 平均、最小、P50、P90、P99、最大延迟
- 单样本延迟与吞吐率
- 峰值设备显存
- 当前启用的优化项

可以通过 `--output result.json` 同时将结果保存为文件。如需加载官方 ImageNet 预训练权重，可以添加 `--pretrained`；测速默认使用随机初始化权重。

## 注意事项

- 三个模型的输入尺寸分别为 DLA-34 `224×224`、Inception-v3 `299×299`、MobileNet-v2 `224×224`。
- MobileNet-v2 的 FP16 图编译必须与 `channels_last` 同时使用；当前环境中 FP16 NCHW + `torch.compile` 会出现严重性能回退。
- TF32 和 FP16 可能影响数值精度。用于生产前应使用真实预训练权重和 ImageNet 验证集检查 Top-1/Top-5。
- 更换硬件、DTK、PyTorch、batch size 或输入尺寸后，应重新建立 MIOpen 和 `torch.compile` 缓存并重新测试。
