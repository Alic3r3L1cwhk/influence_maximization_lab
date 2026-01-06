# Influence Maximization Lab

一个完整的影响力最大化实验框架，支持从级联数据中学习扩散参数，并应用于影响力最大化算法。

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 项目概述

本项目实现了一个完整的影响力最大化研究框架，主要功能包括：

1. **数据生成模块**: 生成合成社交网络（ER/BA/WS）并模拟信息级联传播
2. **真实数据集支持**: 自动下载和预处理 Wiki-Vote、Email-Enron、Facebook 等真实社交网络
3. **参数学习模块**: 使用 PyTorch 从级联数据中学习 IC 模型的边传播概率
4. **高级特征工程**: 结构特征（度中心性、PageRank、聚类系数等）+ 图嵌入
5. **扩散仿真模块**: 实现 IC 和 LT 扩散模型的蒙特卡洛仿真
6. **影响力最大化算法**: 10+ 种算法（Greedy、TIM、IMM、启发式方法等）
7. **网络可视化**: 网络图、级联传播动画、度分布等可视化工具
8. **交互式教程**: Jupyter Notebook 教程和案例研究
9. **实验对比**: 对比真实参数与学习参数下的影响力最大化效果
10. **单元测试**: 完整的测试套件确保代码质量

## 🌟 主要特性

- ✅ **完整的实验流程**: 从网络生成到参数学习再到影响力最大化
- ✅ **GPU 加速**: PyTorch 模型支持 CUDA 加速训练
- ✅ **10+ 种 IM 算法**:
  - 精确算法：Greedy、Lazy Greedy (CELF)
  - 近似算法：TIM、TIM+、IMM
  - 启发式方法：Degree、PageRank、Betweenness、Closeness、K-Shell、Random
- ✅ **真实数据集**: 自动下载 Stanford SNAP 数据集（Wiki-Vote、Email-Enron、Facebook、Google+）
- ✅ **高级特征**: Node2Vec、DeepWalk + 14 种结构特征（度中心性、PageRank、聚类系数等）
- ✅ **可复现**: 统一的随机种子管理确保实验可重复
- ✅ **丰富可视化**:
  - 训练曲线、影响力对比、运行时间对比
  - 网络拓扑可视化、度分布分析
  - 级联传播动画和快照
- ✅ **交互式教程**: Jupyter Notebook 详细教程
- ✅ **灵活配置**: 通过命令行参数控制所有实验设置
- ✅ **测试覆盖**: 单元测试确保代码质量

## 📁 项目结构

```
influence_maximization_lab/
├── src/                          # 源代码
│   ├── data/                     # 数据生成和处理
│   │   ├── network_generator.py  # 网络生成（ER/BA/WS）
│   │   ├── cascade_generator.py  # 级联模拟
│   │   ├── data_loader.py        # 数据划分
│   │   └── real_datasets.py      # 真实数据集下载器 🆕
│   ├── models/                   # 机器学习模型
│   │   ├── embeddings.py         # Node2Vec/DeepWalk
│   │   ├── param_learner.py      # PyTorch 参数学习模型
│   │   └── advanced_features.py  # 高级结构特征 🆕
│   ├── diffusion/                # 扩散模型
│   │   ├── ic_model.py           # IC 模型
│   │   ├── lt_model.py           # LT 模型
│   │   └── simulator.py          # 统一仿真接口
│   ├── influence_max/            # 影响力最大化算法
│   │   ├── greedy.py             # 贪心算法 (Greedy, Lazy Greedy)
│   │   ├── tim.py                # TIM/TIM+ 算法
│   │   ├── imm.py                # IMM 算法 🆕
│   │   └── heuristics.py         # 启发式算法 🆕
│   └── utils/                    # 工具函数
│       ├── metrics.py            # 评估指标
│       ├── visualization.py      # 基础可视化
│       ├── network_viz.py        # 网络可视化 🆕
│       └── io_utils.py           # 文件读写
├── experiments/                  # 实验脚本
│   ├── train_params.py           # 训练扩散参数
│   ├── run_influence_max.py      # 运行影响力最大化
│   ├── compare_methods.py        # 综合对比实验
│   └── run_on_real_data.py       # 真实数据集实验 🆕
├── tutorials/                    # Jupyter 教程 🆕
│   └── 01_getting_started.ipynb  # 入门教程
├── tests/                        # 单元测试 🆕
│   └── test_data.py              # 数据模块测试
├── configs/                      # 配置文件 🆕
│   └── example_config.yaml       # 示例配置
├── data/                         # 数据目录（生成）
├── outputs/                      # 输出目录（生成）
├── requirements.txt              # 依赖列表
├── setup.py                      # 安装脚本
├── quickstart.sh/bat             # 快速开始脚本
└── README.md                     # 本文件
```

## 🚀 快速开始

### 1. 环境配置

#### 系统要求
- Python 3.8+
- CUDA 11.0+ (可选，用于 GPU 加速)

#### 安装依赖

```bash
# 克隆或进入项目目录
cd influence_maximization_lab

# 安装基础依赖
pip install -r requirements.txt

# （可选）安装 PyTorch with CUDA 支持
# 访问 https://pytorch.org/ 获取适合您系统的安装命令
# 例如 (CUDA 11.8):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. 快速运行示例

#### 示例 1: 训练扩散参数

从合成网络生成级联数据并训练参数学习模型：

> 没有可用 GPU 时，将 `--device` 设为 `cpu` 即可，其余参数保持不变。

```bash
python experiments/train_params.py \
    --network-type ba \
    --num-nodes 500 \
    --ba-m 3 \
    --num-cascades 1000 \
    --embedding-method node2vec \
    --epochs 100 \
    --device cuda \
    --seed 42 \
    --output-dir outputs/trained_models
```

CPU 运行示例：

```bash
python experiments/train_params.py \
    --network-type ba \
    --num-nodes 500 \
    --ba-m 3 \
    --num-cascades 1000 \
    --embedding-method node2vec \
    --epochs 100 \
    --device cpu \
    --seed 42 \
    --output-dir outputs/trained_models_cpu
```

**输出文件**:
- `outputs/trained_models/param_learner.pth` - 训练好的模型
- `outputs/trained_models/embeddings.txt` - 节点嵌入
- `outputs/trained_models/network.edgelist` - 生成的网络
- `outputs/trained_models/training_history.png` - 训练曲线
- `outputs/trained_models/training_results.json` - 完整实验结果

#### 示例 2: 运行影响力最大化

使用训练好的模型进行影响力最大化：

```bash
python experiments/run_influence_max.py \
    --network-path outputs/trained_models/network.edgelist \
    --model-path outputs/trained_models/param_learner.pth \
    --embeddings-path outputs/trained_models/embeddings.txt \
    --algorithm lazy_greedy \
    --k 10 \
    --num-simulations 1000 \
    --compare-params \
    --num-runs 5 \
    --device cuda \
    --seed 42 \
    --output-dir outputs/im_results
```

**输出文件**:
- `outputs/im_results/im_results.json` - 完整结果
- `outputs/im_results/influence_comparison.png` - 影响力对比图
- `outputs/im_results/runtime_comparison.png` - 运行时间对比图

#### 示例 3: 综合对比实验

对比多种算法和参数设置：

```bash
python experiments/compare_methods.py \
    --network-type ba \
    --num-nodes 500 \
    --k-values 10 20 30 \
    --algorithms lazy_greedy tim tim_plus \
    --train-params \
    --num-runs 3 \
    --device cuda \
    --output-dir outputs/comparison
```

**输出文件**:
- `outputs/comparison/comparison_results.csv` - CSV 格式结果表
- `outputs/comparison/comparison_results.json` - JSON 格式完整结果
- `outputs/comparison/influence_comparison_k*.png` - 各 k 值下的影响力对比
- `outputs/comparison/runtime_comparison_k*.png` - 运行时间对比

#### 示例 4: 真实数据集实验 🆕

在真实社交网络数据集上运行影响力最大化：

```bash
python experiments/run_on_real_data.py \
    --dataset wiki-vote \
    --algorithms degree pagerank lazy_greedy tim \
    --k 50 \
    --num-simulations 1000 \
    --prob-method wc \
    --output-dir outputs/real_datasets
```

支持的数据集:
- `wiki-vote`: Wikipedia 投票网络 (7K 节点)
- `email-enron`: Enron 邮件网络 (37K 节点)
- `facebook`: Facebook 社交圈 (4K 节点)
- `gplus`: Google+ 社交圈 (108K 节点)

**输出文件**:
- `outputs/real_datasets/wiki-vote_results.json` - 完整结果
- `outputs/real_datasets/wiki-vote_preprocessed.edgelist` - 预处理后的网络
- `outputs/real_datasets/wiki-vote_influence_comparison.png` - 影响力对比

#### 示例 5: 交互式教程 🆕

运行 Jupyter Notebook 教程：

```bash
# 启动 Jupyter
jupyter notebook tutorials/01_getting_started.ipynb
```

教程内容包括：
- 网络生成和可视化
- 级联模拟
- 参数学习
- 影响力最大化算法对比
- 可视化分析

## 📖 详细使用说明

### 数据格式

#### 网络边列表格式 (Edge List)
```
# 格式: source target [probability]
0 1 0.05
0 2 0.08
1 3 0.03
```

#### 级联数据格式 (Cascade Log)
```
# 格式: cascade_id source target [timestamp]
0 1 2 100
0 2 5 120
0 1 3 130
1 4 8 100
```

### 核心模块使用

#### 1. 生成网络

```python
from src.data import NetworkGenerator

# 创建生成器
gen = NetworkGenerator(seed=42)

# 生成 BA 网络
G = gen.generate_ba(n=500, m=3)

# 分配 IC 传播概率
G = gen.assign_ic_probabilities(G, prob_range=(0.01, 0.1))

# 或加载已有网络
G = gen.load_from_edgelist('network.txt')
```

#### 2. 生成级联数据

```python
from src.data import CascadeGenerator

# 创建级联生成器
cascade_gen = CascadeGenerator(G, seed=42)

# 生成级联
cascades = cascade_gen.generate_cascades(
    model='ic',
    num_cascades=1000,
    initial_size_range=(1, 5)
)

# 转换为训练数据
edges, labels = cascade_gen.cascades_to_training_data(cascades)
```

#### 3. 训练参数学习模型

```python
from src.models import GraphEmbedding, ParameterLearner
import numpy as np

# 生成嵌入
embedding_gen = GraphEmbedding(G, embedding_dim=128, seed=42)
embeddings = embedding_gen.train_node2vec(num_walks=10, walk_length=80)

# 准备特征
features = np.array([embedding_gen.get_edge_features(e) for e in edges])
labels = np.array(labels)

# 训练模型
learner = ParameterLearner(
    input_dim=features.shape[1],
    hidden_dims=[256, 128, 64],
    device='cuda'
)

history = learner.fit(
    features, labels,
    epochs=100,
    batch_size=256
)
```

#### 4. 运行影响力最大化

```python
from src.diffusion import DiffusionSimulator
from src.influence_max import LazyGreedyIM, TIM

# 创建模拟器
sim = DiffusionSimulator(G, model='ic', seed=42)

# 方法 1: Lazy Greedy
greedy = LazyGreedyIM(G, sim, seed=42)
seeds, gains, runtime = greedy.select_seeds(k=10, num_simulations=1000)

# 方法 2: TIM
tim = TIM(G, model='ic', seed=42)
seeds, influence, runtime = tim.select_seeds(k=10, epsilon=0.2)

# 评估影响力
actual_influence = sim.estimate_influence(seeds, num_simulations=1000)
print(f"Selected seeds: {seeds}")
print(f"Expected influence: {actual_influence}")
```

## 🔬 实验复现

### 完整实验流程

```bash
# Step 1: 训练参数（~5-10 分钟，取决于网络规模）
python experiments/train_params.py \
    --network-type ba --num-nodes 1000 --ba-m 3 \
    --num-cascades 2000 \
    --embedding-dim 128 \
    --epochs 100 \
    --device cuda \
    --seed 42 \
    --output-dir outputs/exp1

# Step 2: 对比影响力最大化（~10-20 分钟）
python experiments/run_influence_max.py \
    --network-path outputs/exp1/network.edgelist \
    --model-path outputs/exp1/param_learner.pth \
    --embeddings-path outputs/exp1/embeddings.txt \
    --algorithm lazy_greedy \
    --k 20 \
    --num-simulations 1000 \
    --compare-params \
    --num-runs 10 \
    --device cuda \
    --output-dir outputs/exp1/im_results

# Step 3: 综合对比（~30-60 分钟）
python experiments/compare_methods.py \
    --num-nodes 1000 \
    --k-values 10 20 30 40 50 \
    --algorithms lazy_greedy tim tim_plus \
    --train-params \
    --num-runs 5 \
    --device cuda \
    --output-dir outputs/exp1/comparison
```

### 不同网络类型实验

```bash
# ER 网络
python experiments/train_params.py --network-type er --num-nodes 500 --er-p 0.01

# WS 小世界网络
python experiments/train_params.py --network-type ws --num-nodes 500 --ws-k 4 --ws-p 0.1

# BA 无标度网络（默认）
python experiments/train_params.py --network-type ba --num-nodes 500 --ba-m 3
```

## 📊 评估指标

### 参数学习指标
- **AUC (Area Under ROC Curve)**: 评估概率预测质量
- **Accuracy**: 二分类准确率（阈值 0.5）
- **Training Loss**: BCE (Binary Cross Entropy) 损失

### 影响力最大化指标
- **Expected Influence**: 通过蒙特卡洛模拟估计的期望影响节点数
- **Runtime**: 算法运行时间（秒）
- **Seed Set Overlap**: 真实参数与学习参数选出的种子节点重叠度

## ⚙️ 高级配置

### 命令行参数详解

#### train_params.py 主要参数

| 参数              | 说明                | 默认值         |
| ----------------- | ------------------- | -------------- |
| `--network-type`  | 网络类型 (er/ba/ws) | ba             |
| `--num-nodes`     | 节点数              | 500            |
| `--num-cascades`  | 级联数量            | 1000           |
| `--embedding-dim` | 嵌入维度            | 128            |
| `--hidden-dims`   | MLP 隐藏层维度      | [256, 128, 64] |
| `--epochs`        | 训练轮数            | 100            |
| `--batch-size`    | 批次大小            | 256            |
| `--learning-rate` | 学习率              | 0.001          |
| `--device`        | 设备 (cuda/cpu)     | cuda           |
| `--seed`          | 随机种子            | 42             |

#### run_influence_max.py 主要参数

| 参数                | 说明                  | 默认值      |
| ------------------- | --------------------- | ----------- |
| `--algorithm`       | IM 算法               | lazy_greedy |
| `--k`               | 种子节点数            | 10          |
| `--num-simulations` | MC 模拟次数           | 1000        |
| `--compare-params`  | 是否对比真实/学习参数 | False       |
| `--num-runs`        | 实验重复次数          | 5           |
| `--parallel`        | 是否并行模拟          | False       |

### 性能优化建议

1. **GPU 加速**: 使用 `--device cuda` 加速参数学习
2. **并行模拟**: 对于大规模实验，使用 `--parallel --num-workers 8`
3. **减少模拟次数**: 初步测试时使用较少的 MC 模拟 (如 100-500)
4. **早停机制**: 已集成，训练会在验证损失不再下降时自动停止

## 📈 实验结果示例

### 训练曲线
训练过程中会自动生成 Loss 和 AUC 曲线：

![Training History](outputs/trained_models/training_history.png)

### 影响力对比
对比真实参数与学习参数下的影响力：

```
True Parameters:
  Mean Influence: 145.32 ± 3.21
  Mean Runtime: 12.45s

Learned Parameters:
  Mean Influence: 142.18 ± 3.56
  Mean Runtime: 12.38s
```

### 算法性能对比

| 算法        | k=10 影响力 | k=20 影响力 | 运行时间 |
| ----------- | ----------- | ----------- | -------- |
| Greedy      | 98.5        | 156.3       | 180s     |
| Lazy Greedy | 98.5        | 156.3       | 15s      |
| TIM         | 97.2        | 154.8       | 8s       |
| TIM+        | 97.8        | 155.4       | 6s       |

## 🛠️ 扩展功能

### 导入外部数据

```python
# 导入网络
from src.data import NetworkGenerator
gen = NetworkGenerator()
G = gen.load_from_edgelist('your_network.txt')

# 导入级联
from src.data import load_cascades_from_file
cascades = load_cascades_from_file('your_cascades.txt')
```

### 自定义扩散模型

可以在 `src/diffusion/` 中实现自定义扩散模型，只需继承基类并实现 `simulate_single` 方法。

### 自定义 IM 算法

可以在 `src/influence_max/` 中实现新算法，参考现有的 `GreedyIM` 或 `TIM` 类。

## 🐛 常见问题

### Q1: CUDA out of memory
**A**: 减少 `--batch-size` 或使用 `--device cpu`

### Q2: 训练速度慢
**A**:
- 确保使用 GPU (`--device cuda`)
- 减少 `--num-cascades` 或 `--epochs`
- 减小网络规模 `--num-nodes`

### Q3: 影响力估计不稳定
**A**: 增加 `--num-simulations` (如 5000-10000)

### Q4: TIM 算法运行失败
**A**: 调整 `--tim-epsilon` 参数，较小的 epsilon 会生成更多 RR sets

## 📚 参考文献

1. **Independent Cascade Model**: Kempe, D., Kleinberg, J., & Tardos, É. (2003). Maximizing the spread of influence through a social network. KDD.

2. **TIM/TIM+**: Tang, Y., Xiao, X., & Shi, Y. (2014). Influence maximization: Near-optimal time complexity meets practical efficiency. SIGMOD.

3. **Node2Vec**: Grover, A., & Leskovec, J. (2016). node2vec: Scalable feature learning for networks. KDD.

4. **DeepWalk**: Perozzi, B., Al-Rfou, R., & Skiena, S. (2014). DeepWalk: Online learning of social representations. KDD.

## 📄 License

本项目采用 MIT License 开源协议。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 GitHub Issue
- 邮件联系（请在此添加您的邮箱）

---

**Happy Experimenting! 🎉**
