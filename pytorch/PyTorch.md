# PyTorch

## 第一章 PyTorch介绍

### 1.1 PyTorch概念

- PyTorch是一个基于Python的开源机器学习框架，由Facebook的AI研究团队（现Meta AI）于2016年推出，现在由Linux基金会托管。它以动态计算图（Dynamic Computational Graph）、灵活性和高效的GPU加速能力著称，是学术界和工业界广泛使用的深度学习工具。

### 1.2 PyTorch核心

#### 1.2.1 动态计算图（**Dynamic Computation Graph**）

- **定义** ：PyTorch使用**即时执行（Eager Execution）**模式，允许在运行时动态构建和修改计算图。
- **优势** ：调试直观（与普通Python代码行为一致），适合需要灵活控制网络结构的场景（如NLP中的可变长度序列）。

#### 1.2.2 张量（Tensor）

- **定义** ：PyTorch的核心数据结构，类似于NumPy的多维数组，但支持GPU加速和自动微分。

- **关键操作** ：

  ```python
  import torch
  x = torch.tensor([[1, 2], [3, 4]]) # 创建张量
  y = x.to("cuda")                   # 将张量移动到GPU
  z = torch.matmul(x, y)             # 矩阵乘法 
  ```

#### 1.2.3 自动微分（Autograd）

- **机制** ：通过记录张量操作构建计算图，反向传播时自动计算梯度。

  ```python
  x = torch.tensor(3.0, requires_grad=True)
  y = x**2 + 2*x
  y.backward()                        # 自动计算 dy/dx
  print(x.grad)                       # 输出梯度值：8.0
  ```

#### 1.2.4 模块化设计

- **核心组件**：
  - <u>nn.Module</u> ：所有神经网络模块的基类（如层、损失函数）。
  - <u>nn.Sequential</u> : 快速堆叠多层网络。
  - 预训练模型库（如 <u>torchvision.models</u>）

### 1.3 PyTorch生态

| 工具/库                  | 用途                     |
| --------------------- | ---------------------- |
| **TorchServe**        | 模型服务部署                 |
| **PyTorch Lightning** | 简化训练流程（自动分布式训练、日志）     |
| **TorchX**            | 大规模作业调度（Kubernetes 集成） |
| **Captum**            | 模型可解释性分析               |

### 1.4 PyTorch应用领域

| 领域          | 典型任务                   | PyTorch 库                 |
| ----------- | ---------------------- | ------------------------- |
| 计算机视觉（CV）   | 图像分类、目标检测              | `torchvision`             |
| 自然语言处理（NLP） | 文本生成、机器翻译              | `transformers`, `fairseq` |
| 强化学习（RL）    | 游戏 AI 训练               | `PyTorch Lightning`       |
| 生成模型        | 图像生成（Stable Diffusion） | `diffusers`               |

### 1.5 PyTorch vs TensorFlow

| 特性         | PyTorch              | TensorFlow           |
| ---------- | -------------------- | -------------------- |
| **计算图**    | 动态图（即时执行）            | 静态图（默认） + 动态图（Eager） |
| **调试难度**   | 低（与 Python 无缝集成）     | 中等（需熟悉图结构）           |
| **部署便捷性**  | 中等（需 TorchScript 转换） | 高（SavedModel 直接部署）   |
| **学术界使用率** | 高（70%+ 顶会论文）         | 中低（逐渐被 PyTorch 取代）   |



## 第二章 PyTorch安装

### 2.1 CPU安装

- **适用场景** ：所有设备均可安装，不需要考虑是否存在显存。

- **安装方法** ：

  ```shell
  pip install torch
  ```

### 2.2 GPU安装

**适用场景** ： 

- 显存>=6GB（建议8GB以上）。
- 需安装 **CUDA**。

方式一：通过[PyTorch官网](https://pytorch.org/)下载安装。

方式二：通过[静态资源库](https://download.pytorch.org/whl/torch_stable.html)下载安装（推荐方式）。