# 神经网络(ANN/FNN/FCNN)

### 1. 神经网络定义

- 神经网是一种受生物系统启发而设计的计算模型，能够通过数据学习复杂的模式和关系。它由大量相互连接的“神经元”组成，通过调整连接权重来模拟人脑的学习过程。
- 神经网络是人工智能的 “基础设施”，通过模拟人脑学习机制，赋予机器感知、推理和决策能力。它如同一台可以自我升级的万能转换器，将原始数据转换为人类所需的智能输出。



### 2. 神经网络核心比喻

可以将神经网络比作成为 “**像乐高积木搭建的智能工厂**”

- **输入**：原材料（如图片、像素、文件、声音以及信号等）。
- **处理** ：流水线的功能（神经元）对材料层层加工。
- **输出** ：最终产品（如分类结果、预测值）。
- **学习** ：质检员（损失函数）检查产品误差，反馈给工人调整加工方式（权重更新）。




### 3. 损失函数（Loss Function）

#### 3.1 概念

- 损失函数是衡量模式预测结果与真是标签之间的差距的数学函数。
- 目标：通过最小化损失函数，调整网络权重，使预测更接近真实值。

#### 3.2 常见类型

- **均方误差（MSE）** ： 用于回归问题。
  $$
  L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
  $$

- **交叉熵损失（Cross-Entropy）**：用于分类问题。
  $$
  L = -\sum_{i=1}^{n} y_i \log(\hat{y}_i)
  $$

- **Huber Loss**：对离群值鲁棒的损失函数。

作用：提供模型性能的量化指标，指导反向传播的梯度计算方向。



### 4. 链式法则

#### 4.1 概念

- 链式法则是微积分中用于计算复合函数导数的规则。

- 公式： 如 z = f(g(x))，则：
  $$
   \frac{dz}{dx} = \frac{dz}{dg}.\frac{dg}{dx}
  $$








#### 4.2 应用

- 神经网络是多个符合函数的堆叠（如线性变换+激活函数）。
- 反向传播通过链式法则，将总损失对权重的梯度分解为各层局部梯度的乘积。

#### 4.3 计算图示例

假设网络结构：

输入 𝑥→线性层→𝑧=𝑤𝑥+𝑏→激活层→𝑎=𝜎(𝑧)→损失𝐿x

- 梯度计算：
  $$
  \frac{∂L}{∂w} = \frac{∂L}{∂a} . \frac{∂a}{∂z} . \frac{∂z}{∂w}
  $$










### 5. 反向传播

#### 5.1 概念

- 反向传播是一种高效计算损失函数对网络权重梯度的方法。
- 核心思想：从输出层向输入层反向逐层计算梯度，利用链式法则传递误差。

#### 5.2 步骤

1. **向前传播** ：输入数据通过网络计算预测值。
2. **计算损失** ：通过损失函数得到预测误差。
3. **反向传播梯度** ：从输出层开始，逐层计算损失对每层权重的偏导数（梯度）。利用链式法则将误差分解到每个参数。

示例：单层网络
$$
\begin{aligned}
\hat{y} &= \sigma(wx + b) \quad &&\text{(前向传播)} \\
L &= \frac{1}{2}(y - \hat{y})^2 \quad &&\text{(损失函数)}
\end{aligned}
$$

- 计算梯度：
  $$
  \frac{\partial L}{\partial w} = 
  \underbrace{(y - \hat{y})}_{\text{损失梯度}} \cdot 
  \underbrace{\sigma'(wx + b)}_{\text{激活函数梯度}} \cdot 
  \underbrace{x}_{\text{输入}}
  $$










### 6. 损失函数、链式法则及反向传播协作流程

1. **前向传播** ：输入数据通过各层计算，得到预测值。

2. **计算损失** ：通过损失函数量化预测误差。

3. **反向传播** ： 

   - 从输出层开始，逐层计算算是对权重的梯度。
   - 使用链式法则将梯度分解到每一层的参数。

4. **参数更新** ：利用梯度下降等优化算法调整权重
   $$
   w←w−η 
   \frac{∂L}{∂w}
   $$








直接比喻：

- 损失函数：类似于考试，分数越低代表模型越优秀。

- 反向传播：老师批改试卷后，将错误原因（梯度）从最后一题（输出层）逐题向前传递，找出每道题的扣分原因。

- 链式法则：扣分原因传递的规则，确保每一步的误差都能追溯到对应的错误步骤。

  ​

### 7. 权重初始化

#### 7.1 概念

- 权重初始化是影响模型性能和稳定性的三大核心要素之一，决定模型训练的起点，需匹配激活函数特性（如 Xavier 对应 Sigmoid，He 对应 ReLU）。
- **避免对称性陷阱** ：若所有权重初始化相同值，神经元会学习相同的特征，降低网络容量。
- **控制梯度传播** ：初始化不当会导致梯度消失（权重过小）或报错（权重过大）。

#### 7.2 常见初始化方法

- **Xavier/Glorot** ： 适用于 Sigmoid、Tang激活

$$
公式  ： 
w \sim U\left(- \sqrt{\frac{6}{{n_{\text{in}} + n_{\text{out}}}}}, \sqrt{\frac{6}{{n_{\text{in}} + n_{\text{out}}}}}\right)
$$

​	

- **He初始化** ： ReL及其变体（如Leaky ReLU）
  $$
  公式 ： w \sim N(0,  \sqrt{\frac{{2}}{n_{\text{in}}}})
  $$

- **零初始化** ： ❌ 实际中禁用（导致对称性）
  $$
  公式 ：w = 0
  $$

- **随机初始化** ： 简单任务（需谨慎使用）
  $$
  公式 ： w \sim N(0, 0.01)
  $$








#### 7.3 数字直接

- Xavier初始化考虑输入纬度和输出纬度，确保各层激活值得方差一致。
- He初始化针对ReLU的“死区”（输出一半为0），通过调整方差保持梯度稳定。



### 8. 矩阵运算（Matrix Operations）

#### 8.1 概念

- 神经网络的骨架，高效的矩阵实现是 GPU 加速的关键。
- 神经网络中的矩阵表示 : 每一层的前向传播本质上是 **矩阵乘法 + 偏置 + 激活函数** ：

$$
Z^{(l)} = W^{(l)} \cdot A^{(l-1)} + b^{(l)}
$$

$$
A^{(l)} = f\left(Z^{(l)}\right)
$$

**符号说明**：
- $W^{(l)}$：第 $l$ 层的权重矩阵（维度：$n_{\text{out}} \times n_{\text{in}}$）
- $A^{(l-1)}$：前一层的输出（维度：$n_{\text{in}} \times \text{batch\_size}$）
- $b^{(l)}$：偏置向量（维度：$n_{\text{out}} \times 1$）

#### 8.2 维度匹配示例

- 输入数据：100纬特征，批量大小 32 ➝ $A^{(0)}$ 纬度100x32


- 隐藏层：50个神经元 ➝ $W^{(1)}$ 纬度 50X100
- 输出 $Z^{(1)}$ 维度：$50 \times 32$  ➝ 激活后 $A^{(1)}$ 维度：$50 \times 32$（与 $Z^{(l)}$ 相同）

#### 8.3 矩阵运算的优化

- **并行计算** : 使用 **GPU 加速矩阵乘法**（如 CUDA 库）。
  - 单批次内所有样本的运算可并行处理。
- **内存效率** ：避免显存溢出（如分块计算大矩阵）。
  - 减少中间变量存储
  - 复用内存空间



### 9. 正则化

#### 9.1 概念

- 模型的“刹车系统”，通过约束权重或结构复杂性防止过拟合。


- 目标：防止模型过拟合，提升泛化能力。

#### 9.2 常见正则化方法

| 方法             | 原理                      | 数学形式（损失函数 𝐿L）                           |
| -------------- | ----------------------- | ---------------------------------------- |
| **L2 正则化**     | 惩罚权重的大幅值，倾向于分散的权重分布     | $𝐿=𝐿原始+𝜆∑𝑤_{\text{i}}^2$             |
| **L1 正则化**     | 鼓励权重稀疏化（部分权重为0），适用于特征选择 | $L = L_{\text{原始}} + \lambda \sum$       |
| **Dropout**    | 训练时随机丢弃神经元，强制网络学习冗余特征   | 前向传播时按概率 𝑝 置零神经元输出                      |
| **Batch Norm** | 标准化每层输入，加速训练并轻微正则化      | $\hat{x} = \frac{x-u}{\sqrt{{{a^2 + e}}}}$ |

**L2正则化的梯度更新**

- 权重更新公式（梯度下降）：
  $$
  w_i ← w_i - η(\frac{∂L_{原始}}{∂w_i} + 2λw_i)
  $$

- 直观效果：每次更新时权重会衰减（乘以1-2ηλ）。

#### 9.3 正则使用

- **L2 正则化**：在损失函数中添加权重衰减项（如 𝜆=0.0001λ=0.0001）。
- **Dropout**：在全连接层后添加概率为 0.5 的 Dropout。
- **Batch Norm**：在卷积层后添加 BN 层加速训练。

#### 9.4 效果对比

- **无正则化**：训练准确率 100%，验证准确率 70%（严重过拟合）。
- **加入正则化**：训练准确率 95%，验证准确率 88%（泛化能力提升）。



### 10. 权重初始化、矩阵运算与正则化协作

#### 10.1 价值

- 神经网络训练中的**权重初始化**、**矩阵运算**和**正则化**是影响模型性能和稳定性的三大核心要素。

#### 10.2 步骤分解

1. **初始化** ：使用He初始化适应ReLU激活函数，确保梯度稳定传播。
2. **矩阵运算** ： 输入图像（224x224x3）展平为150528纬向量，通过全连接层降纬至4096纬。
3. **正则化** ：防止过拟合，提高其泛化能力，避免死记硬背。



### 11. 神经网络的三大核心组件

#### 11.1 神经元（Neuron）

- **功能**：接收输入信号，加权求和后通过激活输出函数输出。

- **数学公式**：
  $$
  输出=f(w_{1}x_{1}+w_{2}x_{2}+⋯+w_{n}x_{n}+b)
  $$

  - w：权重（调节输入重要性）
  - 𝑏：偏置（调节触发阈值）
  - 𝑓：激活函数（如ReLU、Sigmoid，提供非线性能力）

- **工作原理** ：

  - **输出与权重的线性组合**

    - 神经元对每个输出乘以对应的权重，再求和并加上偏置：
      $$
      z=w_{1}x_{1}+w_{2}x_{2}+···+w_{n}x_{n}+b
      $$

    - 权重的作用：放大或抑制输入信号（例如：wi>1 增强输入，𝑤𝑖<1wi<1 削弱输入）。

    - 偏置的作用：调整神经元的激活难易程度（类似”阈值“）。

  - **激活函数引入非线性**

    - 线性组合的结果z通过激活函数转换为非线性输出。

    - 常见激活函数：

      | 函数名称        | 公式                                       | 特点                |
      | ----------- | ---------------------------------------- | ----------------- |
      | **Sigmoid** | $𝑓(𝑧) = \frac{1}{1+𝑒^{-z}}$           | 输出范围 (0,1)，适合概率   |
      | **ReLU**    | $𝑓(𝑧)=max⁡(0,𝑧)$                      | 计算高效，缓解梯度消失       |
      | **Tanh**    | $𝑓(𝑧)=\frac{𝑒^𝑧 - 𝑒^{-𝑧}}{𝑒^𝑧 + 𝑒^{-𝑧}}$ | 输出范围 (-1,1)，中心化数据 |
      | **Softmax** | $𝑓(𝑧_𝑖)=\frac{𝑒^{𝑧_i}}{∑_𝑗𝑒^{𝑧_𝑗}}$ | 多分类概率归一化          |

  - **输出传递** ：

    - 激活后的输出会被传递到下一层神经元，作为他们的输入。

- **组合方式** ：

  - 全连接层（Dense Layer） ：每个神经元与前一层的所有神经元连接。

    ```
    示例：输入层(3个神经元 1x3) ➝ 隐藏层(4个神经元 1x4)，共 3x4 = 12个权重

    矩阵相乘条件：
    	1. 第一个矩阵的列数必须等于第二个矩阵的行数。
    	2. 结果矩阵的维度为 (第一个矩阵的行数) × (第二个矩阵的列数)。 
    因此：
    	1×3 矩阵（1 行，3 列）与 3×4 矩阵（3 行，4 列）满足相乘条件（3 列 = 3 行）。结果矩阵维度为 1×4（1 行，4 列）。
    ```

  - 系数连接（卷积层）：神经元仅连接局部输入区域（适用于图像、语音等空间数据）。

- **学习过程** ：

  神经网络的训练本质上是调整神经元权重和偏置的过程：

  1. **前向传播** ： 数据通过神经元逐层计算，得到预测结果。
  2. **损失计算** ：通过损失函数（如交叉熵、均方误差）量化预测误差。
  3. **反向传播** ：利用链式法则计算损失对每个权重和偏置的梯度。
  4. **参数更新** ：使用优化器（如梯度下降）调整权重和偏置。

- **局限性及改进** ：

  1. **线性局限性**
     - 单层神经元无法解决非线性问题（如异或逻辑）。
     - 解决方案：堆叠多层神经元（深度网络）+ 非线性激活函数。
  2. **梯度消失**
     - 深层网络中梯度可能趋近于零（如Sigmoid 函数在饱和区的导数接近零）。
     - 解决方案：使用 ReLU 激活函数、残差连接（ReNet）。
  3. **过拟合**
     - 神经元过多可能导致记忆训练数据而非泛化。
     - 解决方案：正则化（如Dropout）、数据增强。

#### 11.2 网络结构

- **输入层** ：接收原始数据（如784个神经元对应28*28的图片像素）。
- **隐藏层** ：多层神经元处理数据（层数越多，网络越“深”）。
- **输出层** ：生成最终结果（如分类标签、回归值）。

```
示例：识别手写数字的神经网络

输入层（784像素）➝ 隐藏层（128神经元）➝ 输出层（10个数字概率）
```

#### 11.3 学习机制

- **前向传播** ： 数据从输入到输出，逐层计算预测结果。
- **损失函数** ： 量化预测误差（如交叉狄损失、均方误差）。
- **反向传播** ：将误差从输出层反向传递，计算每个权重的调整量级。
- **优化器** ：根据梯度更新权重（如SGD、Adam算法）。



### 12. 神经网络的“超能力”来源

#### 12.1 非线性激活函数

- 使用**ReLU、Sigmoid** 等函数，使网络能够拟合任意复杂函数。

  ```
  示例：ReLU函数 f(x) = max(0,x) 可快速过滤无效特征。
  ```

#### 12.2 层次化特征提取

- **浅层** ：识别简单模式（如边缘、颜色）。
- **深层** ：组合成复杂特征（如人脸、物体轮廓）。
- **类比** ：文件 ➝ 字母 ➝ 单词 ➝ 句子 ➝ 段落的理解过程。

#### 12.3 数据驱动学习

- 无需人工设计规则，通过大量数据自动学习。

  ```
  示例：训练10万张猫狗图片后，网络能自动区分新图片中的动物。
  ```




### 13. 神经网络的常见类型

| 类型              | 用途          | 典型结构       |
| --------------- | ----------- | ---------- |
| **全连接网络**       | 基础分类/回归     | 多层感知机（MLP） |
| **卷积网络（CNN）**   | 图像处理、视频分析   | ResNet、VGG |
| **循环网络（RNN）**   | 时序数据（文本、语音） | LSTM、GRU   |
| **Transformer** | 自然语言处理、翻译   | BERT、GPT   |



### 14. 实际应用场景

- **图像识别**：人脸解锁、医学影像分析（如癌症筛查）。
- **自然语言处理**：智能客服、机器翻译（如ChatGPT）。
- **推荐系统**：电商商品推荐、短视频内容推送。
- **自动驾驶**：实时路况感知、决策控制。




### 15. 神经网络的局限性

- **数据依赖** ： 需要大量标注数据，小样本场景效果差。
- **黑箱问题** ：决策过程难以解释（如医疗诊断可能引发信任问题）。
- **计算成本** ：训练大模型需高性能GPU，耗电量大。
- **过拟合风险** ：可能死记硬背训练数据，导致泛化能力差。

### 16. FCNN(全连接神经)和RNN(循环神经网络)区别
- RNN（循环神经网络）中，神经元之间并不是全连接的。

| **特性**     | **全连接网络 (FCNN)** | **RNN**          |
| ---------- | ---------------- | ---------------- |
| **连接方式**   | 层内所有神经元横向全连接     | 跨时间步纵向连接，层内无横向连接 |
| **参数数量**   | 随层数指数增长          | 参数共享，数量固定        |
| **序列处理能力** | 无法处理变长序列         | 天然支持序列建模         |

### 17. 深度学习网络（DNN/MLP）

#### 17.1 概念

- 深度神经网络（DNN）是一种多层次的神经网络模型，通过堆叠多个隐藏层（Hidden Layers）来学习数据的多层次抽象表示。它是深度学习的基础架构，能够自动提取从低到高级的特征，DNN也叫做多层感知机（Multi-Layer-Perceptron,MLP）。

  ![](https://gitee.com/YunboCheng/image-bad/raw/master/imgs/202506270948395.png)

#### 17.2 DNN经典全连接模型

**（1）全连接（Full Connected）**

- **特点**：每一层的每个神经元都与下一层的所有神经元相连，即**密集连接**。

- **数学表达**：
  $$
  z^{(l)} = W^{{l}}a^{(l-1)}+b^{(l)}
  $$

  - $W^{(l)}$是权重矩阵，纬度为$n_{l}$ X $n_{l-1}$($n$为神经元数量)。

- **典型应用** ：MLP（多层感知机）、传统分类/回归任务。

#### 17.3 DNN非全连接变形体模型

**（1）卷积神经网络（CNN）**

- **特点** ：通过**局部连接**和**权重共享**减少参数量。

- **连接方式**：
  - 卷积层的神经元仅连接输入层的局部区域（感受野）。
  - 同一卷积核在不同位置共享权重。


- **优势**：
  适合处理图像、视频等网格化数据，保留空间信息。

**（2）循环神经网络（RNN/LSTM）**

- **特点**：
  通过**时间步展开**和**循环连接**处理序列数据。
- **连接方式**：
  - 同一层的神经元在时间步之间递归连接。
  - 隐藏状态跨时间步传递（非全连接）。

**（3）注意力机制（Transformer）**

- **特点**：
  通过**自注意力**动态计算连接权重。
- **连接方式**：
  - 每个位置（如单词）通过注意力权重与其他位置交互。
  - 连接是稀疏且动态的，而非固定全连接。

**（4）图神经网络（GNN）**

- **特点**：
  基于图结构定义连接方式。
- **连接方式**：
  - 神经元仅连接图中的相邻节点（非全连接）。

#### 17.4 模型关键区别

**（1）连接方式**

- **DNN**：全连接（Fully Connected），相邻层所有神经元两两相连。
- **CNN**：局部连接（卷积核只连接输入区域的局部像素），权重共享（同一卷积核滑动扫描全局）。
- **RNN**：循环连接（隐藏状态跨时间步递归传递），同一层神经元在时间维度上展开。

**（2）数学表达对比**

| **模型** | **前向传播公式**                          | **核心操作**       |
| ------ | ----------------------------------- | -------------- |
| DNN    | $z^{(l)}=W^{(l)}a^{(l−1)}+b^{(l)}$  | 矩阵乘法 + 激活函数    |
| CNN    | $z^{(l)}=W^{(l)}∗a^{(l−1)}+b^{(l)}$ | 卷积运算（*） + 池化   |
| RNN    | $h_{t}=σ(Wh_{t−1}+Ux_{t}+b)$        | 时间步递归 + 隐藏状态传递 |

**(3) 适用场景**

- **DNN**：适合结构化数据（表格数据、简单分类/回归）。
- **CNN**：适合具有**空间相关性**的数据（图像、视频、医学影像）。
- **RNN**：适合具有**时间依赖性**的数据（文本、语音、股票价格）。

