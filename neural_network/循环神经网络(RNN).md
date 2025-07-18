# 循环神经网络(RNN)

## 第一章 循环神经网络介绍

### 1.1 概念

- **RNN（Recurrent Neural Network）**是转为处理**序列数据**（如文本、时间序列、语音）设计的神经网络，核心思想是引入**时间维度**的记忆能力。与传统神经网络不同，RNN的隐藏状态（Hidden State）会传递到下一个时间步，从而捕获序列中的上下文依赖关系。

- 其中包含**长短时记忆网络（LSTM）**、**门控循环单元（GRU）**和**双向循环神经网络（Bi-RNN）**。

  ![](https://gitee.com/YunboCheng/image-bad/raw/master/imgs/202505301535648.png)

### 1.2 核心要点

#### 1.2.1 参数共享

- 所有时间步共享同一组权重（如输入到隐藏层的$W_{xh}$、隐藏层到自身的$W_{hh}$），大幅减少参数量。
- 公式：$h_{t}=σ(W_{xh}x_{t}+W_{hh}h_{t−1}+b_{h})$

#### 1.2.2 隐藏状态（Hidden State）

- 保存历史信息，随时间步传递，用于捕获长期依赖（但普通的RNN存在梯度消失问题，难以捕捉长距离依赖）

#### 1.2.3 处理变长序列

- 动态展开时间步，支持输入和输出长度灵活变化（如机器翻译中不同长度的句子）。

#### 1.2.4 典型问题

- 梯度消失/爆炸：长序列训练时，梯度可能指数级衰减或增长。



### 1.3 循环神经网络结构

- 神经网络是一类具体内部环状连接的人工神经网络，用于处理序列数据。其最大的夜店是网络中存在着环，使得信息能在网络中进行循环，实现对序列信息的存储和处理。

  ![](https://gitee.com/YunboCheng/image-bad/raw/master/imgs/202505301539672.png)

  ```python
  # 一个简单的RNN结构示例
  class SimpleRNN(nn.Module):
      def __init__(self, input_size, hidden_size):
          super(SimpleRNN, self).__init__()
          self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
          
      def forward(self, x):
          out, _ = self.rnn(x)
          return out
  ```



### 1.4 RNN的优缺点

#### 1.4.1 优点

- 能够处理不同长度的序列数据。
- 能够捕捉序列中的时间依赖关系。

#### 1.4.2 缺点

- 对序列的记忆能力较弱，可能出现梯度消失或梯度爆炸问题。
- 训练可能相对复杂和时间消耗大。



### 1.5 总结

- 循环神经网络是一种强大的模型，特别适合于处理具有时间依赖性的序列数据。然而，标准RNN通常难以学习长序列中的依赖关系，因此有了更多复杂的变体（如LSTM、GRU）等。



## 第二章 循环神经网络工作原理

### 2.1 工作原理

- 循环神经网络（RNN）的工作原理是通过网络中的环状连接捕获序列中的时间依赖关系。

### 2.2 RNN的时间展开

- RNN的一个重要特点是可以通过时间展开来理解。这意味着，虽然网络结构在每个时间步看起来相同，但我们可以将其展开为一系列的网络层，每一层对应序列中一个特定时间步。

![](https://gitee.com/YunboCheng/image-bad/raw/master/imgs/202505301542812.png)



![](https://gitee.com/YunboCheng/image-bad/raw/master/imgs/202505301552631.png)



- **输入层** ： RNN能够接受一个输入序列（例如文字、股票价格、语音信号等）并将其传递到隐藏层。
- **隐藏层** ：隐藏层之间存在循环连接，使得网络能够维护一个“记忆”状态，这一状态包含了过去的信息。使得RNN能够理解序列中上下文信息。
- **输出层** ：RNN可以有一个或多个输出，例如在序列生成任务中，每个时间步都会有一个输出。

### 2.3 信息流动

![](https://gitee.com/YunboCheng/image-bad/raw/master/imgs/202505301608721.png)

- **输入到隐藏** ：每个时间步，RNN从输入层接收到一个新的输入，并将其与之前的隐藏状态结合起来，以生成新的隐藏状态。
- **隐藏到隐藏** ：隐藏层之间的循环连接使得信息可以在时间步之间传播，从而捕获序列中的依赖关系。
- **隐藏到输出** ：每个时间步的隐藏状态都会传递到输出层，以生成对应的输出。



### 2.4 实现示例

```python
# RNN的PyTorch实现
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h_0):
        out, h_n = self.rnn(x, h_0) # 运用RNN层
        out = self.fc(out) # 运用全连接层
        return out
```



### 2.5 梯度问题

- **梯度爆炸/消失** ： 由于RNN的循环结构，在训练中可能会出现梯度消失或梯度爆炸的问题。长序列可能会导致训练过程中的梯度变得非常小（消失）或非常大（爆炸），从而影响模型的学习效率。



### 2.6 总结

- 循环网络中的工作原理强调了序列数据的时间依赖关系。通过时间展开和信息的连续流动，RNN能够理解和处理序列中的复杂模式。不过，RNN的训练可能受到梯度消失或爆炸的挑战，需要采用适当的技术来克服。



## 第三章 循环神经网络的应用场景

- 循环神经网络（RNN）因其在捕获序列数据中的时序依赖性方面的优势，在许多应用场景中得到了广泛的使用。RNN的这些应用场景共同反映了其在理解和处理具有时序依赖关系的序列数据方面的强大能力。

![](https://gitee.com/YunboCheng/image-bad/raw/master/imgs/202505301618368.png)

### 3.1 文本分析与生成



#### 3.1.1 自然语言处理

- RNN可用于词性标注、命名实体识别、句子解析等任务。通过捕获文本中的上下文关系，RNN能够理解并处理语言的复杂结构。

#### 3.1.2 机器翻译

- RNN能够理解和生成不同语言的句子结构，使其在机器翻译方面特别有效。

#### 3.1.3 文本生成

- 利用RNN进行文本生成，如生成诗歌、故事等。实现了机器的创造性写作。



### 3.2 语音识别与合成

#### 3.2.1 语音到文本

- RNN可以用于将语音信号转换为文字，即语音识别（Speech To Text），理解声音中的时序依赖关系。

#### 3.2.2 文本到语音

- RNN也用于文本到语音（Text To Speech）转换，生成流畅自然的语言。



### 3.3 时间序列分析

#### 3.3.1 股票预测

- 通过分析历史股票价格和交易数等数据的时间序列，RNN可以用于预测未来的股票走势。

#### 3.3.2 气象预报

- RNN通过分析气象数据的时间序列，可以预测未来的天气情况。



### 3.4 视频分析与生成

#### 3.4.1 动作识别

- RNN能够分析视频中的时序信息，用于识别人物动作和行为模式。

#### 3.4.2 视频生成

- RNN还可以用于视频内容的生成，如生成具有联系逻辑的动画片段。




## 第四章  循环神经网络的变体（LSTM）

### 4.1 长短时记忆网络（LSTM）

- LSTM是一种特殊的RNN结构，LSTM旨在解决传统RNN在训练长序列时遇到的梯度消失问题。

![](https://gitee.com/YunboCheng/image-bad/raw/master/imgs/202505301702342.png)

### 4.2 LSTM的结构

- LSTM 是 RNN 的改进模型，通过引入**细胞状态（Cell State）** 和**门控机制**，解决了普通 RNN 难以捕捉长期依赖的问题。其核心思想是：
  - **细胞状态** ：一条”传送带“，长期保存信息，跨时间步传递时**几乎不衰减**。
  - **门控机制** ：通过控制三个门（遗忘门、输入门、输出门）**选择性**更新、遗忘或输出信息。

![](https://gitee.com/YunboCheng/image-bad/raw/master/imgs/202505301703154.png)

#### 4.2.1 遗忘门（Forget Gate）

- 决定从细胞状态$C_{t-1}$中丢弃哪些信息。

#### 4.2.2 输入门（Input Gate）

- 绝对将哪些新信息存入细胞状态。

#### 4.2.3 更新细胞状态（Cell State）

- 存储过去的信息，通过遗忘门和输入门的调节进行更新。

#### 4.2.4 输出门（Output Gate）

- 绝对从细胞状态$C_{t}$中输出哪些信息到隐藏状态$h_{t}$。


### 4.3 关键组件的作用

#### 4.3.1 细胞状态（Cell State）

- **核心特征**：
  - 线性传递（仅通过逐元素乘法和加法更新），梯度衰减极慢，适合长期记忆。
  - 例如：在文本生成中，细胞状态可记住段落开头的主题（如”科技文章“），直到段落结束。

#### 4.3.2 门控机制的意义

- **动态调节信息流**：
  - **遗忘门** ：丢弃无关信息（如句子中的停用词）。
  - **输入门** ：添加关键新信息（如新出现的主语）。
  - **输出门** ：控制当前时间步的输出（如生成动词时依赖主语）。

#### 4.3.3 隐藏状态（Hidden State）

- 对外输出的”接口“，携带短期记忆信息，传递给下一时间步或用于预测。

### 4.4 Dropout

#### 4.4.1 Dropout的作用原理

Dropout是一种正则化技术，用于防止过拟合。当配置0.5时，它的作用机制和意义如下：

- **随机失活神经元** ：在训练阶段的每次前向传播中，每个神经元以概率$p$（如0.5）被暂时"关闭"（输出置零），不参与当前迭代的计算和参数更新。
- **测试阶段不丢弃** ：验证和测试阶段所有的神经元均保留，必须关系Dropout，确保模型使用完整的网络结构进行预测。

![](https://gitee.com/YunboCheng/image-bad/raw/master/imgs/202506111447586.png)

| **阶段**    | **Dropout 行为**                           | **目的**                   |
| --------- | ---------------------------------------- | ------------------------ |
| **训练阶段**  | 每次前向传播随机丢弃部分神经元（例如按概率 p=0.5 关闭 50% 的神经元）。 | 防止神经元过度依赖特定特征，增强模型的泛化能力。 |
| **验证/测试** | **关闭 Dropout**，所有神经元均参与计算，输出保持稳定。        | 准确评估模型性能，避免随机性干扰预测结果。    |

#### 4.4.2 使用Dropout的意义

1. **防止过拟合（核心目标）** ：强迫网络不依赖某些特定神经元，迫使每个神经元都能独立提取有用特征。若网络依赖某个“关键神经元”，一旦该神经元失效（如输出变化），模型就会失效，Dropout迫使网络分散特征学习。
2. **隐式模型集成（Model Ensemble）** :  每个迭代随机丢弃不同神经元，相当于训练多个“子网络”，最终模型可以看作是这些子网络的加权平均，提升泛化能力。
3. **增强鲁棒性** ：模型在部分信息缺失仍能工作，类似数据增强（如遮挡、噪音），提升对噪声和变化的容忍度

#### 4.4.3 Dropout的局限性

| **优点**             | **缺点**             |
| ------------------ | ------------------ |
| 简单有效，广泛适用          | 可能延长训练时间（需要更多迭代收敛） |
| 无需额外数据或复杂操作        | 对某些任务（如序列建模）效果有限   |
| 可与其它正则化方法（如权重衰减）结合 | 高 Dropout 率可能导致欠拟合 |

### 4.5 实际示例：文本生成

```python
# LSTM的PyTorch实现
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, (h_0, c_0)):
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0)) # 运用LSTM层
        out = self.fc(out) # 运用全连接层
        return out
```

假设模型处理句子`"The cat, which ate the fish, was _"` ，目标是预测最后一个词（如 ”happy“）。

1. **细胞状态** ： 记住主语”cat“和动作”ate the fish“。
2. **遗忘门** ：在遇到逗号后，逐步遗忘无关信息（如”the“）。
3. **输入门** ：当遇到”which ate the fish“时，将动作信息存入细胞状态。
4. **输出门** ：生成隐藏状态时，结合细胞状态中的主语和动作，预测形容词”happy“。

### 4.6 LSTM的优势

- **长期依赖处理** ： 细胞状态的线性传播减少梯度消失。
- **灵活的信息控制** ： 门控机制动态调节记忆和遗忘。
- **广泛应用** ：机器翻译、语音识别、时间序列预测等序列任务。

通过这种结构，LSTM能够像人类一样”记住关键事件，遗忘无关细节“，从而高效处理复杂序列数据。



## 第五章  循环神经网络的变体（GRU）

### 5.1 门控循环单元（GRU）

-  GRU通过将遗忘门和输入门合并，较少了LSTM的复杂性。相比于LSTM减少参数，提高计算效率，通过门控机制筛选关键信息，改进梯度传播路径。

![](https://gitee.com/YunboCheng/image-bad/raw/master/imgs/202506031425791.png)

### 5.2 GRU的结构

#### 5.2.1 更新门（Update Gate）

- 决定保留多少历史信息，综合替代LSTM的输入/遗忘门。

#### 5.2.2 重置门（Reset Gate）

- 控制历史信息的过滤强度。

#### 5.2.3 新的记忆内容（New memory content）

- 计算新的候选隐藏状态，可能会与当前隐藏状态结合。

### 5.3 GRU实现示例

```python
# GRU的PyTorch实现
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h_0):
        out, h_n = self.gru(x, h_0) # 运用GRU层
        out = self.fc(out) # 运用全连接层
        return out
```



### 5.4 GRU优势

- **参数减少约33% ** → 训练速度提升20~30%
- **内存占用降低**→ 更适合移动端部署
- **收敛速度更快** → 在短文本任务中表现更优



## 第六章 GRU与LSTM对比

### 6.1 核心组件对比

| 组件      | LSTM            | GRU             | 功能本质            |
| ------- | --------------- | --------------- | --------------- |
| 输入门     | ✅ (Input Gate)  | ❌               | 控制新信息写入         |
| 遗忘门     | ✅ (Forget Gate) | ❌               | 控制旧信息遗忘         |
| 输出门     | ✅ (Output Gate) | ❌               | 控制信息输出          |
| **重置门** | ❌               | ✅ (Reset Gate)  | 控制历史信息的筛选       |
| **更新门** | ❌               | ✅ (Update Gate) | 综合替代LSTM的输入/遗忘门 |
| 记忆单元    | ✅ (Cell State)  | ❌               | 长期记忆存储          |
| 参数数量    | 约4×(n²+nm)      | 约3×(n²+nm)      | m:输入维度, n:隐藏层维度 |

### 6.2 关键区别

| 特性     | GRU          | LSTM             |
| ------ | ------------ | ---------------- |
| 门控数量   | 2个门（更新门、重置门） | 3个门（输入门、遗忘门、输出门） |
| 记忆单元   | 无单独记忆单元      | 有独立Cell State    |
| 参数数量   | 约减少33%       | 参数较多             |
| 计算速度   | 快20-30%      | 相对较慢             |
| 长序列表现  | 适合中等长度序列     | 更适合超长序列          |
| 初始化敏感性 | 对初始化较敏感      | 相对稳定             |

### 6.2 对比决策树

![](https://gitee.com/YunboCheng/image-bad/raw/master/imgs/202506031446524.png)



## 第七章 循环神经网络的变体（Bi-RNN）

### 7.1 双向循环神经网络（Bi-RNN）

**双线循环神经网络（Bidirectional RNN，Bi-RNN**是一种能够同时利用序列数据过去（Past）和未来（Future）信息的循环神经网络。它通过组合两个独立的RNN（一个正向处理序列，一个反向处理序列），增强模型上下文的理解能力。

- **核心思想** ：每个时间步的输入，既能参考历史信息，也能参考未来信息。
- **典型应用** ：自然语言处理（如机器翻译、命令实体识别）、语音识别、时间序列预测等需要上下文任务。

![](https://gitee.com/YunboCheng/image-bad/raw/master/imgs/202506041005499.png)

### 7.2 核心组件

Bi-RNN由三部分组成：

1. **前向RNN层（Forward Layer）** :
   - 按时间顺序$(t=1→ T)$处理输入序列，生成隐藏状态$ \vec{h_t}$
   - 公式：$\vec{h_{t}}=RNN(x_{t},\overrightarrow{h_{t-1}})$

2. **后向RNN层（Backward Layer）** :
   - 按时间逆序$(t=T →1)$处理输入序列，生成隐藏状态$\overleftarrow{h_{t}}$
   - 公式：$\overleftarrow{h_{t}}=RNN(x_{t},\overleftarrow{h_{t+1}})$

3. **输出组合层** ：
   - 将前向和后向的隐藏状态组合，得到最终输出$h_{t}$。
   - 常见组合方式：
     - **拼接（Concatenate）**: $h_{t}=[\overrightarrow{h_{t}};\overleftarrow{h}]$
     - **相加（Sum）**: $h_{t}=\overrightarrow{h_{t}}+\overleftarrow{h_{t}}$
     - **平均（Average）**:  $h_{t}=\frac{\overrightarrow{h_{t}} + \overleftarrow{h_{t}}}{2}$

   | **组合方式** | **公式**                                   | **特点**                             |
   | -------- | ---------------------------------------- | ---------------------------------- |
   | 拼接       | $h_{t}=[\overrightarrow{h_{t}};\overleftarrow{h}]$ | 保留全部信息，维度翻倍（最常用，适合后续复杂任务如分类、序列标注）。 |
   | 相加       | $h_{t}=\overrightarrow{h_{t}}+\overleftarrow{h_{t}}$ | 维度不变，但可能丢失部分特征（适合轻量级任务）。           |
   | 平均       | $h_{t}=\frac{\overrightarrow{h_{t}} + \overleftarrow{h_{t}}}{2}$ | 维度不变，缓解梯度问题（适合低资源场景）。              |


重点讲解一下拼接方式：

- 符号表示 ：$h_{t}=[\overrightarrow{h_{t}};\overleftarrow{h}]$

  - $\overrightarrow{h_{t}}$ ：前向RNN在时间步 $t$ 的隐藏状态（如维度128）。
  - $\overleftarrow{h_t}$ ：后向RNN在时间步 𝑡t 的隐藏状态（如维度128）。
  - **分号 ;** 表示在**特征维度（向量维度）** 上进行拼接，生成一个更长的向量。

- 如何拼接？

  假设每个方向的隐藏状态维度为 $d$（如128维）：

  - **前向隐藏状态**：$\overrightarrow{h_{t}} \in \mathbb{R}^d$

  - **后向隐藏状态** ：$\overleftarrow{h_{t}} \in \mathbb{R}^d$

  - **拼接后的结果** ： 
    $$
    h_{t} = [\overrightarrow{h_{t}};\overleftarrow{h_{t}}]  \in \mathbb{R}^d 
    $$
    即，将两个向量的首尾相连，形成一个$2d$维的向量（如128+128=256维）。

  示意图：

  以时间步$t=2$为例（假设$d=2$）:

  ```markdown
  前向隐藏状态 →h₂ = [0.8, 0.2]  
  后向隐藏状态 ←h₂ = [0.5, 0.7]  
  拼接结果 h₂ = [0.8, 0.2, 0.5, 0.7] 
  ```

- 拼接的意义？

  - **保留双向信息** ： 前向状态捕捉历史信息（如句子中前文），后向状态捕捉未来信息（如后文），拼接后同时保留两者特征。
  - **增强表达能力** ：拼接后的向量维度翻倍，可输入到后续的全连接层或解码器中，提升模型对上下文的建模能力，使分析得到的结果更加准确。

  ​

### 7.3 为什么需要Bi-RNN

- 普通RNN的局限性：只能利用历史信息（如处理句子时，仅知道当前词之前的内容）。

  **示例**：预测句子 `"The __ is swimming in the pool"` 中的空白词，若只看到前面的 "The"，可能无法确定是 "boy" 还是 "fish"；但结合后面的 "swimming"，Bi-RNN可推断出更准确的答案（如 "fish"）。



### 7.4 Bi-RNN优缺点

| **优点**               | **缺点**               |
| -------------------- | -------------------- |
| 1. 捕获双向上下文信息，提升模型表现。 | 1. 计算成本高（两倍参数和计算量）。  |
| 2. 解决普通RNN的单向信息局限。   | 2. 需完整输入序列，无法实时流式处理。 |
| 3. 灵活适配不同任务（如拼接、相加）。 | 3. 长序列训练时梯度传播可能不稳定。  |



### 7.5 实战示例：命名体识别（NER）

- **输入句子**：`"Apple is based in Cupertino."`


- **Bi-RNN** 处理 ：
  - 前向RNN在 `"Apple"` 处捕捉到“可能是一个公司名”。
  - 后向RNN在 `"Apple"` 处结合后文 `"based in Cupertino"`，确认“苹果公司”。
  - **拼接结果**：将前向和后向的特征组合，输入分类层，输出实体标签 `ORG`（组织机构）。



### 7.6 代码实现

```python
# 使用PyTorch
import torch
import torch.nn as nn

# 定义双向LSTM（拼接方式）
bilstm = nn.LSTM(
    input_size=100,  # 输入特征维度
    hidden_size=128, # 每个方向的隐藏状态维度
    bidirectional=True, # 启用双向
    batch_first=True
)

# 输入数据（batch_size=2，序列长度=5，特征维度=100）
inputs = torch.randn(2, 5, 100)

# 前向传播
outputs, (h_n, c_n) = bilstm(inputs)  # outputs形状：(2,5,256)

# 验证拼接维度
print(outputs.shape)  # torch.Size([2, 5, 256])（256 = 128*2）

```



## 第八章 RNN和CNN的对比

### 8.1 核心定位

- 卷积神经网络（CNN）和循环神经网络（RNN）都是深度学习框架中常用的网络结构，但它们的设计目标、适用场景和底层原理有显著差异。

| **网络类型**         | **本质功能**                 | **典型框架支持**                 |
| ---------------- | ------------------------ | -------------------------- |
| **卷积神经网络 (CNN)** | 处理 **空间相关性数据**（如图像、网格结构） | TensorFlow、PyTorch、Keras 等 |
| **循环神经网络 (RNN)** | 处理 **序列相关性数据**（如文本、时间序列） | TensorFlow、PyTorch、MXNet 等 |

- **本质区别** ：
  - CNN ➔ 空间特征提（Where的问题）
  - RNN ➔ 时序依赖建模（When的问题）

### 8.2 核心原理

**（1）卷积神经网络（CNN）**

- **核心操作** ：卷积核滑动计算（提取局部特征）

- **结构特点** ：

  - 卷积层（Convolutional Layer）
  - 池化层（Pooling Layer）
  - 全连接层（Fully Connected Layer）

- **数学表达** ：
  $$
  Output=f(W*X+b)
  $$

  - $*$ : 卷积运算
  - $W$ ：卷积核权重
  - $X$ ：输入特征图

**（2）循环神经网络（RNN）**

- **核心操作** ：时间步递归计算（保留时序信息）

- **结构特征** ：

  - 隐藏状态（Hidden State）
  - 门控机制（LSTM/GRU变体）

- **数学表达** ：
  $$
  h_{t}=f(W_{hh}h_{t}+W_{xh}x_{t}+b)
  $$

  - $h_{t}$ : 当前时刻隐藏状态
  - $x_{t}$ : 当前时刻输入

### 8.3 典型应用场景

| **CNN 适用领域** | **RNN 适用领域**   |
| ------------ | -------------- |
| 图像分类（ResNet） | 机器翻译（Seq2Seq）  |
| 目标检测（YOLO）   | 文本生成（LSTM）     |
| 图像分割（U-Net）  | 股票预测（时间序列分析）   |
| 医学影像分析       | 语音识别（CTC Loss） |

### 8.4 在PyTorch中的实现差异

**CNN示例（图像分类）**

```python
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 声明卷积网络层
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        # 声明池化层
        self.pool = nn.MaxPool2d(2)
        # 声明全连接层
        self.fc = nn.Linear(16*13*13, 10)

    def forward(self, x):
        # 构建卷积神经网络
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)
```

**RNN示例（文本处理）**

```python
import torch.nn as nn
import torch

class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        # 声明词嵌入模型
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # 声明循环网络层
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        # 声明全连接层
        self.fc = nn.Linear(hidden_dim, 1)  # 二分类任务

    def forward(self, x):
        # 词嵌入
        x = self.embed(x)  # (batch, seq_len, embed_dim)
        # 构建循环神经网络
        _, h_n = self.rnn(x)  # h_n: (1, batch, hidden_dim)
        # Softmax分类器（归一化）：将得分值转换为概率值
        return torch.sigmoid(self.fc(h_n.squeeze(0)))
```

