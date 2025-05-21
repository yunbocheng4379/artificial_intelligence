import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# torchvision自带一些训练数据集，引入数据集供我们使用
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import numpy as np

# 定义超参数
input_size = 28  #图像的总尺寸28*28
num_classes = 10  #标签的种类数
num_epochs = 3  #训练的总循环周期
batch_size = 64  #一个撮（批次）的大小，64张图片

# 训练集
train_dataset = datasets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

# 测试集
test_dataset = datasets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

# 构建batch数据
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # 输入大小 (1, 28, 28)
            nn.Conv2d(
                in_channels=1,  # 灰度图
                out_channels=16,  # 要得到几多少个特征图（卷积核个数）
                kernel_size=5,  # 卷积核大小
                stride=1,  # 步长
                padding=2,  # 如果希望卷积后大小跟原来一样，需要设置padding=(kernel_size-1)/2 if stride=1
            ),  # 输出的特征图为 (16, 28, 28)
            nn.ReLU(),  # relu层，使用非线性函数
            nn.MaxPool2d(kernel_size=2),  # 进行池化操作（2x2 区域）,采用“最大池化”方式， 输出结果为： (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # 下一个套餐的输入 (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # 输出 (32, 14, 14)
            nn.ReLU(),  # relu层
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 输出 (32, 7, 7)
        )

        self.conv3 = nn.Sequential(  # 下一个套餐的输入 (32, 7, 7)
            nn.Conv2d(32, 64, 5, 1, 2),  # 输出 (64, 7, 7)
            nn.ReLU(),  # relu层
        )

        # 作用：将卷积层提取的空间特征映射到分类结果
        # 注意：全连接层需要一维输入，因此需要将（64 * 7 * 7）的3D特征图战平为 64x7x7 = 3136的一维向量
        # 结果：计算出输入该图像在10中特征分类中的占比，寻找最大占比作为最终分类的结果
        self.out = nn.Linear(64 * 7 * 7, 10)

    # 输入x存在四维：batch * Channel * 高度 * 宽度
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = x.view(x.size(0), -1) 是 PyTorch 中用于 展平多维特征图 的操作，目的是将卷积层输出的多维数据转换为一维向量，以便输入全连接层进行分类。
        #   - 保持批量维度不变：通过 x.size(0) 获取当前批量的样本数（如 batch_size=32）。
        #   - 自动计算剩余维度：-1 表示 PyTorch 会自动计算该维度的大小，使得总元素数量不变。
        # 结果：将 (batch_size, 64, 7, 7) 的4D张量 → 转换为 (batch_size, 64*7*7) 的2D张量。
        # 例如：输入是（一批数据：batch_size=32， 经过卷积层后特征图维度为 (32, 64, 7, 7)）

        # 将卷积输出的 空间特征 转换为一维向量。
        # 输入值：(32, 64, 7, 7)
        # 输出值：(32, 3136)
        x = x.view(x.size(0), -1)

        # 将 3136 维特征映射到 10 维分类结果，输出未归一化的 logits
        # 输入值：(32, 3136)
        # 输出值：(32, 10)
        output = self.out(x)
        return output

def accuracy(predictions, labels):
    pred = torch.max(predictions.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)


# 实例化
net = CNN()
# 损失函数
criterion = nn.CrossEntropyLoss()
# 优化器
optimizer = optim.Adam(net.parameters(), lr=0.001)  # 定义优化器，普通的随机梯度下降算法

# 开始训练循环
for epoch in range(num_epochs):
    # 当前epoch的预测正确结果保存下来
    train_rights = []

    # batch_idx = 1,2,3... 用户记录当前执行到的数据集位置
    # data : 原始数据
    # target : 准确值
    for batch_idx, (data, target) in enumerate(train_loader):  # 针对容器中的每一个批进行循环
        # 指定模型为：训练模式
        net.train()
        # 调用定义的卷积神经网络进行数据训练
        output = net(data)
        # 计算损失率
        loss = criterion(output, target)
        # 梯度数据清零
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 参数更新（卷积核权重、偏置等参数）
        optimizer.step()
        # 调用计算准确率函数，计算预测正确的个数
        right = accuracy(output, target)
        # 存储预测正确的个数
        train_rights.append(right)

        if batch_idx % 100 == 0:
            # 执行模型为：验证模式
            net.eval()
            # 当前epoch的验证正确结果保存下来
            val_rights = []

            # 验证数据不需要进行参数更新过程来训练模型
            # 将验证数据传递到训练好的模型中直接进行执行，验证模型的预测能力
            for (data, target) in test_loader:
                output = net(data)
                right = accuracy(output, target)
                val_rights.append(right)

            # 准确率计算
            train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
            val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))

            print('当前epoch: {} [{}/{} ({:.0f}%)]\t损失: {:.6f}\t训练集准确率: {:.2f}%\t测试集正确率: {:.2f}%'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                loss.data,
                       100. * train_r[0].numpy() / train_r[1],
                       100. * val_r[0].numpy() / val_r[1]))


