# 重点：启动该项目需要配置启动参数 --model TextRNN

import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
# 解析命令行参数
import argparse
# tensorFlow提供的可视化工具
from tensorboardX import SummaryWriter

# 说明一下几种文件格式：
# pkl: Python Pickle 序列化文件的常见扩展名，用于将 Python 对象（如模型、数据集、配置等）以二进制格式保存到本地，便于后续快速加载复用。
# npz: NumPy 专用的压缩存档文件格式，用于高效存储多个 NumPy 数组（如训练数据、标签、权重矩阵等），内部通过 ZIP 压缩技术打包多个 .npy 文件，节省存储空间且方便批量管理。
# ckpt: 机器学习框架（如 TensorFlow、PyTorch）中用于保存模型训练状态的通用文件格式，包含模型权重、优化器参数、训练进度等完整信息，便于从断点恢复训练或部署模型。


# 使用Python中的argparse模块定义并解析命令行参数
# 创建一个ArgumentParser对象，用于处理命令行参数。description：程序的描述信息
parser = argparse.ArgumentParser(description='Chinese Text Classification')
# --model（模型选择参数）
# 作用：指定要使用的模型
# 参数解释：type=str：参数值必须为字符串，required=True：该参数是必填项，运行时脚本必须提供，help：参数帮助信息（通过--help显示）
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
# --embedding（词向量化初始化方式）
# 作用：选择词向量的初始化方式
# 参数解释：default='pre_trained'：默认使用预训练（pre_trained）的词向量
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
# --word（分词粒度）
# 作用：控制分词粒度是基于词（word）还是字符（char）
# 参数解释：default=False: 默认使用字符（char）级分词
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
# 解析命令行输入的参数，并将结果存储在args对象中。
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  #TextCNN, TextRNN,
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    else:
        from utils import build_dataset, build_iterator, get_time_dif

    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    # 随机数种子（每次都生成一样的权重、偏置数据），这样做的目的是为了测试模型参数对预测结果的影响
    # 比如：模型参数A和模型参数B存在一些参数差异，如果这两个模型在训练时权重等数据不一致，那么最终将不能分析出到底是什么原因导致的预测结果不一致
    #      必须将权重等数据固定，测试模型参数对训练结果的影响。
    # 当然其中传递的参数不一定是1，是几都可以，但是必须保证这三个方法传递的值是一样的。
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    # 记录起始时间
    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    # 构建RNN网络模型
    model = x.Model(config).to(config.device)
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    if model_name != 'Transformer':
        # 权重参数随机初始化
        init_network(model)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter,writer)
