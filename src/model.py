# coding:utf-8
import os

import torch
import torch.nn.functional as F
from torch import nn


def get_trained_net():
    net = torch.load(save_path.get('model')).to(device)
    net.eval()
    return net


def get_epoch():
    return torch.load(save_path.get('epoch'))


def save_net_epoch(net, epoch):
    net.eval()
    torch.save(net, save_path.get('model'))
    torch.save(epoch, save_path.get('epoch'))


# 模型和输入输出都会保存在这个device中
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# embedding_weight = get_vector_weight(get_vector_model())
# Config = {
#     # 'embedding_weight': get_vector_weight(model_save), 如果每次将这个config输入进去，之前的优化都失效了，只能让模型随机了
#     'kernel_size': (2, 3, 4),
#     'output_channels': 300,
#     'class_num': 12,
#     'linear_one': 250,
#     'linear_two': 120,
#     'dropout': 0.5,
#     'embedding_weight': embedding_weight
# }
module_path = os.path.dirname(__file__)

Config = {
    'kernel_size': (3, 4, 5),  # 卷积核的不同尺寸
    'output_channels': 200,  # 每种尺寸的卷积核有多少个
    'class_num': 12,  # 分类数量,见data_loader.categories
    'linear_one': 250,  # 第一个全连接层的输出节点数
    # 'linear_two': 120,
    'dropout': 0.5,  # 随机丢失节点占比
    'vocab_size': 283302,  # 词库大小，即词的数量, len(model.wv.index_to_key))是word2vec模型中的词库大小
    'vector_size': 100  # 每个词的词向量的长度， word = [.....]
    # 'embedding_weight': embedding_weight
}

# 模型保存路径
save_path = {
    'model': module_path + '/model_save/model.pth',
    'epoch': module_path + '/model_save/epoch.pth'
}


# in_channels 就是词向量的维度, out_channels则是卷积核(每个kernel_size都一样多)的数量
# 对于一种尺寸的卷积核  kernel_size = 3,in_channels=100,out_channels=50时,
# 设 x = [32][100]即一个新闻数据, 进行卷积操作前，先对x维度进行变换->[100][32]，即每一列是一个词向量，conv1d卷积层左右扫描即可
# x经过卷积操作后会得到[50][30], 分别对应50个卷积核分别对x运算得到的一维数据的结果,即[out_channels][in_channels-kernel_size + 1],
# 然后进行 relu运算，形状不变
# 紧接着进行最大池化操作,每个卷积核运算结果中选出一个最大值,即[30]中选出一个, -> max = [50][1]
# 接着将max转为一维数据 ->[50], 即[out_channels]
# 由于有len(kernel_size)种尺寸的卷积核,所以经过卷积层，池化层有 len(kernel_size) * output_channels 个输出
class NewsModel(nn.Module):
    def __init__(self, config):
        super(NewsModel, self).__init__()
        self.kernel_size = config.get('kernel_size')
        self.output_channels = config.get('output_channels')
        self.class_num = config.get('class_num')
        self.liner_one = config.get('linear_one')
        # self.liner_two = config.get('linear_two')
        self.dropout = config.get('dropout')
        self.vocab_size = config.get('vocab_size')
        self.vector_size = config.get('vector_size')

        # 也可以这么写，不用调用 init_embedding方法，利用已有的word2vec预训练模型中初始化
        # self.embedding = nn.Embedding.from_pretrained(torch.tensor(config.get('embedding_weight')), freeze=False)
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.vector_size, padding_idx=0)
        # 这里是不同kernel_size的conv1d
        self.convs = [nn.Conv1d(in_channels=self.embedding.embedding_dim, out_channels=self.output_channels,
                                kernel_size=size, stride=1, padding=0).to(device)
                      for size in self.kernel_size]
        self.fc1 = nn.Linear(len(self.kernel_size) * self.output_channels, self.liner_one)
        # self.fc2 = nn.Linear(self.liner_one, self.liner_two)
        self.fc2 = nn.Linear(self.liner_one, self.class_num)
        self.dropout = nn.Dropout(self.dropout)

    # embedding_matrix就是word2vec的get_vector_weight
    def init_embedding(self, embedding_matrix):
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix).to(device))

    # x = torch.tensor([[word1_index, word2_index,..],[],[]])
    # forward这个函数定义了前向传播的运算
    def forward(self, x):
        x = self.embedding(x)  # 词索引经过词嵌入层转化为词向量, [word_index1,word_index2]->[[vector1][vector2]],
        x = x.permute(0, 2, 1)  # 将(news_num, words_num, vector_size)换为(news_num,vector_size,word_num),方便卷积层运算
        # 将所有经过卷积、池化的结果拼接在一起
        x = torch.cat([self.conv_and_pool(x, conv) for conv in self.convs], 1)
        # 展开,[news_num][..]
        x = x.view(-1, len(self.kernel_size) * self.output_channels)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    @staticmethod
    def conv_and_pool(x, conv):
        x = F.relu(conv(x))
        x = F.max_pool1d(x, x.size(2))  # 最大池化
        x = x.squeeze(2)  # 只保存2维
        return x
