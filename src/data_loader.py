import os

import torch

# 本文件主要是给训练、测试过程提供便捷的取样本操作， 相当于对数据的迭代

# test_data的分类及数量 {'娱乐': 1000, '财经': 1000, '房地产': 1000, '旅游': 1000, '科技': 1000, '体育': 1000, '健康': 1000, '教育': 1000,
# '汽车': 1000, '新闻': 1000, '文化': 1000, '女人': 1000}
# train_data的分类及数量 '娱乐': 1934, '财经': 1877, '房地产': 1872, '旅游': 1998, '科技': 1988, '体育': 1978, '健康': 2000, '教育': 1979,
# '汽车': 1996, '新闻': 1935, '文化': 1995, '女人': 1997


categories = ['娱乐', '财经', '房地产', '旅游', '科技', '体育', '健康', '教育', '汽车', '新闻', '文化', '女人']

# 类别名对应的id
labels = {'娱乐': 0, '财经': 1, '房地产': 2, '旅游': 3, '科技': 4, '体育': 5,
          '健康': 6, '教育': 7, '汽车': 8, '新闻': 9, '文化': 10, '女人': 11}

#sampple-?.pth的最大序号
MAX_TRAIN_INDEX = 235
MAX_TEST_INDEX = 119

module_path = os.path.dirname(__file__)


# 在./data/XXXsamples文件夹下，一个sample= [(),()....] 约100个()
# 一个 () = (类别id,[word1_index, word2_index...]), [word1_index..]为新闻分词后，每个词在word2vec模型中对应的索引值
# 类别id见labels中定义的变量

# trainSamples 是训练数据
# testSamples 测试，从未训练过

# 100个新闻样本, tuple(list,list)
# 第一个list 是100个样本对应的类别id, 后面的list则保存了每条新闻分词后每个词的索引,即[[新闻1的词索引..][新闻2..]]
# return ([label_1,label_2,..,label_100], [[news_1_word_1_index,..,news_1_word_50_index]...[news_100_word_1_index,..] )
def get_lw_of_train_sample(index):
    file = module_path + '/data/trainSamples/sample-' + str(index) + '.pth'
    return get_lw_sample(file)


def get_lw_of_test_sample(index):
    file = module_path + '/data/testSamples/sample-' + str(index) + '.pth'
    return get_lw_sample(file)


def get_lw_sample(file):
    sample = torch.load(file)
    labels = [l for l, w in sample]  # 提取label和words
    words = [w for l, w in sample]
    return labels, words


if __name__ == '__main__':
    # 可以看看具体的数据
    for i in range(0, MAX_TEST_INDEX + 1):
        l, w = get_lw_of_test_sample(i)
        print(l)
