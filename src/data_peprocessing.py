import random

import torch
from data_loader import labels
from jieba_tool import get_key_words_all_step
from word_vector_tool import file_to_sentences, add_sentences, get_vector_model, get_index


# test_data的分类及数量 {'娱乐': 1000, '财经': 1000, '房地产': 1000, '旅游': 1000, '科技': 1000, '体育': 1000, '健康': 1000, '教育': 1000,
# '汽车': 1000, '新闻': 1000, '文化': 1000, '女人': 1000}
# 语料集是txt文件，如果用windows自带的记事本打开，会卡死，建议用notepad++打开，相当流畅
# train_data的分类及数量 '娱乐': 1934, '财经': 1877, '房地产': 1872, '旅游': 1998, '科技': 1988, '体育': 1978, '健康': 2000, '教育': 1979,
# '汽车': 1996, '新闻': 1935, '文化': 1995, '女人': 1997


# 将原有的 pre.txt 打乱行写进新的shuffle_txt文件中
def shuffle_txt_data(pre_txt, shuffle_txt):
    # 用于将原有排序的数据打乱，方便后面训练
    encoding = 'utf-8'
    with open(pre_txt, 'r', encoding=encoding) as lines:
        # 也可以 for line in lines: line_list.append(line)
        line_list = list(lines)
        # 打乱列表顺序
        random.shuffle(line_list)
        with open(shuffle_txt, 'w', encoding=encoding) as file:
            for each_line in line_list:
                file.write(each_line)


# souhu_train.txt 文件中每一行分为分类和正文, 两者以 '\t' 分隔,test也是如此
# 首先先要训练语料集中的词汇，使用word2vec建立词向量模型
# 将分词的结果以每一行一个词，保存在 split_txt中，后面直接将将该文件训练词向量
# 这个过程很慢，2万行数据花了我20多分钟吧
def split_words(news_txt, split_txt):
    with open(news_txt, 'r', encoding='utf-8') as lines:
        with open(split_txt, 'a', encoding='utf-8') as file:
            for line in lines:
                split = line.split('\t')
                # label = split[0], 分类名
                content = split[1]
                # len(content)获取最大词汇,我想让词向量模型更全面些
                keys = get_key_words_all_step(content, len(content))
                for key in keys:
                    file.write(key + '\n')


# 原有词向量模型 model = get_vector_model()
# split_txt 分词文件，上面那个函数的结果
def train_vector(model, split_txt):
    # build_model(file_to_sentences(split_txt)) 当model没有时
    return add_sentences(model, file_to_sentences(split_txt))


# 先将每条新闻正文为 (label,[word1_index,word2_index,....]), label是labels词典中分类名对应的id,
# [....]则是正文内容分词后，每个词在词向量模型中的index，一定要先split_word,后train_vector
# 分词索引[..]长度固定为 50,,不足补0,如果不固定，[[49][50]....] 这种形状 torch.tensor()会出错
def package_news_data(news_txt, directory):
    sample_size = 100  # 每100个新闻为一个sample
    key_size = 50  # 关键词数量
    model = get_vector_model()  # 词向量模型
    with open(news_txt, 'r', encoding='utf-8') as lines:
        sample_index = 0  # sample_序号
        news_count = 0  # sample内的新闻数量
        sample = []  # 保存sample_size个新闻
        for line in lines:
            sp = line.split('\t')
            label = labels[sp[0]]
            content = sp[1]
            # 分词
            content_key = get_key_words_all_step(content, key_size)
            key_index = [get_index(model, key) for key in content_key]
            key_index.extend([0] * (key_size - len(key_index)))  # 补0
            # 加入样本集
            sample.append((label, key_index))
            news_count += 1
            print('[sample %d ][ news %d ]' % (sample_index, news_count))
            if news_count == sample_size:
                torch.save(sample, directory + 'sample-' + str(sample_index) + '.pth')
                print('[sample %d package done]' % sample_index)
                news_count = 0
                sample.clear()
                sample_index += 1
        if news_count > 0:  # 未满sample_size
            torch.save(sample, directory + 'sample-' + str(sample_index) + '.pth')
            print('[sample %d package done]' % sample_index)


if __name__ == '__main__':
    # pre_train_txt = './data/souhu_train.txt'
    # after_train_txt = './data/souhu_train_shuffle.txt'
    # pre_test_txt = './data/souhu_test.txt'
    # after_test_txt = './data/souhu_test_shuffle.txt'
    # words = './data/word_vector/all_words.txt'
    # shuffle_txt_data(pre_train_txt, after_train_txt)
    # shuffle_txt_data(pre_test_txt, after_test_txt)
    # split_words(pre_train_txt, words)
    # split_words(pre_train_txt, words)
    # model_save = get_vector_model()
    # train_vector(model_save, words)
    # package_news_data(after_train_txt, './data/trainSamples/')
    # package_news_data(after_test_txt, './data/testSamples/')
    print('done')
