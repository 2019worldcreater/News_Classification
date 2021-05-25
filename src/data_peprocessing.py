import random

import torch
from data_loader import labels
from jieba_tool import get_key_words_all_step
from word_vector_tool import file_to_sentences, add_sentences, get_vector_model, get_index

# 须知： 我的原始文件是文本文件Utf-8编码，文件中每一行分为分类和正文, 两者以 '\t' 分隔,即：   财经  很长一串新闻内容
# 如果你的数据集不是这个格式，可以自己实现，这个文件无非就三个4个步骤： 打乱，得到词库，训练词向量，打包


# 将原有的原始数据文件pre.txt 打乱行写进新的shuffle_txt文件中
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



# 函数作用： 对每一条新闻的正文提取关键词，将分词的结果以每一行一个词，保存在split_txt中，即词库文件
# 参数: news_txt 数据文件,打没打乱的都行
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


# 函数作用：基于词库文件，使用word2vec建立词向量模型
# 参数：model:原有词向量模型, word_vector_tool的get_vector_model()得到
# 参数 split_txt,上面那个函数的结果，词库文件
def train_vector(model, split_txt):
    # build_model(file_to_sentences(split_txt)) 当model没有时
    return add_sentences(model, file_to_sentences(split_txt))


# 先将每条新闻正文化为 (label,[word1_index,word2_index,....]), label是labels词典中分类名对应的id,
# [....]则是正文内容分词后，每个词在词向量模型中的index，一定要先split_word,后train_vector
# 分词索引[..]长度固定为 50,,不足补0,如果不固定，[[49][50]....] 这种形状 torch.tensor()会出错
# 参数：new_txt, 打乱后的数据文件
# 参数: directory，文件夹路径
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
