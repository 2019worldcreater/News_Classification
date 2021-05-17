import os

import jieba
import jieba.analyse

# 如果直接写 /data/word_vector/cn_stopwords.txt
# 假设不同目录下的py脚本调用该文件下的方法，就会导致文件路径出错，因为实际上import是将代码嵌入到了调用者部分,此时的./data...不存在
module_path = os.path.dirname(__file__)
# 停用词
stop_words = module_path + '/data/word_vector/cn_stopwords.txt'
stopwords = {}.fromkeys([line.rstrip() for line in open(stop_words,
                                                        encoding='utf-8')])

# 提取关键词时允许的词性，具体: https://blog.csdn.net/hyfound/article/details/82700313
pos = ('n', 'nz', 'v', 'ns', 'vn', 'i', 'a', 'nt', 'b', 'vd', 't', 'ad', 'an', 'c', 'nr')
jieba.analyse.set_stop_words(stop_words)


# '字符串'->['词1',...]
def jieba_split(sentence):
    cut = jieba.cut(sentence)
    return list(cut)


# 将上面的['词1','词2'..] 去除其中的停用词返回
def remove_stop_words(words):
    result = []
    for word in words:
        if word not in stopwords:
            result.append(word)
    return result


# 上面的操作中去除停用词后,抽取其中num个关键词返回
def get_key_words(words, num):
    sentence = ''
    for word in words:
        sentence = sentence + ' ' + word
    key_words = jieba.analyse.extract_tags(sentence, topK=num, withWeight=False, allowPOS=pos)
    return list(key_words)


# ('很长的一句话',20)
# return ['key1','key2'...]
def get_key_words_all_step(sentence, num):
    return get_key_words(remove_stop_words(jieba_split(sentence)), num)
