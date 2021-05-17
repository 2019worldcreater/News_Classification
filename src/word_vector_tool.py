import multiprocessing
import os

import numpy
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

module_path = os.path.dirname(__file__)

# 词向量模型
model_path = module_path + '/data/word_vector/words_vector.model'


# 得到的词向量是ndArray类型的,不是list, .tolist()

# vector_path = './data/all_words.txt'

# 获取word2vec词向量模型
def get_vector_model():
    return Word2Vec.load(model_path)


# # 实际上  model_save.wv 也是一样的，而且更快
# def get_vector():
#     return gensim.models.KeyedVectors.load_word2vec_format(vector_path, binary=False)


# 先获取 model = get_vector_model(), 然后 sentences = words_to_sentences(['word1','word2'..]))
# 或者 sentences = file_to_sentences(file_path)

# 往已有模型中添加词汇
def add_sentences(model, sentences):
    model.build_vocab(sentences, update=True)
    model.train(sentences, total_words=model.corpus_count, epochs=model.epochs)
    model_save(model)
    return model


# 没有模型，新建一个词向量模型
def build_model(sentences):
    model = Word2Vec(sentences=sentences, window=1, min_count=1,
                     workers=multiprocessing.cpu_count(), sg=1)
    model_save(model)
    return model


# ./data/all_words.txt
def file_to_sentences(file):
    return LineSentence(file)


# ['word1','word2'...]
def words_to_sentences(words):
    return [words]


# 获取词汇的词向量, 返回类型是 ndArray, 可以.tolist()转化为list
def get_vector_of_text(model, text):
    if has_text(model, text):
        return model.wv.get_vector(text)
    return get_empty_vector()


# 保存模型
def model_save(model):
    model.save(model_path)
    # 下面这个保存太占空间了
    # model_save.wv.save_word2vec_format(vector_path, binary=True)


# 空向量
def get_empty_vector():
    v = numpy.zeros(100, dtype=float, order='C')
    return v


# 词text在模型词库中的索引值
def get_index(model, text):
    if has_text(model, text):
        return model.wv.get_index(text)
    return 0


# 根据索引查找词
def get_text_of_index(model, index):
    # index_to_key 是list,key_to_index是dict
    return model.wv.index_to_key[index]


# 给embedding层用，返回的是Word2Vec的权重参数
def get_vector_weight(model):
    return model.wv.vectors


# 是否有这个词
def has_text(model, text):
    return model.wv.has_index_for(text)
