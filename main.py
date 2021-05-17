import sys

import torch
import torch.nn.functional as F

import src.model as m
from src.data_loader import categories
from src.jieba_tool import get_key_words_all_step
from src.word_vector_tool import get_vector_model, get_index

# 调用不同文件夹下的脚本要加这个
sys.path.append('src/')

# 需要预测的新闻内容粘贴在content.txt中, 百度新闻还不错
if __name__ == '__main__':
    content = ''
    with open('content.txt', encoding='utf-8') as lines:
        for line in lines:
            content += line.lstrip().replace('\n', '')
    keys = get_key_words_all_step(content, len(content))
    chosen = []
    vector = get_vector_model()
    print('开始分词....')
    for key in keys:
        index = get_index(vector, key)
        if index > 0:
            print(key, end=', ')
            chosen.append(index)
            if len(chosen) == 50:
                break
    print('\n\n分词结束')
    x = torch.tensor([chosen]).to(m.device)
    print('load model...')
    net = m.get_trained_net()
    out = F.softmax(net(x), dim=1)
    _, pre = torch.max(out, dim=1)
    index = pre.item()
    print('类别：', categories[index], '\n概率: ', out[0][index].item())
