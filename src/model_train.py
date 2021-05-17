import torch
from torch import nn

from data_loader import get_lw_of_train_sample, MAX_TRAIN_INDEX
from model import device, NewsModel, Config, get_trained_net, get_epoch, save_net_epoch
from word_vector_tool import get_vector_model, get_vector_weight


# flag : True 从已有模型继续训练, False: 新建模型开始训练
# epoch_size : 训练多少轮
def train(flag, epoch_size):
    if flag:
        net = get_trained_net()
        epoch = get_epoch()
    else:
        net = NewsModel(Config).to(device)
        net.init_embedding(get_vector_weight(get_vector_model()))
        epoch = 0
    # 我之前用的一直是 Adam，后面改成了 SGD, lr是学习率，可以理解成每一步优化的幅度，一开始可以较大,后面要逐渐变小
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)
    # optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
    # CrossEntropyLoss 交叉熵损失,F.cross_entropy()也是，但前者会先进行softmax运算,后者不会,
    # 所以在我定义的模型最后没有显式使用return F.SoftMax(x)，而是直接返回全连接的结果
    # 如果在模型部分使用了softmax,那么此时应换成nn.NLLLoss()
    criterion = nn.CrossEntropyLoss().to(device)
    net.train()
    
    #训练也就是调参的过程，在卷积层,全连接层都有许多参数,训练的过程就是调整这些参数使得结果与真实值的差距逐渐缩小
    print("Start Training...")
    for epoch_index in range(epoch, epoch + epoch_size):
        epoch += 1
        for sample_index in range(0, MAX_TRAIN_INDEX + 1):
            labels, sentences = get_lw_of_train_sample(sample_index)  # [32],[32][32]
            labels, sentences = torch.tensor(labels).to(device), torch.tensor(sentences).to(device)
            outputs = net(sentences)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            print('[epoch %d] [sample %d] loss: %.3f' %
                  (epoch, sample_index + 1, loss_value))

    save_net_epoch(net, epoch)
    print('Stop Training...')


if __name__ == '__main__':
    train(True, 4)
