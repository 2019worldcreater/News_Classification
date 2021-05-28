import torch

from data_loader import MAX_TRAIN_INDEX, get_lw_of_train_sample, MAX_TEST_INDEX, get_lw_of_test_sample, categories
from model import get_trained_net, device


# sample_flag, True 用测试数据测试，False:用训练数据测试
def test(sample_flag):
    net = get_trained_net()
    total_acc = 0.0
    first_sample = 0
    if sample_flag:
        last_sample = MAX_TEST_INDEX + 1
    else:
        last_sample = MAX_TRAIN_INDEX + 1
    wrong_label = [0 for i in range(len(categories))]
    for sample_index in range(first_sample, last_sample):  # 每个epoch有10个batch
        each_sample_acc_count = 0
        if sample_flag:
            labels, words = get_lw_of_test_sample(sample_index)
        else:
            labels, words = get_lw_of_train_sample(sample_index)
         labels, words = torch.tensor(labels).to(device), torch.tensor(words).to(device)
        _, pre = torch.max(net(words).data, 1)
        for pre_index in range(labels.size(0)):
            if pre[pre_index] == labels[pre_index]:
                each_sample_acc_count += 1
            else:
                wrong_label[labels[pre_index].item()] += 1
        each_sample_acc = each_sample_acc_count / len(labels) * 100
        print('[ sample %d acc = %.3f ]' % (sample_index, each_sample_acc))
        total_acc += each_sample_acc
    print('[ total_acc = %.3f ]' % (total_acc / (last_sample - first_sample)))
    dict = {}
    for i in range(len(wrong_label)):
        dict[categories[i]] = wrong_label[i]
    print(dict)


if __name__ == '__main__':
    test(True)
