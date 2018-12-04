import numpy as np

import torch

def sequence_accuracy(words, scores, targets, bos_index, eos_index, pad_index):
    _, predict = torch.max(scores,1)

    mask = torch.ones(words.size())
    mask[words==bos_index] = 0
    mask[words==eos_index] = 0
    mask[words==pad_index] = 0
    total = mask.sum().type(torch.FloatTensor)
    correct = (predict == targets).type(torch.FloatTensor)
    masked_correct = mask * correct
    acc = masked_correct.sum().type(torch.FloatTensor) / total
    return acc


def oov_accuracy(words, scores, targets, unk_index):
    _, predictions = torch.max(scores, 1)
    data = [[pred, lab] for (word, pred, lab) in zip(words, predictions, targets)
                    if word == unk_index]
    if len(data) == 0:
        return None
    else:
        data = np.array(data)
        return np.mean(data[:,0] == data[:,1])