# coding=utf-8

from nltk.translate.bleu_score import sentence_bleu


def cal_blue_score(pred, label):
    '''
    :param pred: ndarray: [timestep, batch]
    :param label: bdarray: [timestep, batch]
    :return:
    '''

    T, B = pred.shape

    refs = []
    trans = []
    for i in range(B):
        trans.append(pred[:, i])
        refs.append([label[:, i]])
    return sentence_bleu(refs[0], trans[0])
