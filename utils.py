import logging
import os
import torch
import numpy as np
import random

k_list = [10, 50, 100]
PAD = 0
EOS = 1
INF = 1e8


def init_seeds(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


def get_previous_user_mask(seq, user_size):
    assert seq.dim() == 2
    device = seq.device
    prev_shape = (seq.size(0), seq.size(1), seq.size(1))
    seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
    previous_mask = np.tril(np.ones(prev_shape)).astype("float32")
    previous_mask = torch.from_numpy(previous_mask).to(device)
    masked_seq = previous_mask * seqs.data.float()
    PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1).to(device)
    masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)
    ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size).to(device)
    masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float(-1000))
    return masked_seq


def apk(actual, predicted, k=10):

    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    score = score / min(len(actual), k)
    return score


def compute_metric(y_prob, y_true, k_list=[10, 50, 100]):

    PAD = 0
    scores_len = 0
    y_prob = np.array(y_prob)
    y_true = np.array(y_true)

    scores = {"hits@" + str(k): [] for k in k_list}
    scores.update({"map@" + str(k): [] for k in k_list})

    for p_, y_ in zip(y_prob, y_true):
        if y_ != PAD:
            scores_len += 1.0
            p_sort = p_.argsort()
            for k in k_list:
                topk = p_sort[-k:][::-1]
                scores["hits@" + str(k)].extend([1.0 if y_ in topk else 0.0])
                scores["map@" + str(k)].extend([apk([y_], topk, k)])

    scores = {k: np.mean(v) for k, v in scores.items()}

    return scores, scores_len


def print_scores(scores):
    res_str = ""
    for k in [10, 50, 100]:
        res_str += "H@{k} {hits:.2%}\t".format(k=k, hits=scores[f"hits@{k}"])
    logging.info(res_str)
    res_str = ""
    for k in [10, 50, 100]:
        res_str += "M@{k} {map:.2%}\t".format(k=k, map=scores[f"map@{k}"])
    logging.info(res_str)
