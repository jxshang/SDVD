import torch
import logging
from utils import compute_metric, PAD
from model.model import SDVD


def get_acc(pred, gold):
    gold = gold.contiguous().view(-1)
    pred = pred.contiguous().view(-1, pred.size(-1))
    pred = pred.max(1)[1]
    n_correct = pred.data.eq(gold.data)
    n_correct = n_correct.masked_select(gold.ne(PAD).data).sum().float()

    return n_correct


def train_cascade(model, data, static_graphs, dynamic_graph_list, optimizer, args):
    model.train()
    total_loss = 0.0
    n_total_words = 0.0
    n_total_correct = 0.0

    for i, batch in enumerate(data):
        torch.autograd.set_detect_anomaly(True)
        cas, timestamp, cas_index = batch
        gold = cas[:, 1:]
        n_words = gold.data.ne(PAD).sum().float()
        n_total_words += n_words
        optimizer.zero_grad()
        loss, pred = model(cas, timestamp, cas_index, static_graphs, dynamic_graph_list)
        n_correct = get_acc(pred, gold.to(pred.device))
        loss.backward()
        optimizer.step()
        n_total_correct += n_correct
        total_loss += loss.item()

    total_loss /= len(data)

    return total_loss, n_total_correct / n_total_words


def validation(model, data, static_graphs, dynamic_graph_list, k_list=[10, 50, 100]):
    model.eval()
    n_total_words = 0.0
    scores = {}
    num_total = 0.0
    for k in k_list:
        scores["hits@" + str(k)] = 0
        scores["map@" + str(k)] = 0

    with torch.no_grad():
        for i, batch in enumerate(data):
            cas, timestamp, cas_index = batch
            gold = cas[:, 1:]
            n_words = gold.data.ne(PAD).sum().float()
            n_total_words += n_words
            pred = model(
                cas,
                timestamp,
                cas_index,
                static_graphs,
                dynamic_graph_list,
            )
            y_gold = cas[:, 1:].contiguous().view(-1).detach().cpu().numpy()
            pred = pred.contiguous().view(-1, pred.size(-1))
            y_pred = pred.detach().cpu().numpy()
            scores_batch, scores_len = compute_metric(y_pred, y_gold, k_list)
            num_total += scores_len
            for k in k_list:
                scores["hits@" + str(k)] += scores_batch["hits@" + str(k)] * scores_len
                scores["map@" + str(k)] += scores_batch["map@" + str(k)] * scores_len

    for k in k_list:
        scores["hits@" + str(k)] = scores["hits@" + str(k)] / num_total
        scores["map@" + str(k)] = scores["map@" + str(k)] / num_total

    return scores
