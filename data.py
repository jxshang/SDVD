import os
import pickle
import torch
import random
import logging
import numpy as np
from utils import PAD, EOS


class Options(object):

    def __init__(self, data_name="twitter"):
        base_dir = os.path.join("data", data_name)
        self.data_name = data_name
        self.data = os.path.join(base_dir, "cascades.txt")
        self.net_data = os.path.join(base_dir, "edges.txt")
        self.u2idx_dict = os.path.join(base_dir, "u2idx.pickle")


def build_user_idx(options):
    user_set = set()
    u2idx = {}

    lineid = 0
    for line in open(options.data):
        lineid += 1
        if len(line.strip()) == 0:
            continue
        chunks = line.strip().split(",")
        for chunk in chunks:
            try:
                if len(chunk.split()) == 2:
                    user, timestamp = chunk.split()
                elif len(chunk.split()) == 3:
                    root, user, timestamp = chunk.split()
                    user_set.add(root)
            except:
                print(line)
                print(chunk)
                print(lineid)
            user_set.add(user)

    pos = 0
    u2idx["<blank>"] = pos
    pos += 1
    u2idx["</s>"] = pos
    pos += 1

    for user in user_set:
        u2idx[user] = pos
        pos += 1

    with open(options.u2idx_dict, "wb") as handle:
        pickle.dump(u2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return u2idx


class DataLoader(object):

    def __init__(self, data, batch_size, need_shuffle=True):
        self.cas = data[0]
        self.time = data[1]
        self.idx = data[2]
        self.need_shuffle = need_shuffle
        self._batch_size = batch_size
        self._n_batch = int(np.ceil(len(self.cas) / self._batch_size))
        self._iter_count = 0
        if self.need_shuffle:
            self.shuffle_dataset()

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def shuffle_dataset(self):
        num = [x for x in range(0, len(self.cas))]
        random_seed_int = random.randint(0, 1000)
        random.seed(random_seed_int)
        random.shuffle(num)
        self.cas = [self.cas[num[i]] for i in range(0, len(num))]
        self.time = [self.time[num[i]] for i in range(0, len(num))]
        self.idx = [self.idx[num[i]] for i in range(0, len(num))]

    def next(self):

        def pad_to_longest(insts):
            # TODO 在split_data里按最大长度截取级联
            # max_len = 200
            max_len = min(200, max(len(inst) for inst in insts))
            inst_data = np.array(
                [
                    (
                        inst + [PAD] * (max_len - len(inst))
                        if len(inst) < max_len
                        else inst[:max_len]
                    )
                    for inst in insts
                ]
            )
            inst_data_tensor = torch.LongTensor(inst_data)
            return inst_data_tensor

        if self._iter_count < self._n_batch:
            batch_idx = self._iter_count
            self._iter_count += 1
            start_idx = batch_idx * self._batch_size
            end_idx = (batch_idx + 1) * self._batch_size
            seq_data = pad_to_longest(self.cas[start_idx:end_idx])
            time_data = pad_to_longest(self.time[start_idx:end_idx])
            id_data = torch.tensor(self.idx[start_idx:end_idx], dtype=torch.int64)

            return seq_data, time_data, id_data
        else:
            if self.need_shuffle:
                self.shuffle_dataset()
            self._iter_count = 0
            raise StopIteration()


def split_data(
    data_name,
    train_rate=0.8,
    valid_rate=0.1,
    max_len=200,
    build_dict=True,
    with_EOS=True,
):
    options = Options(data_name)
    u2idx: dict

    if not build_dict and os.path.exists(options.u2idx_dict):
        with open(options.u2idx_dict, "rb") as file:
            u2idx = pickle.load(file)
    else:
        u2idx = build_user_idx(options)

    t_cascades = []
    timestamps = []
    for line in open(options.data):
        u_in_cas = set()
        if len(line.strip()) == 0:
            continue
        timestamplist = []
        userlist = []
        chunks = line.strip().split(",")
        for chunk in chunks:
            if len(chunk.split()) == 2:
                user, timestamp = chunk.split()
            elif len(chunk.split()) == 3:
                root, user, timestamp = chunk.split()
                if root in u2idx:
                    u_in_cas.add(root)
                    userlist.append(u2idx[root])
                    timestamplist.append(float(timestamp))
            if user in u2idx:
                if user not in u_in_cas:
                    u_in_cas.add(user)
                    userlist.append(u2idx[user])
                    timestamplist.append(float(timestamp))

        if max_len < len(userlist) <= 500:
            userlist = userlist[:max_len]
            timestamplist = timestamplist[:max_len]

        if 2 <= len(userlist) <= max_len:
            if with_EOS:
                userlist.append(EOS)
                timestamplist.append(EOS)
            t_cascades.append(userlist)
            timestamps.append(timestamplist)

    """ordered by timestamps"""
    order = [i[0] for i in sorted(enumerate(timestamps), key=lambda x: x[1])]
    timestamps = sorted(timestamps)
    t_cascades[:] = [t_cascades[i] for i in order]
    cas_idx = [i for i in range(1, len(t_cascades) + 1)]

    """data split"""
    train_idx_ = int(train_rate * len(t_cascades))
    train = t_cascades[0:train_idx_]
    train_t = timestamps[0:train_idx_]
    train_idx = cas_idx[0:train_idx_]
    train = [train, train_t, train_idx]

    valid_idx_ = int((train_rate + valid_rate) * len(t_cascades))
    valid = t_cascades[train_idx_:valid_idx_]
    valid_t = timestamps[train_idx_:valid_idx_]
    valid_idx = cas_idx[train_idx_:valid_idx_]
    valid = [valid, valid_t, valid_idx]

    test = t_cascades[valid_idx_:]
    test_t = timestamps[valid_idx_:]
    test_idx = cas_idx[valid_idx_:]
    test = [test, test_t, test_idx]

    user_size = len(u2idx)
    total_len = sum(len(i) - 1 for i in t_cascades)
    logging.info("Data Information:")
    logging.info(f" - Datasets: {data_name}")
    logging.info(" - Total number of users in cascades: %d" % (user_size - 2))
    logging.info(
        f" - Total size: {len(t_cascades)}, Train size: {len(train[0])}, Valid size: {len(valid[0])}, Test size: {len(test[0])}."
    )
    ave_l, max_l = total_len / len(t_cascades), max(len(cas) for cas in t_cascades)
    min_l = min(len(cas) for cas in t_cascades)
    logging.info(
        " - Average length: {:.2f}, Maximum length: {:.2f}, Minimum length: {:.2f}".format(
            ave_l, max_l, min_l
        )
    )

    return user_size, t_cascades, timestamps, train, valid, test


def build_social_graph(data_name):
    opts = Options(data_name)
    edges_list = []

    with open(opts.u2idx_dict, "rb") as f:
        u2idx = pickle.load(f)

    with open(opts.net_data) as f:
        relation_list = f.read().strip().split("\n")
        relation_list = [edge.split(",") for edge in relation_list]

        relation_list = [
            (u2idx[edge[0]], u2idx[edge[1]])
            for edge in relation_list
            if edge[0] in u2idx and edge[1] in u2idx
        ]
        relation_list_reverse = [edge[::-1] for edge in relation_list]
        edges_list += relation_list_reverse

    edge_index = torch.LongTensor(edges_list).t()

    return edge_index


def build_diffusion_graph(cascades, window_size=1):
    u = []
    v = []
    for line in cascades:
        for i in range(len(line)):
            # TODO 要不要排除EOS？
            # if line[i] == EOS or line[i] == PAD:
            #     break
            if i == len(line) - 1 or line[i] == PAD:
                break
            for j in range(i + 1, min(len(line), i + window_size + 1)):
                if line[j] != PAD and line[j] != EOS:
                    u.append(line[i])
                    v.append(line[j])

    edge = torch.LongTensor([u, v])
    return edge


def build_hypergraph_edge_index(cascades, timestamps, n_interval=8):

    edges = []
    for i in range(len(cascades)):
        edges += [
            (cascades[i][j], i + 1, timestamps[i][j])
            for j in range(len(cascades[i]) - 1)
        ]
    edges = sorted(edges, key=lambda x: x[2])
    edge_index = torch.LongTensor([[e[0] for e in edges], [e[1] for e in edges]])

    split_length = len(edges) // n_interval
    graph_list = {}
    pre_index = 0
    for i in range(split_length, split_length * n_interval, split_length):
        graph_list[edges[i - 1][2]] = edge_index[:, pre_index:i]
        pre_index = i
    graph_list[edges[-1][2]] = edge_index[:, pre_index:]
    return graph_list


def build_diffusion_hypergraph(cascades):
    u = []
    v = []

    for i in range(len(cascades)):
        u += cascades[i][:-1]
        v += [i] * (len(cascades[i]) - 1)

    hyperedge_index = torch.tensor([u, v], dtype=torch.int64)

    return hyperedge_index
