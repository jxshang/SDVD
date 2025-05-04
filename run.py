import argparse
import logging
import torch
import os
from data import (
    split_data,
    DataLoader,
    build_social_graph,
    build_diffusion_graph,
    build_hypergraph_edge_index,
    build_diffusion_hypergraph,
)
from epochs import train_cascade, validation
from model.model import SDVD
from utils import print_scores, init_seeds

data_names = [
    "christianity",
    "android",
    "douban",
    "twitter",
]
parser = argparse.ArgumentParser()
parser.add_argument("--data_name", default=data_names[0])
parser.add_argument("--log_prefix", default="test1")
parser.add_argument("--seed", type=int, default=2023, help="set random seed.")
parser.add_argument("--epoch", type=int, default=200)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--hidden_dim", type=int, default=64)
parser.add_argument("--n_heads", type=int, default=8)
parser.add_argument("--n_interval", type=int, default=8)
parser.add_argument("--drop_prob", type=float, default=0.3)
parser.add_argument("--graph_drop_prob", type=float, default=0.3)
parser.add_argument("--train_rate", type=float, default=0.8)
parser.add_argument("--valid_rate", type=float, default=0.1)
parser.add_argument("--patience", type=int, default=30)
parser.add_argument("--gpu_idx", type=int, default=1)
parser.add_argument("--save_model", action="store_true")


args = parser.parse_args()
args.save_path = f"checkpoint/{args.log_prefix}-{args.data_name}.pt"
device = torch.device(f"cuda:{args.gpu_idx}" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
args.device = device

if not os.path.exists("log"):
    os.mkdir("log")

logging.basicConfig(
    level=logging.INFO,
    filename=f"log/{args.log_prefix}-{args.data_name}.log",
    filemode="w",
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.info(f"device: {device}")
logging.info(args)

k_list = [10, 50, 100]


def run():
    init_seeds(args.seed)

    # ========= Preparing Dataset =========#
    user_size, all_cascades, all_timestamps, train, valid, test = split_data(
        args.data_name, args.train_rate, args.valid_rate
    )
    train_data = DataLoader(train, batch_size=args.batch_size, need_shuffle=True)
    valid_data = DataLoader(valid, batch_size=args.batch_size, need_shuffle=False)
    test_data = DataLoader(test, batch_size=args.batch_size, need_shuffle=False)

    args.cas_size = len(all_cascades) + 1
    args.user_size = user_size
    social_graph = build_social_graph(args.data_name)
    diffusion_graph = build_diffusion_graph(train[0])
    diffusion_hypergraph = build_diffusion_hypergraph(train[0])
    static_graphs = [social_graph, diffusion_graph, diffusion_hypergraph]
    root_users = torch.tensor([0] + [cas[0] for cas in all_cascades])
    dynamic_graph_list = build_hypergraph_edge_index(
        cascades=train[0], timestamps=train[1], n_interval=args.n_interval
    )
    dynamic_graph_list = [dynamic_graph_list, root_users]

    test_dynamic_graph_list = build_hypergraph_edge_index(
        cascades=all_cascades, timestamps=all_timestamps, n_interval=args.n_interval
    )
    test_dynamic_graph_list = [test_dynamic_graph_list, root_users]

    # ========= Preparing Model ========= #
    model = SDVD(args).to(device)
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(
        params, lr=args.learning_rate, betas=(0.9, 0.98), weight_decay=1e-5, eps=1e-09
    )

    validation_history = 0.0
    # ========= Train Cascade ========= #
    logging.info(f" ========= Train Cascade ========= ")
    best_scores = {}
    for epoch_i in range(args.epoch):
        logging.info(f"[ Epoch {epoch_i+1}/{args.epoch}]")
        train_loss, train_acc = train_cascade(
            model, train_data, static_graphs, dynamic_graph_list, optimizer, args
        )
        logging.info("Train Loss: {}, Accuracy: {:.2%}".format(train_loss, train_acc))

        scores = validation(
            model,
            valid_data,
            static_graphs,
            test_dynamic_graph_list,
        )
        logging.info("Validation Scores:")
        print_scores(scores)

        scores = validation(model, test_data, static_graphs, test_dynamic_graph_list)
        scores_sum = sum(scores.values())
        logging.info("Test Scores:")
        print_scores(scores)
        scores_sum = sum(scores.values())

        if validation_history <= scores_sum:
            validation_history = scores_sum
            best_scores = scores
            logging.info("Save best model!!")
            if args.save_model:
                torch.save(model.state_dict(), args.save_path)
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= args.patience:
            logging.info("Early Stopping!!")
            break

    logging.info("Best Scores:")
    print_scores(best_scores)


if __name__ == "__main__":
    run()
