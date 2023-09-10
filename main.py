import os
import json
import random
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

from arg_parser import arg_parser
from cifar10_resnet import resnet32, resnet56, mnist_resnet18
from data_loader import data_loader
from train import train
from text_net import TextClassificationNet
from hinge_loss import MultiClassHingeLoss

def main():
    args = arg_parser()

    dist.init_process_group(backend='nccl',
                            init_method=args.init_method,
                            world_size=args.world_size,
                            rank=args.rank, timeout=timedelta(hours=6))
    group = dist.new_group(range(args.world_size))

    torch.cuda.set_device(args.gpu_id)
    print(f"| Rank {args.rank} | Requested GPU {args.gpu_id} "
          f'| Assigned GPU {torch.cuda.current_device()} |')

    # Set the ramdom seed for reproducibility.
    if args.reproducible:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark = False

    # Load data, note we will also call the validation set as the test set.
    print('Loading data...')
    extra_bs = None
    if args.algorithm == "episode":
        extra_bs = args.batchsize * args.communication_interval
    dataset = data_loader(
        dataset_name=args.dataset,
        dataroot=args.dataroot,
        batch_size=args.batchsize,
        val_ratio=(args.val_ratio if args.validation else 0),
        total_clients=args.total_clients,
        world_size=args.world_size,
        rank=args.rank,
        group=group,
        heterogeneity=args.heterogeneity,
        extra_bs=extra_bs,
        small=args.small,
        args=args
    )
    train_loader = dataset[0]
    if args.validation:
        test_loader = dataset[1]
    else:
        test_loader = dataset[2]
    extra_loader = dataset[3]

    if args.model == 'resnet18':
        assert args.dataset == "MNIST"
        net = mnist_resnet18()
    elif args.model == 'resnet32':
        assert args.dataset.startswith("CIFAR")
        net = resnet32()
    elif args.model == 'resnet56':
        assert args.dataset.startswith("CIFAR")
        net = resnet56()
    elif args.model == 'logreg':
        assert args.dataset == "MNIST"
        net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(28 * 28, 10)
        )
    elif args.model == "rnn":
        assert args.dataset in ["SNLI", "Sent140","ShakeSpeare"]
        double_in = args.dataset == "SNLI"
        net = TextClassificationNet(
            word_embed_dim=train_loader.train_set.embed_dim,
            encoder_dim=args.encoder_dim,
            n_enc_layers=args.n_enc_layers,
            dpout_model=args.dpout_model,
            dpout_fc=args.dpout_fc,
            fc_dim=args.fc_dim,
            bsize=args.batchsize,
            n_classes=train_loader.train_set.n_classes,
            pool_type=args.pool_type,
            linear_fc=args.linear_fc,
            bidirectional=(not args.unidirectional),
            rnn=args.rnn,
            double_in=double_in,
        )
    else:
        print(f"Unrecognized model: {args.model}")

    # Initialize or load model weights.
    if not os.path.isfile(args.init_model) and args.rank == 0:
        print("Initializing model weights from scratch.")
        if not os.path.isdir(os.path.dirname(args.init_model)):
            os.makedirs(os.path.dirname(args.init_model))
        torch.save(net.state_dict(), args.init_model)
    dist.barrier()
    print("Loading initial model weights.")
    net.load_state_dict(torch.load(args.init_model))
    net.cuda()

    if args.loss == "svm":
        criterion = MultiClassHingeLoss()
    elif args.loss == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function: {args.loss}.")

    # Train and evaluate the model.
    print("Training...")
    train_results = train(
        args, train_loader, test_loader, extra_loader, net, criterion, group
    )

    # Logging results.
    print('Writing the results.')
    if not os.path.exists(args.log_folder) and args.rank == 0:
        os.makedirs(args.log_folder)
    dist.barrier()
    def get_log_name(rank=None):
        log_name = (f'{args.dataset}_{args.model}_SGDClipGrad_'
                + ('Eta0_%g_' % (args.eta0))
                + ('WD_%g_' % (args.weight_decay))
                + ('Algorithm_%s_' % (args.algorithm))
                + ('Gamma_%g_' % (args.clipping_param))
                + ('Rounds_%d_Batchsize_%d_' % (args.rounds, args.batchsize))
                + ('Comm_I_%d_' % args.communication_interval)
                + ('%s' % ('Validation' if args.validation else 'Test')))
        if rank is not None:
            log_name += f'_Rank_{rank}'
        return log_name
    log_name = get_log_name(args.rank)
    with open(f"{args.log_folder}/{log_name}.json", 'w') as f:
        json.dump(train_results, f)

    # Log average results.
    dist.barrier()
    if args.rank == 0:
        client_results = []
        for rank in range(args.world_size):
            log_name = get_log_name(rank)
            with open(f"{args.log_folder}/{log_name}.json", "r") as f:
                client_results.append(json.load(f))
        keys = list(client_results[0].keys())
        for client_result in client_results[1:]:
            assert keys == list(client_result.keys())

        avg_results = {}
        for key in keys:
            avg_results[key] = np.mean(
                [client_result[key] for client_result in client_results],
                axis=0
            ).tolist()
        log_name = get_log_name()
        with open(f"{args.log_folder}/{log_name}.json", 'w') as f:
            json.dump(avg_results, f)

    print('Finished.')


if __name__ == "__main__":
    main()
