import argparse
import os


def get_args():
    parser = argparse.ArgumentParser(description="Histopathological image classification.")
    parser.add_argument("--name", required=True,
                        help="The name of this run is used to monitor.")
    parser.add_argument("--pretrained_dir", default=r"weights/ViT-B_16.npz",
                        help="Directory of pretrained models.")
    parser.add_argument("--save_dir", default="checkpoints",
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--record_dir", default="csv",
                        help="Record Info.")
    parser.add_argument("--gpu_id", type=str, default="0",
                        help="id(s) for CUDA_VISIBLE_DEVICES")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="How many subprocesses are used for data loading.")
    parser.add_argument("--train", action="store_true",
                        help="training mode")
    parser.add_argument("--test_model", type=str,
                        help="If testing mode is selected, please give the directory of the test model."
                             "If --test_model is NONE, we will use the default path.")
    parser.add_argument("--dataset", choices=["BreakHis", "GlaS", "Kidney", "Lung", "Spleen", "YTMF"], default="BreakHis",
                        help="Which dataset will be run.")
    parser.add_argument("--batch_size", type=int, default=285,
                        help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs to train.")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--img_size", type=int, default=256,
                        help="Image size.")
    args = parser.parse_args()

    args.save_dir = os.path.join(args.save_dir, args.name)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args.record_dir = os.path.join(args.record_dir, args.name)
    if not os.path.exists(args.record_dir):
        os.makedirs(args.record_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    return args


if __name__ == "__main__":
    args = get_args()
    if args.dataset == "Kidney" or args.dataset == "Lung" or args.dataset == "Spleen":
        from train.ADL import initial
    elif args.dataset == "BreakHis":
        from train.BreakHis import initial
    elif args.dataset == "GlaS":
        from train.GlaS import initial
    elif args.dataset == "YTMF":
        from train.YTMF import initial
    initial(args)
