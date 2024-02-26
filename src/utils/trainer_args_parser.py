import argparse


def train_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["tensorflow", "pytorch"],
        help="Specify the type of model to train: 'pytorch' or 'tensorflow'",
        nargs="+",
    )
    parser.add_argument("--mlflow", type=bool, default=False)
    parser.add_argument("--nr-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)

    args = parser.parse_args()

    return args
