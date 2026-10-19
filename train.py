"""Train a PushT head through scripts/train.py (legacy --exp 2 command).

Use scripts/train.py --dataset pusht --exp exp02a for new commands.
Only experiment 2 remains registered. Legacy public imports are retained.
"""
import argparse

from models.vla_train import VLATrainModel  # noqa: F401

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--exp", type=int, default=2, choices=[2])
    _, remaining = parser.parse_known_args(argv)
    from scripts.train import main as run

    run(["--dataset", "pusht", "--exp", "exp02a", *remaining])


if __name__ == "__main__":
    main()
