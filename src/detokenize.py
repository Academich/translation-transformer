"""This script detokenizes the text files with tokenized SMILES that are typically used to train models in OpenNMT."""

from argparse import ArgumentParser
from pathlib import Path
from itertools import product


def main(data_dir: str):
    """
    Detokenizes the text files with tokenized SMILES in-place.
    For example, turns strings like
    O = C ( N c 1 c c c ( O c 2 c c n c 3 [nH] c c c 2 3 ) c ( F ) c 1 ) C ( F ) ( F ) F
    into
    O=C(NCc1ccccc1S(=O)(=O)C1CC1)C(F)(F)F

    The directory is expected to contain files:
    src-train.txt, src-val.txt, src-test.txt, tgt-train.txt, tgt-val.txt, tgt-test.txt
    """
    data_dir = Path(data_dir).resolve()
    for a, b in product(("src", "tgt"), ("test", "val", "train")):
        name = data_dir / f"{a}-{b}.txt"
        try:
            with open(name) as f:
                content = [i.strip().replace(" ", "") for i in f.readlines()]
                content = [i for i in content if i]
        except FileNotFoundError:
            print(f"File {name} not found")
            continue
        with open(name, "w") as f:
            f.write("\n".join(content))
        print(f"Detokenized {name}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        "-d",
        default="./",
        type=str,
        help="Path to the directory with tokenized SMILES files.",
    )
    args = parser.parse_args()
    main(args.data_dir)
