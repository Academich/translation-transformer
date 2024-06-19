from argparse import ArgumentParser
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger


def canonicalize_smiles(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return ''
    return Chem.MolToSmiles(m)


def main(filename: str):
    report = pd.read_csv(filename, header=None).fillna("")
    n_preds = report.shape[1] - 1
    report.columns = ["target"] + [f"pred_{i}" for i in range(1, n_preds + 1)]
    hit = pd.DataFrame()
    for i in range(1, n_preds + 1):
        hit[f"hit_{i}"] = report["target"] == report[f"pred_{i}"].apply(canonicalize_smiles)
    hit_top = pd.DataFrame()
    hit_top["top_1"] = hit["hit_1"]
    for i in range(2, n_preds + 1):
        hit_top[f"top_{i}"] = hit_top[f"top_{i - 1}"] | hit[f"hit_{i}"]
    print("Accuracy")
    print(hit_top.mean(0))
    print("Invalid SMILES")
    print((report[report.columns[1:]] == '').mean(0))


if __name__ == '__main__':
    RDLogger.DisableLog("rdApp.*")
    parser = ArgumentParser()
    parser.add_argument("--filename", '-f', type=str)
    args = parser.parse_args()
    main(args.filename)
