from argparse import ArgumentParser
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger


def canonicalize_smiles(s):
    if s == "":
        return s
    m = Chem.MolFromSmiles(s)
    if m is None:
        return "!"
    return Chem.MolToSmiles(m)


def main(filename: str):
    with open(filename) as f:
        lines = [i.strip() for i in f.readlines()]
    source, target, preds = [], [], []
    for line in lines:
        s, t, *ps = line.split(",")
        source.append(s)
        target.append(t)
        preds.append(ps)
    preds = pd.DataFrame(preds).fillna("")
    target = pd.Series(target)
    n_queries, n_preds = preds.shape

    report = pd.concat((target, preds), axis=1)
    report.columns = ["target"] + [f"prediction {i}" for i in range(1, n_preds + 1)]
    for c in report.columns:
        report[c] = report[c].apply(canonicalize_smiles)
    hit = pd.DataFrame()
    for i in range(1, n_preds + 1):
        hit[f"hit_{i}"] = report["target"] == report[f"prediction {i}"]
    hit_top = pd.DataFrame()
    hit_top["top 1"] = hit["hit_1"]
    for i in range(2, n_preds + 1):
        hit_top[f"top {i}"] = hit_top[f"top {i - 1}"] | hit[f"hit_{i}"]

    accuracy = hit_top.mean(0)[
        [f"top {i}" for i in [1, 3, 5, 10, 50] if i <= hit_top.shape[1]]
    ]
    invalid_smiles = (report[report.columns[1:]] == "!").mean(0)[
        [f"prediction {i}" for i in [1, 3, 5, 10, 50] if i <= hit_top.shape[1]]
    ]
    empty_smiles = (report[report.columns[1:]] == "").mean(0)[
        [f"prediction {i}" for i in [1, 3, 5, 10, 50] if i <= hit_top.shape[1]]
    ]
    print("Accuracy, %")
    print((accuracy * 100).to_string())
    print()
    print("Invalid SMILES, %")
    print((invalid_smiles * 100).to_string())
    print()
    print("Empty SMILES, %")
    print((empty_smiles * 100).to_string())


if __name__ == "__main__":
    RDLogger.DisableLog("rdApp.*")
    parser = ArgumentParser()
    parser.add_argument("--filename", "-f", type=str)
    args = parser.parse_args()
    main(args.filename)
