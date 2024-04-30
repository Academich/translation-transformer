import random
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger

random.seed(12345)


# What exactly should we augment
# Only train and only source?

def random_noncanonical_smiles(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    root_atom = random.randint(0, len(mol.GetAtoms()) - 1)
    return Chem.MolToSmiles(mol, rootedAtAtom=root_atom, canonical=False)


def augment_smiles(smiles: list[str], shuffle_prob: int | float = 0):
    _enable_shuffling = True
    if shuffle_prob == 0:
        _enable_shuffling = False
    augm_smiles = []
    for record in smiles:
        augm_record = [random_noncanonical_smiles(s) for s in record.split(".")]
        if _enable_shuffling and random.choices([True, False], weights=[shuffle_prob, 1 - shuffle_prob])[0]:
            random.shuffle(augm_record)
        augm_smiles.append(".".join(augm_record))
    return augm_smiles


if __name__ == '__main__':
    RDLogger.DisableLog('rdApp.*')
    DATA_DIR = Path("../../../data/MIT_mixed").resolve()
    TRAIN_SRC = DATA_DIR / "src-train.txt"
    TRAIN_TGT = DATA_DIR / "tgt-train.txt"

    with TRAIN_SRC.open() as f1, TRAIN_TGT.open() as f2:
        src_smiles = [r.strip() for r in f1.readlines()]
        tgt_smiles = [r.strip() for r in f2.readlines()]
    src_smiles_augm = augment_smiles(src_smiles, shuffle_prob=0.4)
    tgt_smiles_augm = tgt_smiles
    with (DATA_DIR / "src-train_augm.txt").open("w") as f:
        f.write("\n".join(src_smiles + src_smiles_augm))
    with (DATA_DIR / "tgt-train_augm.txt").open("w") as f:
        f.write("\n".join(tgt_smiles + tgt_smiles_augm))

