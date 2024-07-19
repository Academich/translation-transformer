# SMILES-to-SMILES transformer

The repository contains an implementation of the [Molecular Transformer](https://github.com/pschwllr/MolecularTransformer.git) in Pytorch Lightning.


## Speculative decoding
The repository also contains the code to accompany the [manuscript](https://arxiv.org/abs/2407.09685)  
"Accelerating the inference of string generation-based chemical reaction models for industrial applications".

The idea of the manuscript is that reaction product prediction and single-step retrosynthesis with the Molecular Transformer  
can be accelerated by ~3 times on inference without losing accuracy.

### Installation

Create a new isolated environment and install the necessary packages:

```
pip install lightning
pip install jsonargparse[signatures]
pip install lightning[extra]
pip install rdkit
pip install tensorboard
pip install -e .
```

### Data

For reaction prediction, we used USPTO_MIT mixed from the [Molecular Transformer](https://github.com/pschwllr/MolecularTransformer.git) paper.
For single-step retrosynthesis, we used USPTO50k as prepared in the [RSMILES](https://github.com/otori-bird/retrosynthesis) paper.

### Models

The file `main.py` calls a Pytorch Lightning model for either training or inference.
The directory `scripts` contains bash scripts for reaction product prediction and single-step retrosynthesis with speculative decoding.  


### Checkpoints

Trained checkpoints and config files are available at [Google Drive](https://drive.google.com/drive/folders/1uF_wGEUTCz4_xI1uEEeY0V_1QffkOyXI?usp=sharing).
