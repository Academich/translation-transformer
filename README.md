# SMILES-to-SMILES transformer

The repository contains an implementation of the [Molecular Transformer](https://github.com/pschwllr/MolecularTransformer.git) in Pytorch Lightning.


## Speculative decoding
The repository also contains the code to accompany the [manuscript](https://arxiv.org/abs/2407.09685)  
"Accelerating the inference of string generation-based chemical reaction models for industrial applications".

The idea of the manuscript is that reaction product prediction and single-step retrosynthesis with the Molecular Transformer  
can be accelerated by ~3 times on inference without losing accuracy.

### Installation

Create a new isolated environment with Python 3.10 and install the necessary packages:

```bash
pip install lightning
pip install jsonargparse[signatures]
pip install rdkit
pip install tensorboard
pip install gdown
pip install -e .
```
If you are using Weights&Biases, run also `pip install wandb`.

### Data

For reaction prediction, we used USPTO_MIT mixed from the [Molecular Transformer](https://github.com/pschwllr/MolecularTransformer.git) paper.
For single-step retrosynthesis, we used USPTO50k as prepared in the [RSMILES](https://github.com/otori-bird/retrosynthesis) paper.

**Download USPTO MIT mixed:**
```bash
gdown https://drive.google.com/drive/folders/1fJ7Hm55IDevIi5Apna7v-rQBQStTH7Yg -O data/MIT_mixed --folder
python3 src/detokenize.py --data_dir data/MIT_mixed
```

**Download USPTO 50K** and augment it using [RSMILES](https://github.com/otori-bird/retrosynthesis) augmentation.  
Clone the RSMILES repository to some path in your system.
```bash
pip install pandas textdistance

THIS_REPO_PATH=$(pwd) # The full path to this repository 
RSMILES_PATH=../retrosynthesis  # as an example; the path to the RSMILES repository

gdown https://drive.google.com/drive/folders/1la4OgBKgm2K-IRwuV-GHUNjN3bcCrl6v -O ${RSMILES_PATH}/dataset/USPTO_50K --folder
cd ${RSMILES_PATH}
AUGMENTATIONS=20
PROCESSES=8
python3 preprocessing/generate_PtoR_data.py -augmentation ${AUGMENTATIONS} -processes ${PROCESSES} -test_except
python3 preprocessing/generate_PtoR_data.py -augmentation 1 -processes ${PROCESSES} -test_only -canonical
mv dataset/USPTO_50K_PtoR_aug${AUGMENTATIONS} ${THIS_REPO_PATH}/data # The augmented dataset is now in this repository
mv dataset/USPTO_50K_PtoR_aug1 ${THIS_REPO_PATH}/data
cd $THIS_REPO_PATH
python3 src/detokenize.py --data_dir data/USPTO_50K_PtoR_aug1/test
python3 src/detokenize.py --data_dir data/USPTO_50K_PtoR_aug${AUGMENTATIONS}/train
python3 src/detokenize.py --data_dir data/USPTO_50K_PtoR_aug${AUGMENTATIONS}/val
```

### Models

The file `main.py` calls a Pytorch Lightning model for either training or inference.
The directory `scripts` contains bash scripts for reaction product prediction and single-step retrosynthesis with speculative decoding.  


### Checkpoints

Trained checkpoints and config files are available at [Google Drive](https://drive.google.com/drive/folders/1uF_wGEUTCz4_xI1uEEeY0V_1QffkOyXI?usp=sharing).
Download reaction prediction checkpoints:
```bash
mkdir checkpoints
mkdir checkpoints/reaction_prediction
gdown https://drive.google.com/drive/folders/1sBiVgFZyD4F42nVqR835-0Tl90LkQvU9 -O checkpoints/reaction_prediction --folder
```
Download single-step retrosynthesis checkpoints
```bash
mkdir checkpoints/single_step_retrosynthesis
gdown https://drive.google.com/drive/folders/1v4pKYWlE0qNA-ksa7yX55i7qMeesURON -O checkpoints/single_step_retrosynthesis --folder
```

### Inference
The scripts `scripts/product_prediction.sh` and `scripts/single_step_retrosynthesis.sh` run the models for reaction product prediction and single-step retrosynthesis, respectively.  
The forward pass of the transformer is implemented in `src/model/lightning_model.py`.  
The sampling of sequences from the logits predicted by the transformer is implemented in `src/decoding/standard_decoding.py` and `src/decoding/speculative_decoding.py`.  
The generated sequences are written to CSV files using the `PredictionWriter` callback implemented in `src/callbacks.py`. For example, the script `scripts/product_prediction.sh` loads a checkpoint of the transformer and runs the model on the USPTO MIT mixed test dataset.  
The generated sequences are then saved to, e.g., `results_product_greedy_speculative/MIT_mixed_greedy_speculative_batched_bs_1_draftlen_10.csv`. 

### Evaluating performance
The script `src/score_predictions.py` scores the predictions saved in CSV files. It calculates the top-N accuracy and the percentage of invalid SMILES.

Example usage:
```bash
python3 src/score_predictions.py -f results_product_greedy_speculative/MIT_mixed_greedy_speculative_batched_bs_1_draftlen_10.csv
```
The accuracy is printed to the terminal.
