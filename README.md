# Translation-transformer

The implementation of a vanilla transformer for the translation of arbitrary sequences using written with Pytorch and
Pytorch Lightning.

### Installation

```
pip install lightning
pip install jsonargparse[signatures]
pip install lightning[extra]
pip install tensorboard
pip install -e .
```

## Copy-sequence

### Data generation

```
python3 src/synthetic_tasks/copy_sequence/data_generation.py --data_dir data/copy_sequence
```

### Training

```
CONFIG=src/synthetic_tasks/copy_sequence/example_config.yaml
python3 main.py fit -c ${CONFIG}
```

or see `main.sh`.

### Testing and inference

See `main.sh`:

```
CONFIG=src/synthetic_tasks/copy_sequence/example_config.yaml
CKPT='last.ckpt'
CKPT_PATH=lightning_logs/${MODEL}/checkpoints/${CKPT}
python3 main.py test -c ${CONFIG} --ckpt_path ${CKPT_PATH}
python3 main.py predict -c ${CONFIG} --ckpt_path ${CKPT_PATH}
```
