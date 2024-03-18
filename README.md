# Translation-transformer
The implementation of a vanilla transformer for the translation of arbitrary sequences using written with Pytorch and
Pytorch Lightning.

### Installation
```
pip install lightning
pip install jsonargparse[signatures]
pip install lightning[extra]
pip install -e .
```

## Copy-sequence

### Data generation
```
python3 src/synthetic_tasks/copy_sequence/data_generation.py --data_dir data/copy_sequence
```

### Training
```
python3 main.py -s fit -m vanilla --seed_everything 123456 --data.data_dir data/copy_sequence --data.batch_size 32 --data.shuffle_train true --model.src_vocab_path data/copy_sequence/vocabs/ascii_vocab.json --model.tgt_vocab_path data/copy_sequence/vocabs/ascii_vocab.json --trainer.accelerator gpu --trainer.devices [1] --trainer.max_steps 1500 --model.learning_rate 0.1 --model.warmup_steps 200
```

### Validation
```
python3 main.py -s validate -m vanilla --ckpt_path lightning_logs/version_3/checkpoints/epoch\=1-step\=1500.ckpt --data.data_dir data/copy_sequence/ --data.batch_size 32 --model.src_vocab_path data/copy_sequence/vocabs/ascii_vocab.json --model.tgt_vocab_path data/copy_sequence/vocabs/ascii_vocab.json --trainer.accelerator gpu --trainer.devices [1]
```