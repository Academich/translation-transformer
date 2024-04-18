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
python3 main.py -s fit -m vanilla --seed_everything 123456 --data.data_dir data/copy_sequence --data.batch_size 32 --data.shuffle_train true --model.src_vocab_path data/copy_sequence/vocabs/ascii_vocab.json --model.tgt_vocab_path data/copy_sequence/vocabs/ascii_vocab.json --trainer.accelerator gpu --trainer.devices [1] --trainer.max_steps 1500 --model.learning_rate 0.1 --model.warmup_steps 200
```

### Validation
```
python3 main.py -s validate -m vanilla --ckpt_path lightning_logs/version_3/checkpoints/epoch\=1-step\=1500.ckpt --data.data_dir data/copy_sequence/ --data.batch_size 32 --model.src_vocab_path data/copy_sequence/vocabs/ascii_vocab.json --model.tgt_vocab_path data/copy_sequence/vocabs/ascii_vocab.json --trainer.accelerator gpu --trainer.devices [1]
```

### Reaction prediction example
```
MODEL=transformer_vanilla_4_4
CKPT='step=63915-val_tok_acc=0.992-val_l=0.026.ckpt'
CKPT_PATH=lightning_logs/${MODEL}/checkpoints/${CKPT}
WRITE_PATH=results/reaction_prediction_${MODEL}.csv
python3 main/main_reaction_prediction.py -s fit --seed_everything 1234123 --model VanillaTransformerTranslationLightningModule \
                --write_predictions_path ${WRITE_PATH} \
                --ckpt_path ${CKPT_PATH} \
                --data.data_dir data/MIT_mixed --data.batch_size 32 --data.shuffle_train true --data.num_workers 4 \
                --model.num_encoder_layers 4 --model.num_decoder_layers 4 --model.embedding_dim 256 --model.num_heads 8 \
                --model.feedforward_dim 2048 --model.dropout_rate 0.1 --model.share_embeddings true --model.beam_size 5 --model.max_len 150 \
                --model.warmup_steps 0 --model.scheduler const --model.learning_rate 3e-4 \
                --trainer.accelerator gpu --trainer.devices [0] --trainer.max_steps 2000000 \
                --trainer.logger+=TensorBoardLogger --trainer.logger.save_dir='.' --trainer.logger.version ${MODEL} --trainer.profiler SimpleProfiler
```