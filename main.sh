python3 main/main_copy_sequence.py -s fit --seed_everything 123456 \
                --data.data_dir data/copy_sequence --data.batch_size 32 --data.shuffle_train true --data.num_workers 0 \
                --model.warmup_steps 0 --model.scheduler noam \
                --trainer.accelerator cpu --trainer.max_steps 1500 --model.learning_rate 0.1 \
                --trainer.callbacks+=LearningRateMonitor --trainer.callbacks.logging_interval step \
                --trainer.callbacks+=ModelCheckpoint --trainer.callbacks.monitor="train/loss" \
                --trainer.callbacks.filename='{step}-loss={train/loss:.3f}' --trainer.callbacks.mode='min' --trainer.callbacks.save_last=false

python3 main/main_copy_sequence.py -s predict --ckpt_path lightning_logs/version_0/checkpoints/epoch=1-step=1500.ckpt \
                --data.data_dir data/copy_sequence/ --data.batch_size 32 --src_vocab_path data/copy_sequence/vocabs/src-vocab.json \
                --tgt_vocab_path data/copy_sequence/vocabs/tgt-vocab.json --model.beam_size 3 --write_predictions_path results/predictions2.csv