python3 main/main_copy_sequence.py -s fit -m vanilla --seed_everything 123456 \
                --data.data_dir data/copy_sequence --data.batch_size 32 --data.shuffle_train true --data.num_workers 0 \
                --model.src_vocab_path data/copy_sequence/vocabs/ascii_vocab.json \
                --model.tgt_vocab_path data/copy_sequence/vocabs/ascii_vocab.json \
                --model.learning_rate 0.1 --model.warmup_steps 200 \
                --trainer.accelerator cpu --trainer.max_steps 1500

python3 main/main_copy_sequence.py -s predict -m vanilla --ckpt_path lightning_logs/version_3/checkpoints/epoch\=1-step\=1500.ckpt \
                --data.data_dir data/copy_sequence/ --data.batch_size 32 --model.src_vocab_path data/copy_sequence/vocabs/ascii_vocab.json \
                --model.tgt_vocab_path data/copy_sequence/vocabs/ascii_vocab.json --model.beam_size 3