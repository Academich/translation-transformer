MODEL=example
CONFIG=src/synthetic_tasks/copy_sequence/example_config.yaml
python3 main/main_copy_sequence.py fit -c ${CONFIG} --trainer.logger.init_args.version ${MODEL}

CKPT='last.ckpt'
CKPT_PATH=lightning_logs/${MODEL}/checkpoints/${CKPT}
python3 main/main_copy_sequence.py test -c ${CONFIG} --ckpt_path ${CKPT_PATH} --trainer.logger.init_args.version ${MODEL}
python3 main/main_copy_sequence.py predict -c ${CONFIG} --ckpt_path ${CKPT_PATH} --trainer.logger.init_args.version ${MODEL}
