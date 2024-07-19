PROJECT=<>/translation-transformer
VERSION=transformer_vanilla_4_4_low_pad_tgt_batch

CONFIG=<>/config_fixed.yaml

DATA=MIT_mixed
DATA_PATH=<>/${DATA}

CKPT='last.ckpt'
CKPT_PATH=<>/${CKPT}

function run_prediction() {
  local GEN=${1:-}
  local BS=${2:-1}
  local N_SPEC_TOK=${3:-10}
  local OUTPUT_DIR=${4:-}
  local ATTEMPT=${5:-}

  local MAX_LEN=200
  local OUTPUT_FILE=${DATA}_${GEN}_batched_bs_${BS}_draftlen_${N_SPEC_TOK}_attempt_${ATTEMPT}.csv

  python3 main.py predict -c ${CONFIG} \
          --ckpt_path ${CKPT_PATH} \
          --data.data_dir ${DATA_PATH} \
          --model.report_prediction_time true \
          --model.report_prediction_file ${OUTPUT_DIR}/time.txt \
          --data.batch_size ${BS} \
          --model.generation ${GEN} \
          --model.n_speculative_tokens ${N_SPEC_TOK} \
          --model.max_len ${MAX_LEN} \
          --trainer.callbacks+=PredictionWriter \
          --trainer.callbacks.output_dir=${OUTPUT_DIR}/${OUTPUT_FILE}

}

N_ATTEMPTS=5

for i in $(seq 1 ${N_ATTEMPTS});
do
    for ((d = 1; d <= 20; d += 3));
    do
        run_prediction greedy_speculative 1 ${d} product_results_greedy ${i}
    done
done

for i in $(seq 1 ${N_ATTEMPTS});
do
    run_prediction greedy 1 10 product_results_greedy ${i}
done
