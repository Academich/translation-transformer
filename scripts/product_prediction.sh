CKPT_DIR=checkpoints/reaction_prediction

CONFIG=configs/cfg_standard_product_prediction.yaml

DATA=MIT_mixed
DATA_PATH=data/${DATA}

CKPT='last.ckpt'
CKPT_PATH=${CKPT_DIR}/${CKPT}
VOCAB_PATH=${CKPT_DIR}/vocab.json

function run_prediction() {
  local OUTPUT_DIR=${1:-}
  local GEN=${2:-}
  local BS=${3:-1}
  local N_SPEC_TOK=${4:-10}
  local ATTEMPT_NUM=${5:-}

  local ATTEMPT=""
  if [ -n "${ATTEMPT_NUM}" ]; then
    ATTEMPT="_attempt_${ATTEMPT_NUM}"
  fi

  local MAX_LEN=200
  local OUTPUT_FILE=${DATA}_${GEN}_batched_bs_${BS}_draftlen_${N_SPEC_TOK}${ATTEMPT}.csv

  python3 main.py predict -c ${CONFIG} \
          --ckpt_path ${CKPT_PATH} \
          --data.data_dir ${DATA_PATH} \
          --data.vocab_path ${VOCAB_PATH} \
          --model.report_prediction_time true \
          --model.report_prediction_file ${OUTPUT_DIR}/time.txt \
          --data.batch_size ${BS} \
          --model.generation ${GEN} \
          --model.n_speculative_tokens ${N_SPEC_TOK} \
          --model.max_len ${MAX_LEN} \
          --trainer.callbacks+=callbacks.PredictionWriter \
          --trainer.callbacks.output_dir=${OUTPUT_DIR}/${OUTPUT_FILE}

}

BATCH_SIZE=1
METHOD=greedy

# Greedy decoding
run_prediction results_product_${METHOD} ${METHOD} ${BATCH_SIZE}

# Speculative greedy decoding
METHOD=greedy_speculative
DRAFT_LEN=10
run_prediction results_product_${METHOD} ${METHOD} ${BATCH_SIZE} ${DRAFT_LEN}


#Uncomment to run predictions five times to estimate the spread of inference time

# N_ATTEMPTS=5

# METHOD=greedy_speculative
# for i in $(seq 1 ${N_ATTEMPTS});
# do
#    for ((d = 1; d <= 20; d += 3));
#    do
#        run_prediction results_product_${METHOD} ${METHOD} ${BATCH_SIZE} ${d} ${i}
#    done
# done

# METHOD=greedy
# for i in $(seq 1 ${N_ATTEMPTS});
# do
#    run_prediction results_product_${METHOD} ${METHOD} ${BATCH_SIZE} 1 ${i}
# done
