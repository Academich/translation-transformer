CKPT_DIR=checkpoints/reaction_prediction

CONFIG=configs/cfg_standard_product_prediction.yaml

DATA=MIT_mixed
DATA_PATH=data/${DATA}

CKPT='last.ckpt'
CKPT_PATH=${CKPT_DIR}/${CKPT}
VOCAB_PATH=${CKPT_DIR}/vocab.json

# Function to run greedy decoding
function run_greedy() {
  local OUTPUT_DIR=${1:-} # Directory for the results and reports
  local BS=${2:-1} # Batch size
  local GPU=${3:-1}
  local SAVE_PREDICTIONS=${4:-false} # Whether to save predictions to disk. Slows down the run.

  local DEVICE="--trainer.accelerator cpu --trainer.devices 1"
  if [ -n "${GPU}" ]; then
    DEVICE="--trainer.accelerator gpu --trainer.devices [${GPU}]"
  fi

  local MAX_LEN=200
  local OUTPUT_FILE=${DATA}_greedy_batch_${BS}.csv
  local PREDICTION_WRITER=""
  if [ "${SAVE_PREDICTIONS}" = "true" ]; then
    PREDICTION_WRITER="--trainer.callbacks+=callbacks.PredictionWriter --trainer.callbacks.output_dir=${OUTPUT_DIR}/${OUTPUT_FILE}"
  echo ${OUTPUT_FILE}
  fi

  python3 main.py predict -c ${CONFIG} \
          --ckpt_path ${CKPT_PATH} \
          --data.data_dir ${DATA_PATH} \
          --data.vocab_path ${VOCAB_PATH} \
          --model.report_prediction_time true \
          --model.report_prediction_file ${OUTPUT_DIR}/report.txt \
          --data.batch_size ${BS} \
          --model.generation greedy \
          --model.max_len ${MAX_LEN} ${PREDICTION_WRITER} ${DEVICE}
}


# Function to run speculative greedy decoding
function run_greedy_speculative() {
  local OUTPUT_DIR=${1:-} # Directory for the results and reports
  local BS=${2:-1} # Batch size
  local DRAFT_LEN=${3:-10} # Draft sequence length
  local N_DRAFTS=${4:-23} # Maximum number of parallel drafts 
  local GPU=${5:-1}
  local SAVE_PREDICTIONS=${6:-false} # Whether to save predictions to disk. Slows down the run.

  local DEVICE="--trainer.accelerator cpu --trainer.devices 1"
  if [ -n "${GPU}" ]; then
    DEVICE="--trainer.accelerator gpu --trainer.devices [${GPU}]"
  fi

  local MAX_LEN=200
  local OUTPUT_FILE=${DATA}_greedy_speculative_batch_${BS}_dlen_${DRAFT_LEN}_${N_DRAFTS}_drafts.csv
  local PREDICTION_WRITER=""
  if [ "${SAVE_PREDICTIONS}" = "true" ]; then
    PREDICTION_WRITER="--trainer.callbacks+=callbacks.PredictionWriter --trainer.callbacks.output_dir=${OUTPUT_DIR}/${OUTPUT_FILE}"
  echo ${OUTPUT_FILE}
  fi

  python3 main.py predict -c ${CONFIG} \
          --ckpt_path ${CKPT_PATH} \
          --data.data_dir ${DATA_PATH} \
          --data.vocab_path ${VOCAB_PATH} \
          --model.report_prediction_time true \
          --model.report_prediction_file ${OUTPUT_DIR}/report.txt \
          --data.batch_size ${BS} \
          --model.generation greedy_speculative \
          --model.draft_len ${DRAFT_LEN} \
          --model.n_drafts ${N_DRAFTS} \
          --model.max_len ${MAX_LEN} ${PREDICTION_WRITER} ${DEVICE}
}

# Function to run beam search decoding
function run_beam_search() {
  local OUTPUT_DIR=${1:-} # Directory for the results and reports
  local BS=${2:-1} # Batch size
  local NBEST=${3:-0} # Number of best sequences
  local GPU=${4:-1}
  local SAVE_PREDICTIONS=${5:-false} # Whether to save predictions to disk. Slows down the run.

  local DEVICE="--trainer.accelerator cpu --trainer.devices 1"
  if [ -n "${GPU}" ]; then
    DEVICE="--trainer.accelerator gpu --trainer.devices [${GPU}]"
  fi

  local OUTPUT_FILE=${DATA}_beam_search_batch_${BS}_nbest_${NBEST}.csv
  local PREDICTION_WRITER=""
  if [ "${SAVE_PREDICTIONS}" = "true" ]; then
    PREDICTION_WRITER="--trainer.callbacks+=callbacks.PredictionWriter --trainer.callbacks.output_dir=${OUTPUT_DIR}/${OUTPUT_FILE}"
  echo ${OUTPUT_FILE}
  fi

  local MAX_LEN=200

  python3 main.py predict -c ${CONFIG} \
          --ckpt_path ${CKPT_PATH} \
          --data.data_dir ${DATA_PATH} \
          --data.vocab_path ${VOCAB_PATH} \
          --model.report_prediction_time true \
          --model.report_prediction_file ${OUTPUT_DIR}/report.txt \
          --data.batch_size ${BS} \
          --model.generation beam_search \
          --model.beam_size ${NBEST} \
          --model.max_len ${MAX_LEN} ${PREDICTION_WRITER} ${DEVICE}
}


# Function to run speculative beam search decoding
function run_speculative_beam_search() {
  local OUTPUT_DIR=${1:-} # Directory for the results and reports
  local BS=${2:-1} # Batch size
  local NBEST=${3:-5} # Number of best sequences
  local DRAFT_LEN=${4:-10} # Draft sequence length
  local N_DRAFTS=${5:-23} # Maximum number of parallel drafts 
  local GPU=${6:-1}
  local SAVE_PREDICTIONS=${7:-false} # Whether to save predictions to disk. Slows down the run.

  local DEVICE="--trainer.accelerator cpu --trainer.devices 1"
  if [ -n "${GPU}" ]; then
    DEVICE="--trainer.accelerator gpu --trainer.devices [${GPU}]"
  fi

  local OUTPUT_FILE=${DATA}_beam_search_speculative_batch_${BS}_nbest_${NBEST}_dlen_${DRAFT_LEN}_${N_DRAFTS}_drafts.csv
  local PREDICTION_WRITER=""
  if [ "${SAVE_PREDICTIONS}" = "true" ]; then
    PREDICTION_WRITER="--trainer.callbacks+=callbacks.PredictionWriter --trainer.callbacks.output_dir=${OUTPUT_DIR}/${OUTPUT_FILE}"
  echo ${OUTPUT_FILE}
  fi

  local MAX_LEN=200

  python3 main.py predict -c ${CONFIG} \
          --ckpt_path ${CKPT_PATH} \
          --data.data_dir ${DATA_PATH} \
          --data.vocab_path ${VOCAB_PATH} \
          --model.report_prediction_time true \
          --model.report_prediction_file ${OUTPUT_DIR}/report.txt \
          --data.batch_size ${BS} \
          --model.generation beam_search_speculative \
          --model.draft_len ${DRAFT_LEN} \
          --model.beam_size ${NBEST} \
          --model.max_len ${MAX_LEN} \
          --model.n_drafts ${N_DRAFTS} ${PREDICTION_WRITER} ${DEVICE}
}


GPU=0
SAVE_PREDICTIONS=false

# Greedy decoding
# Five runs for time spread estimation
for i in {1..6}; do

  # Batch size 1, 17 draft tokens, 23 drafts
  batch_size=1
  draft_len=17
  n_drafts=23
  run_greedy results_product_final_greedy ${batch_size} ${GPU} ${SAVE_PREDICTIONS}
  run_greedy_speculative results_product_final_greedy_speculative ${batch_size} ${draft_len} ${n_drafts} ${GPU} ${SAVE_PREDICTIONS}

  # Batch size 4, 14 draft tokens, 15 drafts
  batch_size=4
  draft_len=14
  n_drafts=15
  run_greedy results_product_final_greedy ${batch_size} ${GPU} ${SAVE_PREDICTIONS}
  run_greedy_speculative results_product_final_greedy_speculative ${batch_size} ${draft_len} ${n_drafts} ${GPU} ${SAVE_PREDICTIONS}

  # Batch size 16, 7 draft tokens, 7 drafts
  batch_size=16
  draft_len=7
  n_drafts=7
  run_greedy results_product_final_greedy ${batch_size} ${GPU} ${SAVE_PREDICTIONS}
  run_greedy_speculative results_product_final_greedy_speculative ${batch_size} ${draft_len} ${n_drafts} ${GPU} ${SAVE_PREDICTIONS}

  # Batch size 32, 5 draft tokens, 3 drafts
  batch_size=32
  draft_len=5
  n_drafts=3
  run_greedy results_product_final_greedy ${batch_size} ${GPU} ${SAVE_PREDICTIONS}
  run_greedy_speculative results_product_final_greedy_speculative ${batch_size} ${draft_len} ${n_drafts} ${GPU} ${SAVE_PREDICTIONS}

  if [ "$i" -eq 5 ]; then
    SAVE_PREDICTIONS=true
  fi
done


SAVE_PREDICTIONS=false
N_BEST=5

# Beam search decoding with five hypotheses
# Five runs for time spread estimation
for i in {1..6}; do

  # Batch size 1, 10 draft tokens, 23 drafts
  batch_size=1
  draft_len=10
  n_drafts=23
  run_beam_search results_product_final_beam_search ${batch_size} ${N_BEST} ${GPU} ${SAVE_PREDICTIONS}
  run_speculative_beam_search results_product_final_beam_search_speculative ${batch_size} ${N_BEST} ${draft_len} ${n_drafts} ${GPU} ${SAVE_PREDICTIONS}

  # Batch size 2, 14 draft tokens, 10 drafts
  batch_size=2
  draft_len=14
  n_drafts=10
  run_beam_search results_product_final_beam_search ${batch_size} ${N_BEST} ${GPU} ${SAVE_PREDICTIONS}
  run_speculative_beam_search results_product_final_beam_search_speculative ${batch_size} ${N_BEST} ${draft_len} ${n_drafts} ${GPU} ${SAVE_PREDICTIONS}

  # Batch size 3, 9 draft tokens, 10 drafts
  batch_size=3
  draft_len=9
  n_drafts=10
  run_beam_search results_product_final_beam_search ${batch_size} ${N_BEST} ${GPU} ${SAVE_PREDICTIONS}
  run_speculative_beam_search results_product_final_beam_search_speculative ${batch_size} ${N_BEST} ${draft_len} ${n_drafts} ${GPU} ${SAVE_PREDICTIONS}

  # Batch size 4, 10 draft tokens, 7 drafts
  batch_size=4
  draft_len=10
  n_drafts=7
  run_beam_search results_product_final_beam_search ${batch_size} ${N_BEST} ${GPU} ${SAVE_PREDICTIONS}
  run_speculative_beam_search results_product_final_beam_search_speculative ${batch_size} ${N_BEST} ${draft_len} ${n_drafts} ${GPU} ${SAVE_PREDICTIONS}

  if [ "$i" -eq 5 ]; then
    SAVE_PREDICTIONS=true
  fi
done

