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

  local DEVICE="--trainer.accelerator cpu --trainer.devices 1"
  if [ -n "${GPU}" ]; then
    DEVICE="--trainer.accelerator gpu --trainer.devices [${GPU}]"
  fi

  local MAX_LEN=200
  local OUTPUT_FILE=${DATA}_greedy_batch_${BS}.csv
  echo ${OUTPUT_FILE}

  python3 main.py predict -c ${CONFIG} \
          --ckpt_path ${CKPT_PATH} \
          --data.data_dir ${DATA_PATH} \
          --data.vocab_path ${VOCAB_PATH} \
          --model.report_prediction_time true \
          --model.report_prediction_file ${OUTPUT_DIR}/report.txt \
          --data.batch_size ${BS} \
          --model.generation greedy \
          --model.max_len ${MAX_LEN} \
          --trainer.callbacks+=callbacks.PredictionWriter \
          --trainer.callbacks.output_dir=${OUTPUT_DIR}/${OUTPUT_FILE} ${DEVICE}
}


# Function to run speculative greedy decoding
function run_greedy_speculative() {
  local OUTPUT_DIR=${1:-} # Directory for the results and reports
  local BS=${2:-1} # Batch size
  local DRAFT_LEN=${3:-10} # Draft sequence length
  local N_DRAFTS=${4:-23} # Maximum number of parallel drafts 
  local GPU=${5:-1}

  local DEVICE="--trainer.accelerator cpu --trainer.devices 1"
  if [ -n "${GPU}" ]; then
    DEVICE="--trainer.accelerator gpu --trainer.devices [${GPU}]"
  fi

  local MAX_LEN=200
  local OUTPUT_FILE=${DATA}_greedy_speculative_batch_${BS}_dlen_${DRAFT_LEN}_${N_DRAFTS}_drafts.csv
  echo ${OUTPUT_FILE}

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
          --model.max_len ${MAX_LEN} \
          --trainer.callbacks+=callbacks.PredictionWriter \
          --trainer.callbacks.output_dir=${OUTPUT_DIR}/${OUTPUT_FILE} ${DEVICE}
}

# Function to run beam search decoding
function run_beam_search() {
  local OUTPUT_DIR=${1:-} # Directory for the results and reports
  local BS=${2:-1} # Batch size
  local NBEST=${3:-5} # Number of best sequences
  local GPU=${4:-1}

  local DEVICE="--trainer.accelerator cpu --trainer.devices 1"
  if [ -n "${GPU}" ]; then
    DEVICE="--trainer.accelerator gpu --trainer.devices [${GPU}]"
  fi

  local MAX_LEN=200
  local OUTPUT_FILE=${DATA}_beam_search_batch_${BS}_nbest_${NBEST}.csv
  echo ${OUTPUT_FILE}

  python3 main.py predict -c ${CONFIG} \
          --ckpt_path ${CKPT_PATH} \
          --data.data_dir ${DATA_PATH} \
          --data.vocab_path ${VOCAB_PATH} \
          --model.report_prediction_time true \
          --model.report_prediction_file ${OUTPUT_DIR}/report.txt \
          --data.batch_size ${BS} \
          --model.generation beam_search \
          --model.n_best ${NBEST} \
          --model.max_len ${MAX_LEN} \
          --trainer.callbacks+=callbacks.PredictionWriter \
          --trainer.callbacks.output_dir=${OUTPUT_DIR}/${OUTPUT_FILE} ${DEVICE}
}


# Function to run speculative beam search decoding
function run_speculative_beam_search() {
  local OUTPUT_DIR=${1:-} # Directory for the results and reports
  local BS=${2:-1} # Batch size
  local NBEST=${3:-5} # Number of best sequences
  local DRAFT_LEN=${4:-10} # Draft sequence length
  local N_DRAFTS=${5:-23} # Maximum number of parallel drafts 
  local GPU=${6:-1}

  local DEVICE="--trainer.accelerator cpu --trainer.devices 1"
  if [ -n "${GPU}" ]; then
    DEVICE="--trainer.accelerator gpu --trainer.devices [${GPU}]"
  fi

  local MAX_LEN=200
  local OUTPUT_FILE=${DATA}_beam_search_speculative_batch_${BS}_nbest_${NBEST}_dlen_${DRAFT_LEN}_${N_DRAFTS}_drafts.csv
  echo ${OUTPUT_FILE}

  python3 main.py predict -c ${CONFIG} \
          --ckpt_path ${CKPT_PATH} \
          --data.data_dir ${DATA_PATH} \
          --data.vocab_path ${VOCAB_PATH} \
          --model.report_prediction_time true \
          --model.report_prediction_file ${OUTPUT_DIR}/report.txt \
          --data.batch_size ${BS} \
          --model.generation beam_search_speculative \
          --model.draft_len ${DRAFT_LEN} \
          --model.n_best ${NBEST} \
          --model.max_len ${MAX_LEN} \
          --model.n_drafts ${N_DRAFTS} \
          --trainer.callbacks+=callbacks.PredictionWriter \
          --trainer.callbacks.output_dir=${OUTPUT_DIR}/${OUTPUT_FILE} ${DEVICE}
}


GPU=7

# Greedy decoding
for batch_size in 1 4 16 32; do
  run_greedy results_product_greedy ${batch_size} ${GPU}
done

# Beam search decoding with five beams
NBEST=5
for batch_size in 1 2 4 8; do
  run_beam_search results_product_beam_search ${batch_size} ${NBEST} ${GPU}
done

# # Speculative greedy decoding
# DRAFT_LEN=10
# N_DRAFTS=23
# run_greedy_speculative results_product_greedy_speculative ${BATCH_SIZE} ${DRAFT_LEN} ${N_DRAFTS} ${GPU}

# # Speculative beam search as greedy
# NBEST=5
# run_speculative_beam_search results_product_beam_search_speculative ${BATCH_SIZE} ${NBEST} ${DRAFT_LEN} ${N_DRAFTS} ${GPU}

