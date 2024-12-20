CKPT_DIR=checkpoints/single_step_retrosynthesis

CONFIG=configs/cfg_standard_single_step_retrosyn.yaml

DATA=USPTO_50K_PtoR_aug1
TEST_DATA_PATH=data/${DATA}/test

CKPT='step=337241-v_sq_acc=0.513-v_tk_acc=0.9904-v_l=0.0366.ckpt'
CKPT_PATH=${CKPT_DIR}/${CKPT}
VOCAB_PATH=${CKPT_DIR}/vocab.json


# Function to run beam search decoding
function run_beam_search() {
  local OUTPUT_DIR=${1:-} # Directory for the results and reports
  local BS=${2:-1} # Batch size
  local NBEST=${3:-5} # Number of best sequences
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
          --data.src_test_path ${TEST_DATA_PATH}/src-test.txt \
          --data.tgt_test_path ${TEST_DATA_PATH}/tgt-test.txt \
          --data.vocab_path ${VOCAB_PATH} \
          --model.report_prediction_time true \
          --model.report_prediction_file ${OUTPUT_DIR}/report.txt \
          --data.batch_size ${BS} \
          --model.generation beam_search \
          --model.n_best ${NBEST} \
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
          --data.src_test_path ${TEST_DATA_PATH}/src-test.txt \
          --data.tgt_test_path ${TEST_DATA_PATH}/tgt-test.txt \
          --data.vocab_path ${VOCAB_PATH} \
          --model.report_prediction_time true \
          --model.report_prediction_file ${OUTPUT_DIR}/report.txt \
          --data.batch_size ${BS} \
          --model.generation beam_search_speculative \
          --model.draft_len ${DRAFT_LEN} \
          --model.n_best ${NBEST} \
          --model.max_len ${MAX_LEN} \
          --model.n_drafts ${N_DRAFTS} ${PREDICTION_WRITER} ${DEVICE}
}

# Beam search decoding with five hypotheses
# Five runs for time spread estimation
BATCH_SIZE=1
SAVE_PREDICTIONS=false
for i in {1..5}; do

  # 5 beams, 11 draft tokens, 15 drafts
  run_beam_search results_retrosynthesis_beam_search_bs_${BATCH_SIZE}_nbest_5_gpu${GPU} ${BATCH_SIZE} 5 ${GPU} ${SAVE_PREDICTIONS}
  run_speculative_beam_search results_retrosynthesis_sbs_bs_${BATCH_SIZE}_nbest_5_gpu${GPU} ${BATCH_SIZE} 5 11 15 ${GPU} ${SAVE_PREDICTIONS}

  # 10 beams, 10 draft tokens, 10 drafts
  run_beam_search results_retrosynthesis_beam_search_bs_${BATCH_SIZE}_nbest_10_gpu${GPU} ${BATCH_SIZE} 10 ${GPU} ${SAVE_PREDICTIONS}
  run_speculative_beam_search results_retrosynthesis_sbs_bs_${BATCH_SIZE}_nbest_10_gpu${GPU} ${BATCH_SIZE} 10 10 10 ${GPU} ${SAVE_PREDICTIONS}

  # 15 beams, 10 draft tokens, 10 drafts
  run_beam_search results_retrosynthesis_beam_search_bs_${BATCH_SIZE}_nbest_15_gpu${GPU} ${BATCH_SIZE} 15 ${GPU} ${SAVE_PREDICTIONS}
  run_speculative_beam_search results_retrosynthesis_sbs_bs_${BATCH_SIZE}_nbest_15_gpu${GPU} ${BATCH_SIZE} 15 10 10 ${GPU} ${SAVE_PREDICTIONS}

  # 20 beams, 14 draft tokens, 5 drafts
  run_beam_search results_retrosynthesis_beam_search_bs_${BATCH_SIZE}_nbest_20_gpu${GPU} ${BATCH_SIZE} 20 ${GPU} ${SAVE_PREDICTIONS}
  run_speculative_beam_search results_retrosynthesis_sbs_bs_${BATCH_SIZE}_nbest_20_gpu${GPU} ${BATCH_SIZE} 20 14 5 ${GPU} ${SAVE_PREDICTIONS}

done
