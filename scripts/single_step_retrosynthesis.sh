CKPT_DIR=checkpoints/single_step_retrosynthesis

CONFIG=configs/cfg_standard_single_step_retrosyn.yaml

TEST_DATA_PATH=data/USPTO_50K_PtoR_aug1/test

CKPT='step=337241-v_sq_acc=0.513-v_tk_acc=0.9904-v_l=0.0366.ckpt'
CKPT_PATH=${CKPT_DIR}/${CKPT}
VOCAB_PATH=${CKPT_DIR}/vocab.json

function run_prediction() {
  local OUTPUT_DIR=${1:-}
  local GEN=${2:-}
  local BS=${3:-1} # Batch size
  local NBEST=${4:-5} # Number of best sequences
  local N_SPEC_TOK=${5:-10} # Draft sequence length
  local ATTEMPT_NUM=${6:-}

  local ATTEMPT=""
  if [ -n "${ATTEMPT_NUM}" ]; then
    ATTEMPT="_attempt_${ATTEMPT_NUM}"
  fi

  local MAX_LEN=200
  local NUCLEUS=20.
  local MAX_NUM_OF_DRAFTS=23
  local DRAFT_MODE=true
  local OUTPUT_FILE=${DATA}_${GEN}_bs_${BS}_nbest_${NBEST}_draftlen_${N_SPEC_TOK}${ATTEMPT}.csv

  python3 main.py predict -c ${CONFIG} \
          --ckpt_path ${CKPT_PATH} \
          --data.src_test_path ${TEST_DATA_PATH}/src-test.txt \
          --data.tgt_test_path ${TEST_DATA_PATH}/tgt-test.txt \
          --data.vocab_path ${VOCAB_PATH} \
          --model.report_prediction_time true \
          --model.report_prediction_file ${OUTPUT_DIR}/time.txt \
          --data.batch_size ${BS} \
          --model.generation ${GEN} \
          --model.n_speculative_tokens ${N_SPEC_TOK} \
          --model.beam_size ${NBEST} \
          --model.n_best ${NBEST} \
          --model.nucleus ${NUCLEUS} \
          --model.max_len ${MAX_LEN} \
          --model.max_num_of_drafts ${MAX_NUM_OF_DRAFTS} \
          --model.draft_mode ${DRAFT_MODE} \
          --trainer.callbacks+=callbacks.PredictionWriter \
          --trainer.callbacks.output_dir=${OUTPUT_DIR}/${OUTPUT_FILE}

}

# Beam search
BATCH_SIZE=1
METHOD=beam_search
N_BEST=10
run_prediction results_retrosynthesis_${METHOD} ${METHOD} ${BATCH_SIZE} ${N_BEST}

# Speculative beam search
METHOD=beam_search_speculative
DRAFT_LEN=10
run_prediction results_retrosynthesis_${METHOD} ${METHOD} ${BATCH_SIZE} ${N_BEST} ${DRAFT_LEN}

#Uncomment to run predictions five times to estimate the spread of inference time

# N_ATTEMPTS=5

# METHOD=beam_search
# for i in $(seq 1 ${N_ATTEMPTS});
# do
#    for n in 5 10 25;
#    do
#        run_prediction results_retrosynthesis_${METHOD} ${METHOD} 1 ${n} 1 ${i}
#    done
# done

# METHOD=beam_search_speculative
# for i in $(seq 1 ${N_ATTEMPTS});
# do
#    for n in 5 10 25;
#    do
#      for ((d = 1; d <= 20; d += 3));
#        do
#          run_prediction results_retrosynthesis_${METHOD} ${METHOD} 1 ${n} ${d} ${i}
#        done
#    done
# done