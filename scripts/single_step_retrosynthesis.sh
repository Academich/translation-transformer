CKPT_DIR=checkpoints/single_step_retrosynthesis

CONFIG=${CKPT_DIR}/config.yaml

DATA=USPTO50K
DATA_PATH=data/${DATA}

CKPT='step=337241-v_sq_acc=0.513-v_tk_acc=0.9904-v_l=0.0366.ckpt'
CKPT_PATH=${CKPT_DIR}/${CKPT}

function run_prediction() {
  local GEN=${1:-}
  local BS=${2:-1} # Batch size
  local NBEST=${3:-5} # Number of best sequences
  local N_SPEC_TOK=${4:-10} # Draft sequence length
  local OUTPUT_DIR=${5:-}
  local ATTEMPT=${6:-}

  local MAX_LEN=200
  local NUCLEUS=20.
  local MAX_NUM_OF_DRAFTS=23
  local DRAFT_MODE=true
  local OUTPUT_FILE=${DATA}_${GEN}_bs_${BS}_nbest_${NBEST}_draftlen_${N_SPEC_TOK}_attempt_${ATTEMPT}.csv

  python3 main.py predict -c ${CONFIG} \
          --ckpt_path ${CKPT_PATH} \
          --data.data_dir ${DATA_PATH} \
          --data.vocab_path ${DATA_PATH}/vocabs/vocab.json \
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
          --trainer.callbacks+=PredictionWriter \
          --trainer.callbacks.output_dir=${OUTPUT_DIR}/${OUTPUT_FILE}

}

# Beam search
run_prediction beam_search 1 10 1 retrosynthesis_results_beam_search

# Speculative beam search
run_prediction beam_search_speculative 1 10 10 retrosynthesis_results_beam_search

#Uncomment to run predictions five times to estimate the spread of inference time
#
#N_ATTEMPTS=5
#
#for i in $(seq 1 ${N_ATTEMPTS});
#do
#    for n in 5 10 25;
#    do
#        run_prediction beam_search 1 ${n} 1 retrosynthesis_results_beam_search ${i}
#    done
#done
#
#for i in $(seq 1 ${N_ATTEMPTS});
#do
#    for n in 5 10 25;
#    do
#      for ((d = 1; d <= 20; d += 3));
#        do
#          run_prediction beam_search_speculative 1 ${n} ${d} retrosynthesis_results_beam_search ${i}
#        done
#    done
#done