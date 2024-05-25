#!/bin/bash
DIR=`pwd`
###############################################################################
### Main configs
## GPT-3 models use 2K sequence length/context window
SEQ_LEN=1024

## GPT-3 Small 125M
MODEL_SIZE=0.08
NUM_LAYERS=12
HIDDEN_SIZE=768
FFN_HIDDEN_SIZE=3072
MOE_FFN_HIDDEN_SIZE=384
NUM_FFN_HEADS=1
NUM_ATTN_HEADS=12
GLOBAL_BATCH_SIZE=256

###############################################################################
### Training duration configs
## The main termination condition, original GPT-3 paper trains for 300B tokens
## For MoE model, we found sometimes training a bit more to 330B tokens helps
TRAIN_TOKENS=300000000000

## TRAIN_ITERS is another termination condition and also affect the number of
## data samples to be indexed. Since we want to reach the TRAIN_TOKENS
## above, and techniques like curriculum learning has less token in some steps,
## so we just set this config large enough to make sure we have enough
## processed data and don't terminate by TRAIN_ITERS.
TRAIN_ITERS=60000
WARMUP_FRACTION=0.1

## Another termination condition in minutes. Set it large enough to avoid
## undesired early termination.
EXIT_DURATION=30000000
###############################################################################
### LR configs
## LR warmup and decay duration, this token-based config is preferable since
## no need to readjust when the batch size/seqlen is changed.
## Original GPT-3 paper uses 375M warmup tokens and 260B decay tokens.
## For MoE model, we found that setting the decay token to 300B helps.
WARMUP_TOKENS=375000000
LR_DECAY_TOKENS=300000000000
###############################################################################
### Parallelism configs
## Micro batch size per GPU
## Make sure that BATCH_SIZE <= GLOBAL_BATCH_SIZE*PP_SIZE*MP_SIZE/NUM_GPUS
BATCH_SIZE=16

## Model parallelism, 1 is no MP
## Currently MoE models have divergence issue when MP > 1.
MP_SIZE=1

## Pipeline parallelism
## Currently we don't support PP for MoE. To disable PP, set PP_SIZE
## to 1 and use the "--no-pipeline-parallel" arg.
PP_SIZE=1
NUM_GPUS=$(($(ds_ssh nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)-2))
NUM_GPUS_PERNODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
NUM_NODE=$(( ${NUM_GPUS} / ${NUM_GPUS_PERNODE} ))

NUM_GPUS=4
NUM_NODE=1
###############################################################################
### MoE configs
## Number of experts. EP_SIZE 1 means dense model without MoE
EP_SIZE=64
EP_INTERVAL=2
TOPK=8
THRESHOLD=0.90
GATE_VIEW_NUM=1

if [[ $EP_SIZE -gt $NUM_GPUS ]]; then
    EP_PARALLEL_SIZE=$NUM_GPUS
else
    EP_PARALLEL_SIZE=$EP_SIZE
fi

## Original GPT-3 model always set min LR at 10% of max LR. For MoE model, we
## found that lower LR and min LR (than the base dense model) helps.
## For 1.3B MoE-128 model we used LR=1.2e-4 and MIN_LR=1.0e-6.
## For 350M MoE-128 model we used LR=2.0e-4 and MIN_LR=2.0e-6, but they are not
## heavily tuned.
LR=4.5e-4
MIN_LR=4.5e-06

## Coefficient for MoE loss. We find that 0.01 is a good value at least for
## 1.3B MoE-128 model
MLC=0.01

## Below configs adjust the MoE expert token capacity limit during training and
## eval. To completely disable capacity limit, set MOE_DROP_TOKEN to false.
## Larger capacity factor or disabling capacity limit could improve training
## convergence, but will also reduce training throughput.
MOE_TRAIN_CAP_FACTOR=8.0
MOE_EVAL_CAP_FACTOR=8.0
MOE_MIN_CAP=4
MOE_DROP_TOKEN="true"
###############################################################################
### Curriculum learning (CL) configs
## Enable/disable CL
CL_ENABLED="false"
## Consult the tutorial https://www.deepspeed.ai/tutorials/curriculum-learning/
## for tuning the following configs
CL_START_SEQLEN=80
CL_AVG_SEQLEN=$(( (${CL_START_SEQLEN} + ${SEQ_LEN}) / 2 ))
CL_TOKENS=60
CL_TOKENS=$((${CL_TOKENS} * 1000000000))
CL_STEP=$(( ${CL_TOKENS} / (${GLOBAL_BATCH_SIZE} * ${CL_AVG_SEQLEN}) ))
###############################################################################
### Misc configs
LOG_INTERVAL=10
TEST_ITERS=100
EVAL_ITERS=10
EVAL_INTERVAL=100
SAVE_INTERVAL=1000 # i don't need save for exploring experiments

## Standard deviation for weight initialization
## We used 0.014 for 350M/1.3B dense/MoE models, and used 0.01 for 6.7B
## dense model. Usually larger model needs lower std.
INIT_STD=0.014

## Activation checkpointing saves GPU memory, but reduces training speed
ACTIVATION_CHECKPOINT="false"
###############################################################################
### Output and data configs
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
host="${HOSTNAME}"
NAME="gpt-${MODEL_SIZE}B-lr-${LR}-minlr-${MIN_LR}-bs-${GLOBAL_BATCH_SIZE}-gpus-${NUM_GPUS}-mp-${MP_SIZE}-pp-${PP_SIZE}"

if [[ $EP_SIZE -gt 1 ]]; then
    NAME="Threshold-local-${THRESHOLD}-${NAME}-ep-${EP_SIZE}-size-${MOE_FFN_HIDDEN_SIZE}-interval-${EP_INTERVAL}-mlc-${MLC}-cap-${MOE_TRAIN_CAP_FACTOR}-${MOE_EVAL_CAP_FACTOR}-drop-${MOE_DROP_TOKEN}"
fi
if [ "${CL_ENABLED}" = "true" ]; then
    NAME="${NAME}-cl-${CL_START_SEQLEN}-${CL_STEP}"
fi

OUTPUT_BASEPATH=$DIR/output
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}_${host}_${current_time}"
mkdir -p ${TENSORBOARD_DIR}
## Note that for MoE model with billion-scale base model, the checkpoint can be
## as large as TB-scale which normal NFS cannot handle efficiently.
CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${NAME}"

# USE_INTERNAL_DATA="true"
USE_INTERNAL_DATA="false"

VOCAB_PATH=/data/yuanhang/MoE-Megatron-DeepSpeed/dataset/gpt2-vocab.json
MERGE_PATH=/data/yuanhang/MoE-Megatron-DeepSpeed/dataset/gpt2-merges.txt
DATA_PATH=/data/yuanhang/pretrain_data/gpt2-dataset_text_document

###############################################################################
data_options=" \
         --vocab-file ${VOCAB_PATH} \
         --merge-file ${MERGE_PATH} \
         --data-path ${DATA_PATH} \
         --data-impl mmap"

megatron_options=" \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --tensor-model-parallel-size ${MP_SIZE} \
        --moe-expert-parallel-size ${EP_PARALLEL_SIZE} \
        --num-experts ${EP_SIZE} \
        --topk ${TOPK} \
        --expert-interval ${EP_INTERVAL}
        --moe-loss-coeff ${MLC} \
        --moe-train-capacity-factor ${MOE_TRAIN_CAP_FACTOR} \
        --moe-eval-capacity-factor ${MOE_EVAL_CAP_FACTOR} \
        --moe-min-capacity ${MOE_MIN_CAP} \
        --init-method-std ${INIT_STD} \
        --lr-warmup-fraction ${WARMUP_FRACTION} \
        --micro-batch-size ${BATCH_SIZE} \
        --exit-duration-in-mins ${EXIT_DURATION} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
        --moe-ffn-hidden-size ${MOE_FFN_HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --num-ffn-heads ${NUM_FFN_HEADS} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${SEQ_LEN} \
        --train-iters ${TRAIN_ITERS} \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --split 97,2,1 \
        --log-interval ${LOG_INTERVAL} \
        --eval-interval ${EVAL_INTERVAL} \
        --eval-iters ${EVAL_ITERS} \
        --test-iters ${TEST_ITERS} \
        --save-interval ${SAVE_INTERVAL} \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --hysteresis 2 \
        --num-workers 0 \
        --fp16 \
        --load ${CHECKPOINT_PATH} \
        --save ${CHECKPOINT_PATH} \
        --threshold ${THRESHOLD} \
        --gate-view-num ${GATE_VIEW_NUM} \
        --no-pipeline-parallel \
        --use-threshold \
        --sparse-mlp"

if [ "${ACTIVATION_CHECKPOINT}" = "true" ]; then
megatron_options="${megatron_options} \
        --checkpoint-activations"
fi

if [[ $EP_SIZE -gt 1 ]]; then
megatron_options="${megatron_options} \
        --create-moe-param-group"
fi

if [ "${MOE_DROP_TOKEN}" = "false" ]; then
megatron_options="${megatron_options} \
        --disable-moe-token-dropping"
fi

template_json="ds_config_gpt_TEMPLATE.json"
mkdir -p "config/"
config_json="config/ds_config_gpt_${NAME}.json"
sed "s/CONFIG_BATCH_SIZE/${GLOBAL_BATCH_SIZE}/" ${template_json} \
    | sed "s/CONFIG_MBSIZE/${BATCH_SIZE}/" \
    | sed "s/LOG_INTERVAL/${LOG_INTERVAL}/" \
    | sed "s/ZERO_STAGE/0/" \
    | sed "s/PRESCALE_GRAD/true/" \
    | sed "s/CONFIG_FP16_ENABLED/true/" \
    | sed "s/CONFIG_BF16_ENABLED/false/" \
    | sed "s/CONFIG_CL_ENABLED/${CL_ENABLED}/" \
    | sed "s/CONFIG_CL_MIN/${CL_START_SEQLEN}/" \
    | sed "s/CONFIG_CL_MAX/${SEQ_LEN}/" \
    | sed "s/CONFIG_CL_DURATION/${CL_STEP}/" \
      > ${config_json}

deepspeed_options=" \
            --deepspeed \
            --deepspeed_config ${config_json} \
            --pipeline-model-parallel-size ${PP_SIZE}"

# Currently MoE is not compatible with pipeline parallel
if [[ $EP_SIZE -gt 1 ]]; then
deepspeed_options="${deepspeed_options} \
        --no-pipeline-parallel"
fi

if [ "${ACTIVATION_CHECKPOINT}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
        --deepspeed-activation-checkpointing"
fi

## When saving checkpoint to a storage with cache, their could be consistency
## issue of the pointer to latest checkpoint. Here we find the correct pointer
## and broadcast it to all nodes.
ITERATION_FILE="$CHECKPOINT_PATH/latest_checkpointed_iteration.txt"
ITERATION_FILE_2="$CHECKPOINT_PATH/latest"
ITERATION=0
for (( node = 0; node <= NUM_NODE-1; node++ ))
do
    if $(ssh -q worker-"$node" "test -f \"$ITERATION_FILE\""); then
        LOCAL_ITERATION=$(ssh -q worker-"$node" cat $ITERATION_FILE)
        ITERATION=$(( ${LOCAL_ITERATION} > ${ITERATION} ? ${LOCAL_ITERATION} :  ${ITERATION} ))
    fi
done
if [[ $ITERATION -gt 0 ]]; then
    ITERATION_2="global_step${ITERATION}"
    ds_ssh "echo $ITERATION > $ITERATION_FILE"
    ds_ssh "echo $ITERATION_2 > $ITERATION_FILE_2"
fi

run_cmd="deepspeed --include localhost:0,1 --master_port 61500 ${DIR}/../../pretrain_gpt.py ${megatron_options} ${data_options} ${deepspeed_options} &> ${OUTPUT_BASEPATH}/log/${NAME}_${host}_${current_time}.log"
echo ${run_cmd}
eval ${run_cmd}
set +x

