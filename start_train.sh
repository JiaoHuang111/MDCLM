export CUDA_VISIBLE_DEVICES=XXXX

export PROJECT_ROOT=/XXX/dmlm

# export HF_ENDPOINT="https://hf-mirror.com"

export PYTHONPATH="/XXX/dmlm/src:$PYTHONPATH"

export PYTHONPATH=$PYTHONPATH:/XXX/dmlm/vendor/openfold


max_tokens=8192
accumulate_grad_batches=1

exp=dmlm/dmlm_base
model_name=dmlm_650m

python train.py \
    experiment=${exp} name=${model_name} \
    datamodule.max_tokens=${max_tokens} \
    trainer.accumulate_grad_batches=${accumulate_grad_batches}\
    2>&1 | tee debug_log.txt
