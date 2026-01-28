export CUDA_VISIBLE_DEVICES=0,1,2,3,4

export PROJECT_ROOT=/data/huangjiao/dmlm

export HF_ENDPOINT="https://hf-mirror.com"

export PYTHONPATH="/data/huangjiao/dmlm/src:$PYTHONPATH"

export PYTHONPATH=$PYTHONPATH:/data/huangjiao/dmlm/vendor/openfold


# 单个 batch 中的最大 token 数（输入序列长度上限）
max_tokens=8192

# 梯度累计的步数（相当于在 16 个小 batch 后再做一次反向传播）
# 实际有效 batch size = GPU 数量(8) × max_tokens(8192) × accumulate_grad_batches(16) ≈ 1百万 token
accumulate_grad_batches=1
# this means the effective batch size is #GPUs(8) * max_tokens(8192) * accumulate_grad_batches(16), resulting in approximately 1 million.

exp=dmlm/dmlm_base
model_name=dmlm_650m

# 启动训练脚本 train.py，同时传入实验名、模型名、最大 token 数和梯度累计步数等参数
python train.py \
    experiment=${exp} name=${model_name} \
    datamodule.max_tokens=${max_tokens} \
    trainer.accumulate_grad_batches=${accumulate_grad_batches}\
    2>&1 | tee debug_log.txt
