#!/bin/bash

# ============== Arguments ==============
gpu=$1
MODE=$2

# ============== Hyperparameters ==============

# Shared
cfg_mld="configs/ft_mld_t2m.yaml"
cfg_spm="configs/spm_kit.yaml"
spm_path="FT-[TFS_Pured_T1000_M0T0_E200]-_Mixed_T1000_M1T0_E14_1e-2.pth"

# eval: MLD checkpoint to evaluate
eval_ckpt="checkpoints/ft_mld_x0reward/FT-[TFS_Pured_T1000_M0T0_E200]-_Mixed_T1000_M1T0_E14_1e-2/NIPS_K10_R1e00_FT-[TFS_Pured_T1000_M0T0_E200]-_Mixed_T1000_M1T0_E14_1e-2/checkpoints/E9-R1-0.578-FID-0.217.ckpt"

# mld: fine-tuning hyperparameters
ft_type="ReFL"
ft_m=20
ft_lambda_reward=1.0

# spm: training hyperparameters
noise_thr=1.1
max_T=1000 
step_aware="M1T0"
spm_finetune="TFS_KIT_Pured_T1000_M1T0_E100.pth"

# ============== Run ==============

if [ "$MODE" == "eval" ]; then
    CUDA_VISIBLE_DEVICES=${gpu} python -m test \
        --cfg ${cfg_mld} \
        --mld_path ${eval_ckpt}

elif [ "$MODE" == "mld" ]; then
    CUDA_VISIBLE_DEVICES=${gpu} python -m ft_mld \
        --cfg ${cfg_mld} \
        --spm_path ${spm_path} \
        --ft_type ${ft_type} \
        --ft_m ${ft_m} \
        --ft_lambda_reward ${ft_lambda_reward}

    CUDA_VISIBLE_DEVICES=${gpu} python -m ft_mld_chain \
        --cfg ${cfg_mld} \
        --spm_path ${spm_path} \
        --ft_type ${ft_type} \
        --ft_m ${ft_m} \
        --ft_lambda_reward ${ft_lambda_reward}

elif [ "$MODE" == "spm" ]; then
    CUDA_VISIBLE_DEVICES=${gpu} python -m train_spm \
        --cfg ${cfg_spm} \
        --NoiseThr ${noise_thr} \
        --maxT ${max_T} \
        --step_aware ${step_aware} \
        --finetune ${spm_finetune} \
        | tee -a kit_tmr.log

elif [ "$MODE" == "tmr" ]; then
    CUDA_VISIBLE_DEVICES=${gpu} python -m eval_tmr \
        --cfg ${cfg_mld} \
        | tee -a tmr.log

else
    echo "Invalid MODE. Please choose from: eval, mld, spm, tmr"
fi
