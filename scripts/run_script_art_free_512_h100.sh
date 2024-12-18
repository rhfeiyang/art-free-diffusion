# Authors: Hui Ren (rhfeiyang.github.io)
# export CUDA_VISIBLE_DEVICES=0,1,2
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="whole_sam"
export machine_rank=${1:-0}
vae_path="rhfeiyang/art-free-diffusion-v1"
text_encoder_path="bert-base-uncased"
resume_checkpoint="latest"
resolution=512
workers=8
epochs=10000000
batch_size=90
p_uncond=0.1
learning_rate=5e-05
exp_name="${DATASET_NAME}_r${resolution}_p${p_uncond}_art_free_h100"
output_dir="experiments/${exp_name}"

num_gpus=$(( $(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c) + 1 ))
hostname=`hostname`
precision="bf16"
# config=./multi_node_config.yaml
config=./accelerator_config.yaml
CMD="accelerate launch --config_file ${config}  --mixed_precision ${precision} --multi_gpu --num_processes ${num_gpus} --machine_rank ${machine_rank} pretrain.py \
  --mixed_precision ${precision} \
  --pretrained_model_name_or_path $MODEL_NAME \
  --vae_path ${vae_path} \
  --text_encoder_path ${text_encoder_path} \
  --dataset_name sam_whole_filtered \
  --use_ema \
  --resolution ${resolution} --center_crop --random_flip \
  --train_batch_size ${batch_size} \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing \
  --num_train_epochs ${epochs} \
  --learning_rate ${learning_rate} \
  --max_grad_norm 1 \
  --lr_scheduler constant --lr_warmup_steps=0 \
  --p_uncond ${p_uncond} \
  --exp_name ${exp_name} \
  --output_dir ${output_dir} \
  --tracker_project_name wholeSAM_filtered_cleanModule_${resolution} \
  --dataloader_num_workers ${workers} \
  --resume_from_checkpoint ${resume_checkpoint} \
  --checkpoints_total_limit 10
  --validation_prompts \"A city street.\" \"a passenger bus that is driving down a street.\" \"A photo of a cat.\" \"A painting by Van Gogh\" \"teddy bear playing in the pool\"
  --fid
  --checkpointing_steps
  100
  --validation_epochs
  1
  "

eval mkdir -p ${output_dir}

CMD="nohup ${CMD} > ${output_dir}/${exp_name}_$(date +"%Y-%m-%d-%H:%M:%S")_${machine_rank}_${hostname}.log 2>&1 &"
echo ${CMD}

#eval conda activate diffusion
#eval cd ${dirname $0}

eval ${CMD}
