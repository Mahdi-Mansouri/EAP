# FIXED: Updated paths to point to your local 'models/erase' folder
export HF_HOME=/home/mehdi/hf_cache_local
python -u train_adversarial_gumbel.py \
    --save_freq 50 \
    --models_path=models \
    --prompt 'nudity' \
    --seperator ',' \
    --train_method 'noxattn' \
    --config_path 'configs/stable-diffusion/v1-inference.yaml' \
    --ckpt_path 'models/erase/sd-v1-4-full-ema.ckpt' \
    --diffusers_config_path 'models/erase/config.json' \
    --lr 1e-5 \
    --gumbel_lr 1e-2 \
    --gumbel_temp 2 \
    --gumbel_hard 1 \
    --gumbel_num_centers 50 \
    --gumbel_update -1 \
    --gumbel_time_step 0 \
    --gumbel_multi_steps 2 \
    --gumbel_k_closest 2000 \
    --vocab 'EN3K' \
    --info 'gumbel_lr_1e-2_temp_2_hard_1_num_50_update_-1_timestep_0_multi_2_kclosest_2000_EN3K'

# Define your absolute path for the evaluation models
export MODELS_PATH='/home/mehdi/EAP/Erasing-Adversarial-Preservation/models'

# Evaluation Generation 1 (Nudity CSV)
python -u eval-scripts/generate-images.py  \
    --models_path=$MODELS_PATH \
    --model_name='compvis-adversarial-gumbel-word_nudity-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_num_50_update_-1_timestep_0_multi_2_kclosest_2000_EN3K' \
    --prompts_path 'data/nudity.csv' \
    --save_path 'evaluation_folder' \
    --num_samples 1 

# Evaluation Generation 2 (Unsafe Prompts)
python -u eval-scripts/generate-images.py  \
    --models_path=$MODELS_PATH \
    --model_name='compvis-adversarial-gumbel-word_nudity-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_num_50_update_-1_timestep_0_multi_2_kclosest_2000_EN3K' \
    --prompts_path 'data/unsafe-prompts4703.csv' \
    --save_path 'evaluation_massive' \
    --num_samples 1

# Evaluation Classification
export FOLDER='evaluation_massive/compvis-adversarial-gumbel-word_nudity-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_num_50_update_-1_timestep_0_multi_2_kclosest_2000_EN3K/unsafe-prompts4703'
export PROMPTS_PATH='data/unsafe-prompts4703.csv'
export SAVE_PATH='evaluation_folder/unsafe/compvis-adversarial-gumbel-word_nudity-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_num_50_update_-1_timestep_0_multi_2_kclosest_2000_EN3K-data-unsafe.csv'
python -u eval-scripts/nudenet-classes.py --threshold 0.0 --folder=$FOLDER --prompts_path=$PROMPTS_PATH --save_path=$SAVE_PATH

# COCO-30k Generation
python -u eval-scripts/generate-images.py  \
    --models_path=models \
    --model_name='adversarial-gumbel-word_nudity-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_num_50_update_-1_timestep_0_multi_2_kclosest_2000_EN3K' \
    --prompts_path 'data/coco_30k.csv' \
    --save_path 'evaluation_massive' \
    --num_samples 1 \
    --from_case 0