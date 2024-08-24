ulimit -n 4096
set -x

swapon --show
free -h
export NCCL_P2P_LEVEL=NVL
export OMP_NUM_THREADS=8

EXP_DIR=exps/nvi
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
        --nproc_per_node=4 \
        --master_port 29433 \
        --use_env \
        main.py \
        --output_dir ${EXP_DIR} \
        --data_path dataset/PIC_2.0 \
        --dataset_file nvi \
        --backbone resnet50 \
        --num_queries 64 \
        --dec_layers 3 \
        --num_workers 2 \
        --epochs 90 \
        --lr_drop 60 \
        --batch_size 4 \
        --use_nms_filter \
        --set_cost_bbox 2.5 \
        --set_cost_giou 1 \
        --bbox_loss_coef 2.5 \
        --giou_loss_coef 1 \
        --mimic_loss_coef 2 \
        --pretrained params/detr-r50-pre-2branch-vpic.pth \
        --n_layer 5 \





