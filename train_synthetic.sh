# ======== data dim. 80 ========
# ======== single-layer ========
# CUDA_VISIBLE_DEVICES=1 python train_synthetic.py --dataset 'synthetic' --num_dim 80 \
#                     --num_layers 1 --D 256 --lr 1e-3 --epochs 1000 --wd 0 --print_freq 100
# ======== 2 layers ============
# CUDA_VISIBLE_DEVICES=1 python train_synthetic.py --dataset 'synthetic' --num_dim 80 \
#                     --num_layers 2 --D 256 64 --lr 1e-3 --epochs 200 1000 --wd 0 --print_freq 100
# ======== 3 layers ============
# CUDA_VISIBLE_DEVICES=1 python train_synthetic.py --dataset 'synthetic' --num_dim 80 \
#                     --num_layers 3 --D 64 64 64 --lr 2e-4 --epochs 200 200 1000 --wd 0 --print_freq 100
# ======== 4 layers ============
# CUDA_VISIBLE_DEVICES=1 python train_synthetic.py --dataset 'synthetic' --num_dim 80 \
#                     --num_layers 4 --D 64 64 64 64 --lr 1e-4 --epochs 200 200 200 1000 --wd 0 --print_freq 100
# ======== 5 layers ============
# CUDA_VISIBLE_DEVICES=1 python train_synthetic.py --dataset 'synthetic' --num_dim 80 \
#                     --num_layers 5 --D 64 64 64 64 64 --lr 2e-4 --epochs 200 200 200 200 1000 --wd 0 --print_freq 100

# ======== data dim. 100 =======
# ======== single-layer ========
# CUDA_VISIBLE_DEVICES=1 python train_synthetic.py --dataset 'synthetic' --num_dim 100 \
#                     --num_layers 1 --D 256 --lr 1e-3 --epochs 1000 --wd 0 --print_freq 100
# ======== 2 layers ============
# CUDA_VISIBLE_DEVICES=1 python train_synthetic.py --dataset 'synthetic' --num_dim 100 \
#                     --num_layers 2 --D 256 64 --lr 1e-3 --epochs 200 1000 --wd 0 --print_freq 100
# ======== 3 layers ============
# CUDA_VISIBLE_DEVICES=1 python train_synthetic.py --dataset 'synthetic' --num_dim 100 \
#                     --num_layers 3 --D 64 64 64 --lr 2e-4 --epochs 200 200 1000 --wd 0 --print_freq 100
# ======== 4 layers ============
# CUDA_VISIBLE_DEVICES=1 python train_synthetic.py --dataset 'synthetic' --num_dim 100 \
#                     --num_layers 4 --D 64 64 64 64 --lr 1e-4 --epochs 200 200 200 1000 --wd 0 --print_freq 100
# ======== 5 layers ============
CUDA_VISIBLE_DEVICES=1 python train_synthetic.py --dataset 'synthetic' --num_dim 100 \
                    --num_layers 5 --D 64 64 64 64 64 --lr 2e-4 --epochs 200 200 200 200 1000 --wd 0 --print_freq 100