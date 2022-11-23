###################################################################################################
###################################################################################################
###################################################################################################
# ======== SYNTHETIC data ======
# ======== data dim. 80 ========
# ======== single-layer ========
CUDA_VISIBLE_DEVICES=1 python train.py --dataset 'synthetic' --num_dim 80 \
                    --num_layers 1 --D 256 --lr 1e-3 --epochs 1000 --wd 0 --print_freq 100
# ======== 2 layers ============
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset 'synthetic' --num_dim 80 \
#                     --num_layers 2 --D 256 64 --lr 1e-3 --epochs 200 1000 --wd 0 --print_freq 100
# ======== 3 layers ============
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset 'synthetic' --num_dim 80 \
#                     --num_layers 3 --D 64 64 64 --lr 2e-4 --epochs 200 200 1000 --wd 0 --print_freq 100
# ======== 4 layers ============
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset 'synthetic' --num_dim 80 \
#                     --num_layers 4 --D 64 64 64 64 --lr 1e-4 --epochs 200 200 200 1000 --wd 0 --print_freq 100

# ======== data dim. 100 =======
# ======== single-layer ========
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset 'synthetic' --num_dim 100 \
#                     --num_layers 1 --D 256 --lr 1e-3 --epochs 1000 --wd 0 --print_freq 100
# ======== 2 layers ============
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset 'synthetic' --num_dim 100 \
#                     --num_layers 2 --D 256 64 --lr 1e-3 --epochs 200 1000 --wd 0 --print_freq 100
# ======== 3 layers ============
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset 'synthetic' --num_dim 100 \
#                     --num_layers 3 --D 64 64 64 --lr 2e-4 --epochs 200 200 1000 --wd 0 --print_freq 100
# ======== 4 layers ============
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset 'synthetic' --num_dim 100 \
#                     --num_layers 4 --D 64 64 64 64 --lr 1e-4 --epochs 200 200 200 1000 --wd 0 --print_freq 100

###################################################################################################
###################################################################################################
###################################################################################################
# ======== BENCHMARK data ========
# ======== austra ========
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset 'austra' --lr 1e-4 --epochs 800 1500 --wd 0 --print_freq 500

# ======== climate =======
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset 'climate' --lr 1e-4 --epochs 1000 1500 --wd 0 --print_freq 500

# ======== diabetic ======
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset 'diabetic' --lr 2e-3 --epochs 5000 1500 --wd 1e-5 --print_freq 500

# ======== sonar ========
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset 'sonar' --lr 5e-4 --epochs 1000 1500 --wd 0 --print_freq 500

# ======== Adult ========
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset 'adult' --lr 1e-3 --epochs 50 80 --wd 5e-3

# ======== Ijcnn ========
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset 'ijcnn' --lr 9e-4 --epochs 50 80 --wd 0

# ======== Phishing =====
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset 'phishing' --lr 0.002 --epochs 50 100 --wd 0

