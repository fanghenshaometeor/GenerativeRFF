# -------- training
CUDA_VISIBLE_DEVICES=0 python train_mnist.py --dataset 'mnist' \
                    --lr 9e-4 --D 16 8 --epochs 200 800 --wd 0 --print_freq 5 --num_repeat 1



# -------- attack
CUDA_VISIBLE_DEVICES=0 python attack_mnist.py --dataset 'mnist' --D 16 8