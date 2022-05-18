dataset='monks1'
data_folder='/home/dev/fangkun/data/UCI/monks1'

# dataset='monks2'
# data_folder='/home/dev/fangkun/data/UCI/monks2'

# dataset='monks3'
# data_folder='/home/dev/fangkun/data/UCI/monks3'

# dataset='australian'
# data_folder='/home/dev/fangkun/data/UCI/austra'

# dataset='climate'
# data_folder='/home/dev/fangkun/data/UCI/climate'

# dataset='diabetic'
# data_folder='/home/dev/fangkun/data/UCI/diabetic'

# dataset='sonar'
# data_folder='/home/dev/fangkun/data/UCI/sonar'

# dataset='adult'
# data_folder='/home/dev/fangkun/data/UCI/adult'

# dataset='ijcnn'
# data_folder='/home/dev/fangkun/data/UCI/ijcnn'

# dataset='phishing'
# data_folder='/home/dev/fangkun/data/UCI/phishing'

CUDA_VISIBLE_DEVICES=0 python train_benchmark.py \
    --dataset ${dataset} \
    --data_folder ${data_folder}