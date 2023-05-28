# # Train StarGAN on custom datasets
# LABEL_DIM=9
# CROP_SIZE=128
# IMG_SIZE=128
# TRAIN_IMG_DIR="/home/k64t/person-reid/datasets/4Gan/mark2duke/train/market1501"
# FAKEDIR="../../datasets/SyntheImgs/mark2duke-results"
# BATCHSIZE=1
# ITER=190000
# DOMAIN=8
# PROB=0.8 #save 80% images

# python main.py --mode sample --dataset RaFD --rafd_crop_size $CROP_SIZE --image_size $IMG_SIZE \
#                --c_dim $LABEL_DIM --rafd_image_dir $TRAIN_IMG_DIR --batch_size $BATCHSIZE\
#                --sample_dir ../../saves/Gan-mark2duke/samples \
#                --log_dir ../../saves/Gan-mark2duke/logs \
#                --model_save_dir ../../saves/Gan-mark2duke/models \
#                --result_dir ../../saves/Gan-mark2duke/results \
#                --test_iters $ITER --except_domain=$DOMAIN \
#                --pattern "{ID}_{CX}_f{RANDOM}.jpg" \
#                --gen_dir $FAKEDIR \
#                --prob $PROB

# Train StarGAN on custom datasets
# LABEL_DIM=7
# CROP_SIZE=128
# IMG_SIZE=128
# TRAIN_IMG_DIR="../../datasets/SyntheImgs/duke2mark/train/dukemtmc"
# FAKEDIR="../../datasets/SyntheImgs/duke2mark-results"
# BATCHSIZE=1
# ITER=200000
# DOMAIN=0

# python main.py --mode sample --dataset RaFD --rafd_crop_size $CROP_SIZE --image_size $IMG_SIZE \
#                --c_dim $LABEL_DIM --rafd_image_dir $TRAIN_IMG_DIR --batch_size $BATCHSIZE\
#                --sample_dir ../../saves/Gan-duke2mark/samples \
#                --log_dir ../../saves/Gan-duke2mark/logs \
#                --model_save_dir ../../saves/Gan-duke2mark/models \
#                --result_dir ../../saves/Gan-duke2mark/results \
#                --test_iters $ITER --except_domain=$DOMAIN \
#                --pattern "{ID}_{CX}_f{RANDOM}.jpg" \
#                --gen_dir $FAKEDIR \
#                --prob $PROB 

# Train StarGAN on custom datasets
# LABEL_DIM=16
# CROP_SIZE=128
# IMG_SIZE=128
# TRAIN_IMG_DIR="/home/k64t/person-reid/datasets/4Gan/mark2msmt/train/market1501"
# FAKEDIR="../../datasets/SyntheImgs/mark2msmt-results"
# BATCHSIZE=1
# ITER=200000
# DOMAIN=0
# PROB=1 #save 80% images

# python main.py --mode sample --dataset RaFD --rafd_crop_size $CROP_SIZE --image_size $IMG_SIZE \
#                --c_dim $LABEL_DIM --rafd_image_dir $TRAIN_IMG_DIR --batch_size $BATCHSIZE\
#                --sample_dir ../../saves/Gan-mark2msmt/samples \
#                --log_dir ../../saves/Gan-mark2msmt/logs \
#                --model_save_dir ../../saves/Gan-mark2msmt/models \
#                --result_dir ../../saves/Gan-mark2msmt/results \
#                --test_iters $ITER --except_domain=$DOMAIN \
#                --pattern "{ID}_{CX}_f{RANDOM}.jpg" \
#                --gen_dir $FAKEDIR \
#                --prob $PROB

LABEL_DIM=16
CROP_SIZE=128
IMG_SIZE=128
TRAIN_IMG_DIR="/home/k64t/person-reid/datasets/4Gan/duke2msmt/train/dukemtmc"
FAKEDIR="../../datasets/SyntheImgs/duke2msmt-results"
BATCHSIZE=1
ITER=190001
DOMAIN=0
PROB=0.8 #save 80% images

python main.py --mode sample --dataset RaFD --rafd_crop_size $CROP_SIZE --image_size $IMG_SIZE \
               --c_dim $LABEL_DIM --rafd_image_dir $TRAIN_IMG_DIR --batch_size $BATCHSIZE\
               --sample_dir ../../saves/Gan-duke2msmt/samples \
               --log_dir ../../saves/Gan-duke2msmt/logs \
               --model_save_dir ../../saves/Gan-duke2msmt/models \
               --result_dir ../../saves/Gan-duke2msmt/results \
               --test_iters $ITER --except_domain=$DOMAIN \
               --pattern "{ID}_{CX}_f{RANDOM}.jpg" \
               --gen_dir $FAKEDIR \
               --prob $PROB