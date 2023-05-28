# Train StarGAN on custom datasets
LABEL_DIM=7
CROP_SIZE=256
IMG_SIZE=256
TRAIN_IMG_DIR="../../datasets/ReidGan/duke2mark/test"
BATCHSIZE=16
Lidt=0

python main.py --mode test --dataset RaFD --rafd_crop_size $CROP_SIZE --image_size $IMG_SIZE \
               --c_dim $LABEL_DIM --rafd_image_dir $TRAIN_IMG_DIR --batch_size $BATCHSIZE\
               --sample_dir ../../saves/Gan-duke2mark/samples \
               --log_dir ../../saves/Gan-duke2mark/logs \
               --model_save_dir ../../saves/Gan-duke2mark/models \
               --result_dir ../../saves/Gan-duke2mark/results \
               --lambda_idt $Lidt