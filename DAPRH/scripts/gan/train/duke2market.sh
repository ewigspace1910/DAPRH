# Train StarGAN on custom datasets
LABEL_DIM=7
CROP_SIZE=128
IMG_SIZE=128
TRAIN_IMG_DIR="../../datasets/ReidGan/duke2mark/train"
BATCHSIZE=16
Lidt=0
Lrec=10
Lgp=10
Lcls=1

python main.py --mode train --dataset RaFD --rafd_crop_size $CROP_SIZE --image_size $IMG_SIZE \
               --c_dim $LABEL_DIM --rafd_image_dir $TRAIN_IMG_DIR --batch_size $BATCHSIZE\
               --sample_dir ../../saves/Gan-duke2mark/samples \
               --log_dir ../../saves/Gan-duke2mark/logs \
               --model_save_dir ../../saves/Gan-duke2mark/models \
               --result_dir ../../saves/Gan-duke2mark/results \
               --lambda_idt $Lidt \
               --lambda_rec $Lrec \
               --lambda_gp $Lgp --lambda_cls $Lcls