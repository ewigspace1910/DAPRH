python _finetune.py \
-dt unlabel -b 64 -j 2 --num-clusters 1500 \
-a resnet50part --features 2048 \
--lr 0.00035 --alpha 0.999 --soft-ce-weight 0.5 --soft-tri-weight 0.8 \
--epochs 40 --iters 800 --print-fred 100 \
--multiple_kmeans --fast-kmeans \
--data-dir ""\
--logs-dir ""\
--init-1 ""\
--init-2 ""\

--offline_test
