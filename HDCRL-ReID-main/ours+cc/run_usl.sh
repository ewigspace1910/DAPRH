python cluster_contrast_train_usl.py -d market1501 -b 256 -- num-instance 16 --eps 0.4 --epochs 120 --step-size 40 --eval-step 10 --iters 200 --pooling-type "avg" --use-hard --logs logs/market1501 --data-dir " "

python cluster_contrast_train_usl.py -d duke -b 256 -- num-instance 16 --eps 0.7 --epochs 120 --step-size 40 --eval-step 10 --iters 200 --pooling-type "avg" --use-hard --logs logs/duke --data-dir " "
