python train_uda.py -ds duke -dt market1501 -b 256 --lr 0.00035 --epochs 70 --step-size 30 --iters 100 --num-instance 16 --momentum 0.2 --eval-step 10 --eps 0.5 --logs logs/d2m --data-dir " "

python train_uda.py -ds market1501 -dt duke -b 256 --lr 0.00035 --epochs 70 --step-size 30 --iters 100 --num-instance 16 --momentum 0.2 --eval-step 10 --eps 0.6 --logs logs/m2d --data-dir " "










