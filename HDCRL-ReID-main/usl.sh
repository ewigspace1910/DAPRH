python train_usl.py -d market1501 --lr 0.00035 -b 256 --num-instance 16 --eps 0.5 --eval-step 10 --epochs 70 --step-size 30 --iters 100 --momentum 0.2 --logs logs/market --data-dir ""

python train_usl.py -d duke --lr 0.00035 -b 256 --num-instance 16 --eps 0.6 --eval-step 10 --epochs 70 --step-size 30 --iters 100 --momentum 0.2 --logs logs/duke --data-dir ""









