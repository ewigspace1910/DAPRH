##########DUKE2MARKET##############
python _target_finetune.py \
-dt "market1501" -b 32 --num-instances 16 --min-sample 8 \
-a resnet50mulpart --num-parts 2 \
--epochs 40 --iters 400 \
--logs-dir "../../saves/reid/trial/duke2market/0.5"   \
--init "../../saves/reid/duke2market/S1/R50Mix-4:1-lam/model_best.pth.tar"  \
--data-dir "/dis/DS/ducanh/person-reid/datasets" \
--pho 0.0   --uet-al 0.5  --ece 0.4 --etri 0.6 --ema 

# ###########MARKET2DUKE##############
# python _target_finetune.py \
# -dt "dukemtmc" -b 128 --num-instances 16 --min-sample 16 \
# -a resnet50mulpart --num-parts 2 \
# --epochs 40 --iters 400 \
# --logs-dir "../saves/reid/trial/market2duke/0.7"   \
# --init "../saves/reid/market2duke/S1/R50Mix-4:1-lam/model_best.pth.tar"  \
# --data-dir "../datasets" \
# --pho 0.0   --uet-al 0.7 --ece 0.4 --etri 0.6 --ema 

# ###########MARKET2MSMT##############
# python _target_finetune.py \
# -dt "msmt17" -b 128 --num-instances 16 --min-sample 16 \
# -a resnet50mulpart --num-parts 2 \
# --epochs 40 --iters 400 \
# --cluster-eps 0.6  \
# --logs-dir "../saves/reid/trial/market2msmt/0.2"   \
# --init "../saves/reid/market2msmt/S1/R50Mix-4:1-lam(re)/model_best.pth.tar"  \
# --data-dir "../datasets" \
# --pho 0.0   --uet-al 0.2 --ece 0.4 --etri 0.6 --ema 


# # ###########DUKE2MSMT##############
# python _target_finetune.py \
# -dt "msmt17" -b 128 --num-instances 16 --min-sample 16 \
# -a resnet50mulpart --num-parts 2 \
# --epochs 40 --iters 400 \
# --cluster-eps 0.6  \
# --logs-dir "../saves/reid/trial/duke2msmt/0.5"   \
# --init "../saves/reid/duke2msmt/S1/R50Mix-4:1-lam/model_best.pth.tar"  \
# --data-dir "../datasets" \
# --pho 0.0   --uet-al 0.5 --ece 0.4 --etri 0.6 --ema 