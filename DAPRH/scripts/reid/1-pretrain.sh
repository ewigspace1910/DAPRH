# python examples/pretrain/_source_pretrain_mix.py \
#     -ds "dukemtmc" -dt "market1501" \
#     -a "resnet50" --feature 0 --iters 200 --print-freq 100\
# 	--num-instances 8 -b 128 -j 4 --seed 123 --margin 0.3 \
#     --warmup-step 10 --lr 0.00035 --milestones 40 70  --epochs 80 --eval-step 1 \
# 	--logs-dir "../saves/reid/duke2market/S1/R50Mix-4:1-lam"  \
#     --data-dir "../datasets" \
#     --fake-data-dir "../datasets/SystheImgs"   \
#     --ratio 4 1 \
#     --dim --lamda 0.05

# python examples/pretrain/_source_pretrain_mix.py \
#     -dt "dukemtmc" -ds "market1501" \
#     -a "resnet50" --feature 0 --iters 200 --print-freq 100\
# 	--num-instances 8 -b 64 -j 4 --seed 123 --margin 0.3 \
#     --warmup-step 128 --lr 0.00035 --milestones 40 70  --epochs 80 --eval-step 1 \
# 	--logs-dir "../saves/reid/market2duke/S1/R50Mix-4:1-lam"  \
#     --data-dir "../datasets" \
#     --fake-data-dir "../datasets/SystheImgs"   \
#     --ratio 4 1 \
#     --dim --lamda 0.05

# python examples/pretrain/_source_pretrain_mix.py \
#     -dt "msmt17" -ds "market1501" \
#     -a "resnet50" --feature 0 --iters 200 --print-freq 100\
# 	--num-instances 4 -b 64 -j 4 --seed 123 --margin 0.3 \
#     --warmup-step 128 --lr 0.00035 --milestones 40 70  --epochs 80 --eval-step 1 \
# 	--logs-dir "../saves/reid/market2msmt/S1/R50Mix-4:1-lam(re)"  \
#     --data-dir "../datasets" \
#     --fake-data-dir "../datasets/SystheImgs"   \
#     --ratio 4 1 \
#     --dim --lamda 0.05

python examples/pretrain/_source_pretrain_mix.py \
    -dt "msmt17" -ds "dukemtmc" \
    -a "resnet50" --feature 0 --iters 200 --print-freq 100\
	--num-instances 4 -b 64 -j 4 --seed 123 --margin 0.3 \
    --warmup-step 128 --lr 0.00035 --milestones 40 70  --epochs 80 --eval-step 1 \
	--logs-dir "../saves/reid/duke2msmt/S1/R50Mix-4:1-lam"  \
    --data-dir "../datasets" \
    --fake-data-dir "../datasets/SystheImgs"   \
    --ratio 4 1 \
    --dim --lamda 0.05
