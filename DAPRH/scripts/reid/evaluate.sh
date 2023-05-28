python test_model.py \
-dt "market1501" --data-dir "../datasets" \
-a resnet50 --features 0  -b 8 \
--resume "../saves/reid/duke2market/S1/R50Mix-4:1-lam/model_best.pth.tar" \
#--rerank
