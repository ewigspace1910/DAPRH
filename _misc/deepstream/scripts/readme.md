require:
- pytorch: 1.12
- tensorrt: 7.0.0

### Export model
python export_onnx.py -a resnet50 --resume /mnt/sda1/p2lr/deepstream/weights/model701_checkpoint.pth.tar