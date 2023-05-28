Sample help better understand trt
## Env
```
docker run --name p2lr --runtime nvidia -dit --ipc host -v /mnt/sda1/hoangnt/p2lr/deepstream:/workspace nvcr.io/nvidia/pytorch:20.03-py3
```

## Build

```
mkdir build && cd build
cmake -DCMAKE_CUDA_FLAGS="--expt-extended-lambda -std=c++14" ..
make
```

## Check onnx to trt
```
cd /usr/src/tensorrt/bin
./trtexec --onnx=/workspace/weights/model701_checkpoint.onnx --verbose --explicitBatch
```

## Export

```
./export /workspace/weights/model701_checkpoint.onnx /workspace/weights/p2lr_r50_x86.plan
```

## Benchmark trt

```
cd /usr/src/tensorrt/bin
./trtexec --loadEngine=/workspace/weights/p2lr_r50_x86.plan --explicitBatch --shapes=input_0:4x3x128x256
```