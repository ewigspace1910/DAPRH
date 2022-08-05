import onnxruntime
import onnx
import numpy as np
import tensorrt as trt


model_path = "../weights/model701_checkpoint.onnx"
onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)

batch_size = 1
image_shape = [256, 128]
image_channel = 3
input_shape = [batch_size, image_channel, *image_shape]
fake_input = np.random.random(input_shape).astype(np.float32)

print("||||====Test onnx====||||")
session = onnxruntime.InferenceSession(model_path)
print("====INPUT====")
for i in session.get_inputs():
    print("Name: {}, Shape: {}, Dtype: {}".format(i.name, i.shape, i.type))
print("====OUTPUT====")
for i in session.get_outputs():
    print("Name: {}, Shape: {}, Dtype: {}".format(i.name, i.shape, i.type))
print("====INFER====")
outputs = session.run(None, {'input_0': fake_input})[0]
print("Shape: {}".format(outputs.shape))