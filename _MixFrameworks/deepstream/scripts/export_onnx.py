from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import io
import numpy as np
import os
import sys
sys.path.insert(0, "../../")

import torch
from torch import nn

from modules import models
from modules.utils.data import transforms as T
from modules.utils.serialization import load_checkpoint, copy_state_dict
from modules.feature_extraction import extract_cnn_feature
from modules.utils import to_torch

def get_args():
	parser = argparse.ArgumentParser(description="Testing the model")
	# data
	parser.add_argument('-b', '--batch-size', type=int, default=256)
	parser.add_argument('-j', '--workers', type=int, default=4)
	parser.add_argument('--height', type=int, default=256, help="input height")
	parser.add_argument('--width', type=int, default=128, help="input width")
	# model
	parser.add_argument('-a', '--arch', type=str, required=True,
						choices=models.names())
	parser.add_argument('--features', type=int, default=0)
	parser.add_argument('--dropout', type=float, default=0)
	# testing configs
	parser.add_argument('--resume', type=str, required=True, metavar='PATH')
	parser.add_argument('--rerank', action='store_true', help="evaluation only")
	parser.add_argument('--kmean', action='store_true', help="evaluation only")
	parser.add_argument('--clusters', type=int, default=0)
	parser.add_argument('--seed', type=int, default=1)
	# path
	working_dir = osp.dirname(osp.abspath(__file__))
	parser.add_argument('--data-dir', type=str, metavar='PATH',
						default=osp.join(working_dir, 'data'))

	return parser.parse_args()

args = get_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Create model
model = models.create(args.arch, pretrained=False, num_features=args.features, dropout=args.dropout, num_classes=0, is_export=True)
# model.cuda()
model.eval()
model = nn.DataParallel(model)
print(model)

# # Load from checkpoint
checkpoint = load_checkpoint(args.resume)
copy_state_dict(checkpoint['state_dict'], model)

batch_size = 1
image_shape = [args.height, args.width] #HxW
image_channel = 3
input_shape = [batch_size, image_channel, *image_shape]
fake_input = np.random.random(input_shape).astype(np.float32)
print(fake_input.shape)

# inputs = to_torch(fake_input).cuda()
# outputs = model(inputs)
# print(outputs.shape)

output_onnx = args.resume.replace("pth.tar", "onnx")
input_names = ['input_0']
output_names = ['output_0']

onnx_bytes = io.BytesIO()
zero_input = torch.zeros(input_shape)
zero_input = zero_input.to(device)
dynamic_axes = {input_names[0]: {0:'batch'}}
for _, name in enumerate(output_names):
	dynamic_axes[name] = dynamic_axes[input_names[0]]
extra_args = {'opset_version': 10, 'verbose': False,
				'input_names': input_names, 'output_names': output_names,
				'dynamic_axes': dynamic_axes}
torch.onnx.export(model.module, zero_input, onnx_bytes, **extra_args)
with open(output_onnx, 'wb') as out:
	out.write(onnx_bytes.getvalue())

print("==> Exporting model to ONNX format at '{}'".format(output_onnx))

# import onnx
# from onnxsim import simplify

# use onnxsimplify to reduce reduent model.
# onnx_model = onnx.load(output_onnx)
# model_simp, check = simplify(onnx_model)
# assert check, "Simplified ONNX model could not be validated"
# onnx.save(model_simp, output_onnx)