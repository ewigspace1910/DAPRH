import tensorrt as trt

print("||||====Test trt====||||")
model_path = "../weights/model701_checkpoint.onnx"
G_LOGGER = trt.Logger(trt.Logger.WARNING)
	
with trt.Builder(G_LOGGER) as builder:
	builder.max_batch_size = 16
	builder.max_workspace_size = 1 << 20

	explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
	network = builder.create_network(explicit_batch)
	parser = trt.OnnxParser(network, G_LOGGER)

	with open(model_path, 'rb') as model:
		print('Beginning ONNX file parsing')
		parser.parse(model.read())
	
	print("====ERROR====")
	for index in range(parser.num_errors):
		print(parser.get_error(index))