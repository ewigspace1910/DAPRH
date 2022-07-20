#include <vector>
#include <iostream>
#include <fstream>

#include "engine.h"

using namespace std;

int main(int argc, char *argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " core_model.onnx engine.plan" << endl;
    }

	ifstream onnxFile;
	onnxFile.open(argv[1], ios::in | ios::binary);
	cout << "Load model from " << argv[1] << endl;

    if (!onnxFile.good()) {
		cerr << "\nERROR: Unable to read specified ONNX model " << argv[1] << endl;
		return -1;
	}

	onnxFile.seekg (0, onnxFile.end);
	size_t size = onnxFile.tellg();
	onnxFile.seekg (0, onnxFile.beg);

	auto *buffer = new char[size];
	onnxFile.read(buffer, size);
	onnxFile.close();

	bool verbose = true;
	size_t workspace_size =(1ULL << 30);
	const vector<int> dynamic_batch_opts{1, 8, 16};

	cout << "Building engine..." << endl;
	auto engine = sample_onnx::Engine(buffer, size, dynamic_batch_opts, verbose, workspace_size);
	engine.save(string(argv[2]));
	
	delete [] buffer;
	return 0;
}