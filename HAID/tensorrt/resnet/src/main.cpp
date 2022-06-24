#include <NvInfer.h>

#include <onnx-tensorrt/NvOnnxParser.h>

#include <NvInferRuntime.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <math.h>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <functional>
#include <unistd.h>

#include <time.h>

#include <opencv2/opencv.hpp>

using namespace std;

#define checkRuntime(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line) {
	if (code != cudaSuccess) {
		const char* err_name = cudaGetErrorName(code);
		const char* err_message = cudaGetErrorString(code);
		printf("runtime error %s: %d  %s failed.\n  code = %s, message = %s", file, line, op, err_name, err_message);
		return false;
	}
	return true;
}

inline const char* severity_string(nvinfer1::ILogger::Severity t) {
	switch (t) {
		case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
		case nvinfer1::ILogger::Severity::kERROR: return "error";
		case nvinfer1::ILogger::Severity::kWARNING: return "warning";
		case nvinfer1::ILogger::Severity::kINFO: return "info";
		case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
		default: return "unknown";
	}
}

class TRTLogger : public nvinfer1::ILogger {
public:
	virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override {
		if (severity <= Severity::kWARNING) {
			if (severity == Severity::kWARNING) printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
			else if (severity == Severity::kERROR) printf("\031[33m%s: %s\033[0m\n", severity_string(severity), msg);
			else printf("%s: %s\n", severity_string(severity), msg);
		}
	}
} logger;


template<typename _T>
shared_ptr<_T> make_nvshared(_T *ptr) {
	return shared_ptr<_T>(ptr, [](_T* p){p->destroy();});
}

bool isFileExist(const string& path) {
	return access(path.c_str(), R_OK) == 0;
}

bool build_model(const string& model_name) {
	if (isFileExist( "resnet50.trtmodel" )) {
		printf("resnet50.trtmodel already exists.\n");
		return true;
	}

	TRTLogger logger;

	auto builder = make_nvshared(nvinfer1::createInferBuilder(logger));
	auto config = make_nvshared(builder->createBuilderConfig());
	auto network = make_nvshared(builder->createNetworkV2(1));

	auto parser = make_nvshared(nvonnxparser::createParser(*network, logger));
	if (!parser->parseFromFile(model_name.c_str(), 1)) {
		printf("Failed to parse %s\n", model_name.c_str());
		return false;
	}
	int maxBatchSize = 10;
	printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f);
	config->setMaxWorkspaceSize(1 << 28);

	auto profile = builder->createOptimizationProfile();
	auto input_tensor = network->getInput(0);
	auto input_dims = input_tensor->getDimensions();

	input_dims.d[0] = 1;
	profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
	profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);

	input_dims.d[0] = maxBatchSize;
	profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
	config->addOptimizationProfile(profile);

	auto engine = make_nvshared(builder->buildEngineWithConfig(*network, *config));
	if (engine == nullptr) {
		printf("Build engine failed.\n");
		return false;
	}

	auto model_data = make_nvshared(engine->serialize());
	FILE* f = fopen("resnet50.trtmodel", "wb");
	fwrite(model_data->data(), 1, model_data->size(), f);
	fclose(f);

	printf("Build Done.\n");
	return true;
}

vector<unsigned char> load_file(const string& file) {
	ifstream in(file, ios::in | ios::binary);
	if (!in.is_open()) return {};

	in.seekg(0, ios::end);
	size_t length = in.tellg();

	vector<uint8_t> data;
	if (length > 0) {
		in.seekg(0, ios::beg);
		data.resize(length);

		in.read((char*)&data[0], length);
	}
	in.close();
	return data;
}

vector<string> load_imagenet_labels(const string& file_path) {
	vector<string> lines;

	ifstream in(file_path, ios::in | ios::binary);
	if (!in.is_open()) {
		printf("open label file %s failed.\n", file_path.c_str());
		return lines;
	}

	string line;
	while (getline(in, line)) {
		lines.push_back(line);
	}
	
	in.close();
	return lines;
}


void inference(const string& image_path) {
	TRTLogger logger;
	auto engine_data = load_file("resnet50.trtmodel");
	auto runtime = make_nvshared(nvinfer1::createInferRuntime(logger));
	auto engine = make_nvshared(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
	if (engine == nullptr) {
		printf("Deserialize cuda engine failed.\n");
		runtime->destroy();
		return;
	}

	if (engine->getNbBindings() != 2) {
		printf("Must be single input, single Output, got %d output.\n", engine->getNbBindings() - 1);
		return;
	}

	cudaStream_t stream = nullptr;
	checkRuntime(cudaStreamCreate(&stream));
	auto execution_context = make_nvshared(engine->createExecutionContext());

	int input_batch = 1;
	int input_channel = 3;
	int input_height = 224;
	int input_width = 224;

	int input_numel = input_batch * input_channel * input_height * input_width;
	float* input_data_host = nullptr;
	float* input_data_device = nullptr;

	checkRuntime(cudaMallocHost(&input_data_host, input_numel * sizeof(float)));
	checkRuntime(cudaMalloc(&input_data_device, input_numel * sizeof(float)));

	auto image = cv::imread(image_path);
	float mean[] = {0.406, 0.456, 0.485};
	float std[] = {0.225, 0.224, 0.229};

	cv::resize(image, image, cv::Size(input_width, input_height));

	int image_area = image.cols * image.rows;
	unsigned char* pimage = image.data;
	float* phost_b = input_data_host + image_area * 0;
	float* phost_g = input_data_host + image_area * 1;
	float* phost_r = input_data_host + image_area * 2;
	for (int i=0; i<image_area; ++i, pimage += 3) {
		 *phost_r++ = (pimage[0] / 255.0f - mean[0]) / std[0];
		 *phost_g++ = (pimage[1] / 255.0f - mean[1]) / std[1];
		 *phost_b++ = (pimage[2] / 255.0f - mean[2]) / std[2];
	 }

	clock_t t0 = clock();
	checkRuntime(cudaMemcpyAsync(input_data_device, input_data_host, input_numel *sizeof(float), cudaMemcpyHostToDevice, stream));
	clock_t t1 = clock();
	printf("Pure inference time: %lfms\n", (double)(t1 - t0)/CLOCKS_PER_SEC * 1000);

	const int num_classes = 1000;
	float output_data_host[num_classes];
	float* output_data_device = nullptr;
	checkRuntime(cudaMalloc(&output_data_device, sizeof(output_data_host)));

	auto input_dims = engine->getBindingDimensions(0);
	input_dims.d[0] = input_batch;

	execution_context->setBindingDimensions(0, input_dims);
	float* bindings[] = {input_data_device, output_data_device};
	bool success = execution_context->enqueueV2((void**)bindings, stream, nullptr);
	 
	checkRuntime(cudaMemcpyAsync(output_data_host, output_data_device, sizeof(output_data_host), cudaMemcpyDeviceToHost, stream));
	checkRuntime(cudaStreamSynchronize(stream));

	float* prob = output_data_host;
	int predict_label = max_element(prob, prob + num_classes) - prob;
	vector<string> labels = load_imagenet_labels("labels.imagenet.txt");
	string predict_class_name = labels[predict_label];
	float conf = prob[predict_label];
	printf("Predicted indx: %d, class name: %s, confidence: %f\n", predict_label, predict_class_name.c_str(), conf);

	checkRuntime(cudaStreamDestroy(stream));
	checkRuntime(cudaFreeHost(input_data_host));
	checkRuntime(cudaFree(input_data_device));
	checkRuntime(cudaFree(output_data_device));
}

int main() {
	if (!build_model("resnet50.onnx")) printf("trt model building failed.\n");
	string image_path;
	while (true) {
		cout << "Please input image path: " << endl;
		cin >> image_path;
		if (!isFileExist(image_path)) {
			printf("%s not exists.\n", image_path.c_str());
			continue;
		}
		inference(image_path);
	}

	return 0;
}
