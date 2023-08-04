#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_err.h"
#include "esp_log.h"

#include "main_functions.h"
#include "IO_manager.h"
#include "model_info.h"
#include "model_provider.h"
#include "input_provider.h"
#include "output_handler.h"
#include "wifi_manager.h"


namespace {
	const int SERIAL_WAIT_MS = 500;
	const char* TAG_MAIN = "main";
	const char* TAG_PERF = "perf";

	IOManager io_manager;
	ModelProvider model_provider(io_manager);
	InputProvider input_provider(io_manager);

	QuantType quant_type;
	alignas(8) unsigned char* model_data = nullptr;
	const tflite::Model* model = nullptr;
	tflite::MicroInterpreter* interpreter = nullptr;
	TfLiteTensor* input = nullptr;
	constexpr int tensor_arena_size = 50 * 1024;
	static uint8_t* tensor_arena = nullptr;
	int max_score_idx;
	char actual[20];
	char runtime_stats[400];
	esp_err_t inputs_stat;
}

void setup() {
	// Initialise Serial IO, File IO, GPIO, and wait till WiFi is connected
	ESP_ERROR_CHECK(esp_register_shutdown_handler(&io_manager.TerminateIO));
	WiFiWaitConnected();
	io_manager.InitializeIO();

	// Update and load model
	if (WIFI_CONNECTED) {
		if (model_provider.UpdateModel() != ESP_OK) {
			ESP_LOGW(TAG_MAIN, "Failed to update model from server");
		}
	} else {
		ESP_LOGW(TAG_MAIN, "WiFi not connected, model update not attempted");
	}
	if (model_provider.LoadModel() != ESP_OK) {
		ESP_LOGE(TAG_MAIN, "Failed to load model");
		esp_restart();
	}
	model_data = model_provider.GetModelData();
	quant_type = model_provider.GetQuantType();
	model = tflite::GetModel(model_data);
	if (model->version() != TFLITE_SCHEMA_VERSION) {
		ESP_LOGE(
			TAG_MAIN, "Model provided is schema version %lu not equal to supported version %d", 
			model->version(), 
			TFLITE_SCHEMA_VERSION
		);
		esp_restart();
	}
	
	// Initialize interpreter
    if (tensor_arena == nullptr) {
        tensor_arena = (uint8_t*) heap_caps_malloc(tensor_arena_size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    }
    if (tensor_arena == nullptr) {
        ESP_LOGE(TAG_MAIN, "Couldn't allocate %d bytes for model", tensor_arena_size);
        esp_restart();
    }
    static tflite::MicroMutableOpResolver<11> resolver;
	resolver.AddConv2D();
	resolver.AddDepthwiseConv2D();
	resolver.AddMaxPool2D();
	resolver.AddFullyConnected();
	resolver.AddPack();
	resolver.AddUnpack();
	resolver.AddReshape();
	resolver.AddShape();
	resolver.AddStridedSlice();
	resolver.AddRelu();
	resolver.AddSoftmax();
	static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, tensor_arena_size);
	interpreter = &static_interpreter;
	TfLiteStatus allocate_status = interpreter->AllocateTensors();
	if (allocate_status != kTfLiteOk) {
		ESP_LOGE(TAG_MAIN, "AllocateTensors() failed");
		ESP_LOGE(TAG_MAIN, "Failed to initialise interpreter");
		esp_restart();
	}

	// Obtain pointers to the model's input tensor
	input = interpreter->input(0);
}

void loop() {
	switch (quant_type) {
		case QuantType::INT_8:
			inputs_stat = input_provider.GetFileInput(
				tflite::GetTensorData<int8_t>(input), 
				actual, 
				input->params.scale, 
				input->params.zero_point
			);
			break;
		case QuantType::NO_QUANT:
		case QuantType::DYNAMIC:
			inputs_stat = input_provider.GetFileInput(
				tflite::GetTensorData<float>(input),
				actual
			);
			break;
		default:
			inputs_stat = ESP_ERR_NOT_SUPPORTED;
			ESP_LOGE(TAG_MAIN, "Invalid model quantization type");
	}

	if (inputs_stat == ESP_OK) {
		ESP_LOGI(TAG_MAIN, "Actual: %s", actual);
		if (interpreter->Invoke() == kTfLiteOk) {
			TfLiteTensor* output = interpreter->output(0);
			if (quant_type == QuantType::INT_8) {
				max_score_idx = GetMaxScoreIdx(
					tflite::GetTensorData<int8_t>(output), 
					num_classes, 
					output->params.scale, 
					output->params.zero_point
				);
			} else {
				max_score_idx = GetMaxScoreIdx(
					tflite::GetTensorData<float>(output), 
					num_classes
				);
			}
			ESP_LOGI(TAG_MAIN, "Predicted: %s\n", classes[max_score_idx]);

			// todo: better logging, see: https://github.com/espressif/tflite-micro-esp-examples/blob/master/examples/person_detection/main/main_functions.cc
			vTaskGetRunTimeStats(runtime_stats);
			// ESP_LOGI(TAG_PERF, "%s", runtime_stats);
		}
	} else {
		ESP_LOGE(TAG_MAIN, "File image capture failed\n");
	}

	vTaskDelay(5);
}
