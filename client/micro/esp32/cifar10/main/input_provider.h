#ifndef CIFAR10_ESP32_INPUT_PROVIDER_H_
#define CIFAR10_ESP32_INPUT_PROVIDER_H_

#include <cstdint>
#include <cstdio>
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "esp_err.h"
#include "model_info.h"
#include "IO_manager.h"

class InputProvider {
	private:
		static const int image_data_len = num_inputs + 1;
		long image_file_pos;
		int num_file_samples;
		unsigned char image_data[image_data_len];
		IOManager& io_manager;
	public:
		esp_err_t GetFileInput(float* input_data, char* label);
		esp_err_t GetFileInput(int8_t* input_data, char* label, float scale, int zero_point);
		esp_err_t GetSerialInput(float* input_data, unsigned char* serial_input);
		esp_err_t GetSerialInput(int8_t* input_data, unsigned char* serial_input, float scale, int zero_point);
		InputProvider(IOManager& io_manager);
};

#endif