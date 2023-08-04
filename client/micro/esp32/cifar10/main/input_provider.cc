#include "tensorflow/lite/micro/micro_utils.h"
#include "esp_log.h"

#include <cstdio>
#include <cstdint>
#include <cstring>

#include "input_provider.h"
#include "model_info.h"

namespace {
	const char* TAG_INPUT = "input";
}


esp_err_t InputProvider::GetFileInput(float* input_data, char* label) {
    FILE* images = io_manager.GetFile("/spiffs/images.bin", "r");
    if (images == NULL) {
		return ESP_ERR_NOT_FOUND;
	}

    fseek(images, image_file_pos, SEEK_SET);
    size_t bytes_read = fread(image_data, sizeof(unsigned char), image_data_len, images);
    if (bytes_read < image_data_len || feof(images)) {
        ESP_LOGD(TAG_INPUT, "EOF reached, rewinding...");
        image_file_pos = 0;
        return ESP_ERR_INVALID_SIZE;
    }
    for (int i = 0; i < num_inputs; i++) {
        input_data[i] = (float) image_data[i + 1] / 255.;
    }
    snprintf(label, 20, classes[image_data[0]]);
    image_file_pos += image_data_len;

    fclose(images);
    return ESP_OK;
}

esp_err_t InputProvider::GetFileInput(int8_t* input_data, char* label, float scale, int zero_point) {
    float float_inputs[num_inputs] = {};
    esp_err_t stat = GetFileInput(float_inputs, label);
	if (stat == ESP_OK) {
		tflite::Quantize<int8_t>(float_inputs, input_data, num_inputs, scale, zero_point);
	}
	return stat;
}

esp_err_t InputProvider::GetSerialInput(float* input_data, unsigned char* serial_input) {
    for (int i = 0; i < num_inputs; i++) {
        input_data[i] = (float) serial_input[i] / 255.;
    }
    return ESP_OK;
}

esp_err_t InputProvider::GetSerialInput(int8_t* input_data, unsigned char* serial_input, float scale, int zero_point) {
    float float_inputs[num_inputs] = {};
    esp_err_t stat = GetSerialInput(float_inputs, serial_input);
    if (stat == ESP_OK) {
        tflite::Quantize<int8_t>(float_inputs, input_data, num_inputs, scale, zero_point);
    }
    return stat;
}

InputProvider::InputProvider(IOManager& io_manager_)
    : image_file_pos(0)
    , num_file_samples(0)
    , image_data{}
    , io_manager(io_manager_)
{}
