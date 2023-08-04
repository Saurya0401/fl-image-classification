#ifndef CIFAR10_ESP32_MODEL_PROVIDER_H_
#define CIFAR10_ESP32_MODEL_PROVIDER_H_

#include "esp_http_client.h"
#include "esp_err.h"
#include "IO_manager.h"
#include "model_info.h"

class ModelProvider {
    public:
        inline unsigned char* GetModelData() const { return model_data; }
        inline QuantType GetQuantType() const { return quant_type; }
        esp_err_t UpdateModel(const char* server_url, const char* model_filepath);
        esp_err_t UpdateModel();
        esp_err_t LoadModel(const char* model_file);
        esp_err_t LoadModel();
        ModelProvider(IOManager io_handler_);
        ~ModelProvider();
    private: 
        IOManager& io_manager;
        QuantType quant_type;
        int http_model_size;
        char* http_model_buffer;
        alignas(8) unsigned char* model_data;
    private:
        esp_err_t LargeModelError(size_t model_size, const char* tag);
        esp_err_t GetModelFromServer(const char* server_url);
        esp_err_t WriteModelToSPIFFS(const char* model_filepath);
};

#endif
