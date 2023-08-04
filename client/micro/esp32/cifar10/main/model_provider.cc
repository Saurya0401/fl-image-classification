#include <cstdio>
#include <cstring>
#include "esp_log.h"
#include "model_info.h"
#include "model_provider.h"

#define DEFAULT_MODEL_FILEPATH (CONFIG_MODEL_FILEPATH)
#define DEFAULT_SERVER_ADDRESS (CONFIG_SERVER_ADDRESS)

namespace {
    const char* TAG_HTTP = "http";
    const char* TAG_MODEL = "model";
    const int MODEL_ALIGNMENT = 8;
}

esp_err_t _http_event_handler(esp_http_client_event_t *event) {
    switch(event->event_id) {
        case HTTP_EVENT_ERROR:
            ESP_LOGE(TAG_HTTP, "HTTP error");
            break;
        case HTTP_EVENT_ON_CONNECTED:
            ESP_LOGD(TAG_HTTP, "Connected to server");
            break;
        case HTTP_EVENT_HEADER_SENT:
            ESP_LOGD(TAG_HTTP, "Header sent to server");
            break;
        case HTTP_EVENT_ON_HEADER:
            ESP_LOGD(TAG_HTTP, "Header received from server, len = %d", event->data_len);
            break;
        case HTTP_EVENT_ON_DATA:
            ESP_LOGD(TAG_HTTP, "Data received from server, len = %d", event->data_len);
            break;
        case HTTP_EVENT_ON_FINISH:
            ESP_LOGD(TAG_HTTP, "HTTP session finished");
            break;
        case HTTP_EVENT_DISCONNECTED:
            ESP_LOGD(TAG_HTTP, "Disconnected from server");
            break;
        default:
            ESP_LOGW(TAG_HTTP, "Unknown HTTP event");
    }
    return ESP_OK;
}


esp_err_t ModelProvider::LargeModelError(size_t model_size, const char* tag) {
    size_t free_memory = heap_caps_get_largest_free_block(MALLOC_CAP_8BIT);
    ESP_LOGE(tag, "Model too large (size: %u bytes, free: %u bytes)", model_size, free_memory);
    return ESP_ERR_NO_MEM;
}

esp_err_t ModelProvider::GetModelFromServer(const char* server_url) {
    esp_http_client_config_t config = {
        .url = server_url,
        .event_handler = _http_event_handler
    };
    esp_http_client_handle_t client = esp_http_client_init(&config);
    esp_err_t err;
    if ((err = esp_http_client_open(client, 0)) != ESP_OK) {
        ESP_LOGE(TAG_HTTP, "Failed to open HTTP connection: %s", esp_err_to_name(err));
        return err;
    }
    heap_caps_free(http_model_buffer);
    http_model_buffer = nullptr;
    http_model_size = esp_http_client_fetch_headers(client);
    http_model_buffer = (char*) heap_caps_malloc(http_model_size, MALLOC_CAP_8BIT);
    if (http_model_buffer == NULL) {
        http_model_buffer = nullptr;
        return LargeModelError(http_model_size, TAG_HTTP);
    }
    int read_len = esp_http_client_read(client, http_model_buffer, http_model_size);
    if (read_len <= 0) {
        ESP_LOGE(TAG_HTTP, "Couldn't read data");
        return ESP_FAIL;
    }
    ESP_LOGD(TAG_HTTP, "read_len = %d", read_len);
    ESP_LOGI(TAG_HTTP, "HTTP Stream reader Status = %d, content_length = %lld",
                    esp_http_client_get_status_code(client),
                    esp_http_client_get_content_length(client));
    esp_http_client_close(client);
    esp_http_client_cleanup(client);
    return ESP_OK;
}

esp_err_t ModelProvider::WriteModelToSPIFFS(const char* model_filepath) {
    FILE* model_file = io_manager.GetFile(model_filepath, "wb");
    if (model_file == NULL) {
        ESP_LOGE(TAG_MODEL, "Failed to open model file");
        return ESP_ERR_NOT_FOUND;
    }
    int bytes_written = fwrite(http_model_buffer, sizeof(unsigned char), http_model_size, model_file);
    if (bytes_written != http_model_size) {
        ESP_LOGE(TAG_MODEL, "Error while writing model");
        fclose(model_file);
        return ESP_FAIL;
    }
    fclose(model_file);
    return ESP_OK;
}

esp_err_t ModelProvider::UpdateModel(const char* server_url, const char* model_filepath) {
    esp_err_t update_stat;
    update_stat = GetModelFromServer(server_url);
    if (update_stat != ESP_OK || http_model_buffer == nullptr) {
        ESP_LOGE(TAG_MODEL, "Failed to get model from server");
        return update_stat;
    }
    update_stat = WriteModelToSPIFFS(model_filepath);
    if (update_stat != ESP_OK) {
        ESP_LOGE(TAG_MODEL, "Failed to write model to storage");
        return update_stat;
    }
    ESP_LOGI(TAG_MODEL, "Successfully updated model from server");
    heap_caps_free(http_model_buffer);
    http_model_buffer = nullptr;
    return update_stat;
}

esp_err_t ModelProvider::UpdateModel() {
    return UpdateModel(DEFAULT_SERVER_ADDRESS, DEFAULT_MODEL_FILEPATH);
}

esp_err_t ModelProvider::LoadModel(const char* model_path) {
    int model_size = io_manager.GetFileSize(model_path, "rb");
    FILE* model_file = io_manager.GetFile(model_path, "rb");
    if (model_size == -1 || model_file == NULL) {
        ESP_LOGE(TAG_MODEL, "Failed to locate model file");
        return ESP_ERR_NOT_FOUND;
    }
    heap_caps_free(model_data);
    model_data = nullptr;
    model_data = (unsigned char*) heap_caps_aligned_alloc(MODEL_ALIGNMENT, model_size, MALLOC_CAP_8BIT);
    if (model_data == NULL) {
        model_data = nullptr;
        return LargeModelError(model_size, TAG_MODEL);
    }
    int bytes_read = fread(model_data, sizeof(unsigned char), model_size, model_file);
    if (bytes_read != model_size) {
        ESP_LOGE(TAG_MODEL, "Mismatch between model filesize and bytes");
        fclose(model_file);
        return ESP_FAIL;
    }
    int quant_type_id = model_data[bytes_read - 1];
    quant_type = ParseQuantTypeFromInt(quant_type_id);
    fclose(model_file);
    ESP_LOGI(TAG_MODEL, "Model loaded (size: %d bytes, quant type: %x)", bytes_read - 1, quant_type_id);
    return ESP_OK;
}

esp_err_t ModelProvider::LoadModel() {
    return LoadModel(DEFAULT_MODEL_FILEPATH);
}

ModelProvider::ModelProvider(IOManager io_handler_)
    : io_manager(io_handler_)
    , quant_type(QuantType::UNKNOWN)
    , http_model_size(0)
    , http_model_buffer(nullptr)
    , model_data(nullptr) {}

ModelProvider::~ModelProvider() {
    free(model_data);
    free(http_model_buffer);
}
