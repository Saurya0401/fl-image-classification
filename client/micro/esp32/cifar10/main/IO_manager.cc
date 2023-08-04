#include "driver/uart.h"
#include "esp_spiffs.h"
#include "esp_log.h"
#include "esp_err.h"
#include "IO_manager.h"

#define UART_PORT   (CONFIG_UART_PORT)
#define TXD_PIN     (CONFIG_TXD_PIN)
#define RXD_PIN     (CONFIG_RXD_PIN)

namespace {
    const int RX_BUF_SIZE = 1024;
    const int BAUD_RATE = 115200;
    const char* TAG_IO = "io";
}

void IOManager::InitUART() {
    const uart_config_t uart_config = {
        .baud_rate = BAUD_RATE,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_APB,
    };
    ESP_ERROR_CHECK(uart_driver_install(UART_PORT, RX_BUF_SIZE * 2, 0, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_param_config(UART_PORT, &uart_config));
    ESP_ERROR_CHECK(uart_set_pin(UART_PORT, TXD_PIN, RXD_PIN, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
}

void IOManager::InitSPIFFS() {
    esp_vfs_spiffs_conf_t config = {
        .base_path = "/spiffs",
        .partition_label = NULL,
        .max_files = 5,
        .format_if_mount_failed = true,
    };
    ESP_ERROR_CHECK(esp_vfs_spiffs_register(&config));
}

void IOManager::InitializeIO() {
    InitUART();
    ESP_LOGI(TAG_IO, "UART initialsied");
    InitSPIFFS();
    ESP_LOGI(TAG_IO, "SPIFFS initialsied");
}

void IOManager::TerminateIO() {
    ESP_ERROR_CHECK(uart_driver_delete(UART_PORT));
    ESP_LOGI(TAG_IO, "UART terminated");
    ESP_ERROR_CHECK(esp_vfs_spiffs_unregister(NULL));
    ESP_LOGI(TAG_IO, "SPIFFS terminated");
}

unsigned char* IOManager::CaptureSerialInput(const int wait_ms) {
    uint8_t* data = (uint8_t*) malloc(RX_BUF_SIZE + 1);
    while (true) {
        const int rxBytes = uart_read_bytes(UART_PORT, data, RX_BUF_SIZE, wait_ms / portTICK_PERIOD_MS);
        if (rxBytes > 0) {
            data[rxBytes] = '\0';
            break;
        }
    }
    return (unsigned char*) data;
}

FILE* IOManager::GetFile(const char* filepath, const char* mode) {
    FILE* file = fopen(filepath, mode);
	if (file == NULL) {
		ESP_LOGE(TAG_IO, "File '%s' not found", filepath);
	}
    return file;
}


int IOManager::GetFileSize(const char* filepath, const char* mode) {
    FILE* file = GetFile(filepath, mode);
    if (file == NULL) {
        ESP_LOGE(TAG_IO, "Failed to calculate filesize");
        return -1;
    }
    fseek(file, 0, SEEK_END);
    int size = ftell(file);
    fclose(file);
    return size;
}