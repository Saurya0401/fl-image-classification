#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs_flash.h"
#include "esp_err.h"
#include "main_functions.h"
#include "wifi_manager.h"

void tf_main(void) {
	setup();
	while (true) {
		loop();
	}
}

extern "C" void app_main() {
	esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
      ESP_ERROR_CHECK(nvs_flash_erase());
      ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);
    InitializeWiFi();

	xTaskCreate((TaskFunction_t)& tf_main, "tf_main", 20 * 1024, NULL, 10, NULL);
	vTaskDelete(NULL);
}