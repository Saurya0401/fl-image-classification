idf_component_register(
    SRCS
        "main.cc"
        "main_functions.cc"
        "IO_manager.cc"
        "model_provider.cc"
        "input_provider.cc"
        "output_handler.cc"
        "wifi_manager.cc"
        "model_info.h"
    
    PRIV_REQUIRES tflite-lib driver spiffs esp_http_client esp_wifi nvs_flash
    INCLUDE_DIRS ".")

spiffs_create_partition_image(storage ../data FLASH_IN_PROJECT)
