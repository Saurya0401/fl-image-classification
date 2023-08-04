#ifndef DRY_BEAN_ESP32_WIFI_MANAGER_H_
#define DRY_BEAN_ESP32_WIFI_MANAGER_H_

extern bool WIFI_CONNECTED;

void InitializeWiFi();
void WiFiWaitConnected();
void WiFiStop();

#endif
