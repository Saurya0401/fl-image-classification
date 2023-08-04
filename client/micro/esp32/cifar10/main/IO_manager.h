#ifndef CIFAR10_ESP32_IO_MANAGER_H_
#define CIFAR10_ESP32_IO_MANAGER_H_

#include <cstdio>

class IOManager {
    private:
        static void InitUART();
        static void InitSPIFFS();
    public:
        static void InitializeIO();
        static void TerminateIO();
        unsigned char* CaptureSerialInput(const int wait_ms);
        FILE* GetFile(const char* filepath, const char* mode);
        int GetFileSize(const char* filepath, const char* mode);
};

#endif