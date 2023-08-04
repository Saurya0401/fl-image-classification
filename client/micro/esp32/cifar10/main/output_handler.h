#ifndef CIFAR10_ESP32_OUTPUT_HANDLER_H_
#define CIFAR10_ESP32_OUTPUT_HANDLER_H_

#include <cstdint>

int GetMaxScoreIdx(float* scores, int num_scores);
int GetMaxScoreIdx(int8_t* scores, int num_scores, float scale, int zero_point);

#endif