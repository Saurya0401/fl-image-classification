#include "tensorflow/lite/micro/micro_utils.h"
#include "output_handler.h"

int GetMaxScoreIdx(float* scores, int num_scores) {
	int max_score_idx = 0;
	float max_score = scores[max_score_idx];
	for (int i = 1; i < num_scores; i++) {
		if (scores[i] > max_score) {
			max_score_idx = i;
			max_score = scores[i];
		}
	}
	return max_score_idx;
}

int GetMaxScoreIdx(int8_t* scores, int num_scores, float scale, int zero_point) {
	float f_scores[num_scores] = {};
	tflite::Dequantize<int8_t>(scores, num_scores, scale, zero_point, f_scores);
	return GetMaxScoreIdx(f_scores, num_scores);
}