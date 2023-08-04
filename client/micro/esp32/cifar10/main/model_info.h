#ifndef CIFAR10_ESP32_MODEL_INFO_H_
#define CIFAR10_ESP32_MODEL_INFO_H_

enum QuantType {NO_QUANT = 0xD0, INT_8 = 0xD1, DYNAMIC = 0xD2, UNKNOWN = 0xDD};

constexpr QuantType ParseQuantTypeFromInt(const int& val) {
	switch (val) {
		case static_cast<int>(QuantType::NO_QUANT):
		case static_cast<int>(QuantType::DYNAMIC):
		case static_cast<int>(QuantType::INT_8):
			return static_cast<QuantType>(val);
		default:
			return QuantType::UNKNOWN;
	}
}

constexpr int img_dim_rows = 32;
constexpr int img_dim_cols = 32;
constexpr int img_channels = 3;
constexpr int num_inputs = img_dim_cols * img_dim_cols * img_channels;
constexpr int num_classes = 10;
constexpr char* classes[num_classes] = {
	(char*) "AIRPLANE",
	(char*) "AUTOMOBILE",
	(char*) "BIRD",
	(char*) "CAT",
	(char*) "DEER",
	(char*) "DOG",
	(char*) "FROG",
	(char*) "HORSE",
	(char*) "SHIP",
	(char*) "TRUCK"
};
constexpr int airplane_idx = 0;
constexpr int automobile_idx = 1;
constexpr int bird_idx = 2;
constexpr int cat_idx = 3;
constexpr int deer_idx = 4;
constexpr int dog_idx = 5;
constexpr int frog_idx = 6;
constexpr int horse_idx = 7;
constexpr int ship_idx = 8;
constexpr int truck_idx = 9;


#endif
