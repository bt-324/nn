#include <iostream>
#include <vector>
#include "../include/nn.hpp"

int main() {
	
	// Zmienne "niutkowane" lubią być przechowywane
	// w std::auto_ptr
	// https://devcode.pl/cpp11-unique-ptr/
	
	std::cout << "\t\tNeural Network\n";

	uint8_t layerCount = 3;
	uint8_t hiddenLayerSize = 3;
	std::vector <double> input { 2.0, -1.0, 5.0 };
	std::vector <double> output { 1.0, 0.0, 1.0 };

	NeuralNetwork *nn = new NeuralNetwork(layerCount, input, hiddenLayerSize, output);
	nn->init();
	nn->print();

	return 0;
}
