#include <iostream>
#include <vector>
#include "../include/nn.hpp"


/*
struct LayerConfig {
	uint8t m_layerCount
};

*/

/*
class App {
public:
	Init();
		Run();
		Done();
private:
	std::unique_ptr (...)
};
*/

int main() {

	uint8_t layerCount = 3;
	uint8_t hiddenLayerSize = 3;
	std::vector <double> input {2.0, -1.0, 5.0};
	std::vector <double> output {1.0, 0.0, 1.0};

    std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork(layerCount, input, hiddenLayerSize, output));
	/*
	LayerConfig config
		(...)
    std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork(&config));
	*/

	uint64_t epochs = 10;
	double lr = 0.5;

	nn->setLearningRate(lr);
	nn->init(50);
	nn->init(100);
	nn->init(500);

//	nn;
//	nn->Train(&schoolConfig)
//	std::unique_ptr<Trainer> nt(new NetworkTrainer(&schoolConfig));
// at this point network is somewhat smarter
//   nn->Aswer(&inputData)
}
