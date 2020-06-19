#include "../include/nn.hpp"

////////// ACTIVATION FUNCTIONS ////////////////

double sigmoid(double val) {

	return ( 1 / (1 + exp(-val)) );
}

double sigmoidPrime(double val) {

    return ( sigmoid(val) * (1 - sigmoid(val)) );

}
///////////////////////////////////////////////
///////////////// NEURON ///////////////////////

Neuron::Neuron(double val) : m_weight(1.0), m_bias(0.0) {

	m_val = val; 

}

void Neuron::input(std::vector<Neuron *> l) {

	double tempSum = 0.0;

	for(int i = 0; i < l.size(); i++)
	{
		tempSum += l.at(i)->output();
	}

	m_z = (tempSum * m_weight + m_bias);
	m_val = (sigmoid(tempSum * m_weight + m_bias));

}


///////////////////////////////////////////////
///////////////// LAYER ///////////////////////

Layer::Layer(uint8_t size) {

	m_size = size;

	for(int i = 0; i < m_size; i++)
	{
		Neuron *n = new Neuron(0.0);
		m_neurons.push_back(n);
	}	

}

Layer::Layer(std::vector<Neuron *> l) {

	for(int i = 0; i < l.size(); i++)
	{
		m_neurons.push_back(new Neuron(l.at(i)->output()));
	}	

}

void Layer::input(std::vector <Neuron *> l) {

	for(int i = 0; i < m_neurons.size(); i++)
	{
		m_neurons.at(i)->input(l);
	}

}

void Layer::print() const {

	for(int i = 0; i < m_neurons.size(); i++)
	{
	  std::cout << "Neuron: " << i << "\t";
	  std::cout << "Val: " << m_neurons.at(i)->getVal() << "\t";
	  std::cout << "Weight: " << m_neurons.at(i)->getWeight() << "\t";
	  std::cout << "Bias: " << m_neurons.at(i)->getBias() << "\n";
	}	

}
///////////////////////////////////////////////
////////////// NEURAL NETWORK ////////////////

NeuralNetwork::NeuralNetwork(uint8_t layerCount, std::vector <double> input , uint8_t hiddenLayerSize, std::vector <double> targetOutput) {

	Layer *inp = new Layer(input.size());

	for(int i = 0; i < inp->getLayer().size(); i++)
	{
		inp->getLayer().at(i)->setVal(input.at(i));	
	}	

	m_layers.push_back(inp);
	
	for(int i = 1; i < layerCount; i++)
	{
		Layer *l = new Layer(hiddenLayerSize);
		m_layers.push_back(l);	
	}	

	setTargetOutput(targetOutput);
}

void NeuralNetwork::print() const {

	for(int i = 0; i < m_layers.size(); i++)
	{
	  std::cout << "\n\t\tLayer: " << i << "\t\n\n";
	  m_layers.at(i)->print();

	}	

	for(int i = 0; i < m_output.size(); i++)
	{
		std::cout << "Error of output neuron: " << i << "\t\n";
		std::cout << m_targetOutput.at(i) << " - " <<  m_output.at(i) <<  " = "; 
		std::cout << m_errors.at(i) << "\t\n";
	}
		std::cout << "Total error = " << m_totalError << "\t\n";
}

void NeuralNetwork::init() {

	// Forward propagation

	for(int i = 1; i < m_layers.size(); i++)
	{
	  uint8_t prevLayer = i - 1;
	   m_layers.at(i)->input(m_layers.at(prevLayer)->getLayer());
	}	

	// Error values calculation

	for(int i = 0; i < m_layers.back()->getLayer().size(); i++)
	{
		m_output.push_back(m_layers.back()->getLayer().at(i)->getVal());
		m_errors.push_back( (pow((m_targetOutput.at(i) - m_output.at(i)), 2.0)) / m_output.size() );
		m_totalError += m_errors.at(i);
	}

	// Backpropagation
	
	double dEdw = 0.0;
	
	for(int i = m_layers.size(); i > 0 ; i--)
	{
		for(int j = 0; j < m_layers.at(i)->getLayer().size(); j++)
		{
		
			int prevLayer = i--;

			dEdw = m_layers.at(prevLayer)->getLayer().at(j)->getVal() 

				* sigmoidPrime(m_layers.at(i)->getLayer().at(j)->getZ()) 
				
				* 2 * (m_layers.at(i)->getLayer().at(j)->getVal() - m_targetOutput.at(j)); 
 

		}

	}

}
