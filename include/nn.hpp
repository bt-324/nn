#ifndef __NN_HPP
#define __NN_HPP

#include <iostream>
#include <vector>
#include <math.h>
#include <iomanip>

double sigmoid(double val); 

///////////////// NEURON ///////////////////////

class Neuron {

	public:

		Neuron(double val);

		void setVal(double val) { m_val = val; }
		void setWeight(double weight) { m_weight = weight; }
		void setBias(double bias) { m_bias = bias; } 

		double getVal() const { return m_val; }
		double getWeight() const { return m_weight; }
		double getBias() const { return m_bias; }
		double getZ() const { return m_z; }

		void input(std::vector<Neuron *> l);
		double output() const { return m_val; }

	private:

		double m_val;
		double m_weight;
		double m_bias;
		double m_z;
};

///////////////////////////////////////////////
///////////////// LAYER ///////////////////////

class Layer {

	public:

		// Construct a layer of given size with nodes of value 0.0
		Layer(uint8_t size);
		// Construct a layer of nodes with given values from vector of nodes
		Layer(std::vector<Neuron *> l);

		void input(std::vector <Neuron *> layer);
		std::vector <Neuron *> output() const { return m_neurons; }
		std::vector <Neuron *> getLayer() const { return m_neurons; }
		void print() const;

	private:

		uint8_t m_size;
		std::vector<Neuron *> m_neurons;
};

///////////////////////////////////////////////
////////////// NEURAL NETWORK ////////////////

class NeuralNetwork {

	public:

		NeuralNetwork(uint8_t layerCount, std::vector <double> input, uint8_t hiddenLayerSize, std::vector <double> targetOutput); 

		void setTargetOutput(std::vector <double> targetOutput) { m_targetOutput = targetOutput; }
		void setLearningRate(double lr) { m_learningRate = lr; } 

		double getLearningRate() const { return m_learningRate; }

		void output() const;
		void print() const;
		void feedForward();
		void errorCalc();
		void backprop();
		void init(uint64_t epochs);

		std::vector <Layer *> getLayers() const { return m_layers; }

	private:

		std::vector <Layer *> m_layers;
		std::vector <double> m_targetOutput;
		std::vector <double> m_output;
		std::vector <double> m_errors;
		double m_totalError;
		double m_learningRate;
};

#endif
