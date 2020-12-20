#include "../include/Perceptron.h"
#include <stdexcept>

using namespace std;

// Constructor
Perceptron::Perceptron(int numberOfInputs, float learningRate)
{
    _learningRate = learningRate;
    _weights.assign(numberOfInputs + 1, 0);
}

// Prediction function based on current weights
bool Perceptron::predict(vector<int> inputs)
{
    // Check if input size if right
    if (inputs.size() != _weights.size() - 1)
        throw invalid_argument("Wrong number of inputs for perceptron");

    // Sum is initially the bias
    float sum = _weights[0];
    for (int i = 0; i != inputs.size(); i++)
        sum += inputs[i] * _weights[i + 1]; // Summation of inputs * weights

    // Heaviside function
    return sum >= 0;
}

// Supervised learning that modifies weights
void Perceptron::train(vector<int> trainingInputs, int label)
{
    // Use the perceptron to predict an answer
    int prediction = this->predict(trainingInputs);

    // Mutate
    _weights[0] += _learningRate * (label - prediction); // Bias
    for (int i = 0; i < _weights.size() - 1; i++)        // Input weights
        _weights[i + 1] += _learningRate * (label - prediction) * trainingInputs[i];
}

// Returns current weights
vector<float> Perceptron::getWeights()
{
    return _weights;
}