#pragma once

#include <vector>

class Perceptron
{
    public: 

                            Perceptron(int numberOfInputs, float learningRate);
        bool                predict(std::vector<int> inputs);
        void                train(std::vector<int> trainingInputs, int label);
        std::vector<float>  getWeights();

    private:

        float               _learningRate;
        std::vector<float>  _weights;

};