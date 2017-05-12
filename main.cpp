#include <iostream>

#include "neuralnetwork.hpp"

int main()
{
    Eigen::VectorXf inputs(2);
    inputs[0] = 1;
    inputs[1] = 0;
    std::vector<unsigned> t = {2, 2, 1};
    Eigen::MatrixXf w1(3, 2);
    Eigen::MatrixXf w2(3, 1);
    w1.col(0) << 20, 20, -30;
    w1.col(1) << -20, -20, 10;
    w2 << 20, 20, -10;
    NeuralNetwork nn(t);
    nn.setWeights(0, w1);
    nn.setWeights(1, w2);
    nn.feedforward(inputs);
    nn.displayLayers();
    return 0;
}
