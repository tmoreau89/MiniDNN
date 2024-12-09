#include "MiniDNN.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <Eigen/Dense>
#include "Output/KLDivergenceLoss.h"
#include "Output/RegressionMSE.h"

using namespace MiniDNN;

// Function to load a CSV into an Eigen matrix
void load_csv(const std::string& filename, Eigen::MatrixXd& matrix)
{
    std::ifstream file(filename);
    std::string line;
    std::vector<std::vector<double>> data;

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> row;

        while (std::getline(ss, value, ','))
        {
            row.push_back(std::stod(value));
        }

        data.push_back(row);
    }

    // Convert vector to Eigen matrix
    matrix = Eigen::MatrixXd(data.size(), data[0].size());
    for (size_t i = 0; i < data.size(); ++i)
    {
        for (size_t j = 0; j < data[i].size(); ++j)
        {
            matrix(i, j) = data[i][j];
        }
    }
}

int main()
{
    // Load training data
    Eigen::MatrixXd X_train, Y_train;
    load_csv("iris_features_train.csv", X_train);
    load_csv("iris_soft_targets_train.csv", Y_train);

    // Tranpose and print dimensions
    X_train.transposeInPlace();
    Y_train.transposeInPlace();
    std::cout << "X_train dimensions: " << X_train.rows() << " x " << X_train.cols() << std::endl;
    std::cout << "Y_train dimensions: " << Y_train.rows() << " x " << Y_train.cols() << std::endl;

    // Construct a network object
    Network net;

    // Define the network layers
    std::vector<int> hidden_nodes = {64, 32}; // Example hidden layer sizes
    int input_size = X_train.rows();
    for (int nodes : hidden_nodes)
    {
        if (nodes > 0)
        {
            net.add_layer(new FullyConnected<ReLU>(input_size, nodes));
            input_size = nodes;
        }
    }
    net.add_layer(new FullyConnected<Softmax>(input_size, Y_train.rows()));

    // Set the output layer
    net.set_output(new KLDivergenceLoss());

    // Create optimizer object
    Adam opt;
    opt.m_lrate = 0.01;

    // Set a callback to monitor training progress
    VerboseCallback callback;
    net.set_callback(callback);

    // Initialize parameters with N(0, 0.01^2) using random seed 123
    net.init(0, 0.01, 123);

    // Train the network
    int BATCH_SIZE = 16;
    int NUM_EPOCHS = 128;
    net.fit(opt, X_train, Y_train, BATCH_SIZE, NUM_EPOCHS, 123);
    std::cout << "Training complete!" << std::endl;

    // Load testing data, and transpose
    Eigen::MatrixXd X_test, Y_test;
    load_csv("iris_features_test.csv", X_test);
    load_csv("iris_soft_targets_test.csv", Y_test);
    X_test.transposeInPlace();
    Y_test.transposeInPlace();

    // Predict on the test set
    Eigen::MatrixXd predictions = net.predict(X_test);
    if (predictions.rows() != Y_test.rows() || predictions.cols() != Y_test.cols())
    {
        std::cerr << "Dimension mismatch between predictions and Y_test!" << std::endl;
        std::abort();
    }

    // Calculate accuracy or other metrics
    double total_loss = 0.0;
    int num_samples = X_test.cols(); // Number of samples is now the number of columns

    for (int i = 0; i < num_samples; ++i) // Iterate over each observation
    {
        for (int j = 0; j < Y_test.rows(); ++j) // Iterate over each class
        {
            double prediction_val = predictions(j, i) + 1e-8; // Avoid zero
            double y_test_val = Y_test(j, i) + 1e-8; // Avoid zero

            if (prediction_val <= 0 || y_test_val <= 0)
            {
                std::cerr << "Invalid test values: "
                          << "predictions(" << j << ", " << i << ") = " << predictions(j, i)
                          << ", Y_test(" << j << ", " << i << ") = " << Y_test(j, i) << std::endl;
                std::abort();
            }

            total_loss += y_test_val * (std::log(y_test_val) - std::log(prediction_val));
        }
    }

    total_loss /= num_samples;
    std::cout << "Test KL Divergence Loss: " << total_loss << std::endl;

    return 0;
}

