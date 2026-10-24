
#include <torch/torch.h>
#include <iostream>

struct Net : torch::nn::Module {  // Inherit from torch::nn::Module to define a custom neural network.

    Net() {
        // Fully connected layer: 784 input features -> 128 neurons.
        // 784 corresponds to a flattened 28x28 image.
        fc1 = register_module(
            "fc1",
            torch::nn::Linear(784, 128)
        );

        // Fully connected output layer: 128 neurons -> 10 classes.
        // The 10 outputs can represent digits from 0 to 9.
        fc2 = register_module(
            "fc2",
            torch::nn::Linear(128, 10)
        );
    }

    torch::Tensor forward(torch::Tensor x) {
        // -------------------------------------------------
        // Reshape each image into a vector of 784 features
        // Input: [batch_size, 1, 28, 28]
        // Output: [batch_size, 784]
        // -------------------------------------------------
        x = x.reshape({x.size(0), 784});

        // First fully connected layer followed by ReLU activation.
        // ReLU introduces non-linearity: f(x) = max(0, x).
        // Output shape: [batch_size, 128]
        x = torch::relu(fc1->forward(x));

        // Output layer produces 10 scores, one per class.
        // Output shape: [batch_size, 10]
        x = fc2->forward(x);

        // Convert scores into log-probabilities along dimension 1.
        // Commonly used with torch::nn::NLLLoss during training.
        return torch::log_softmax(x, /*dim=*/1);
    }

    // Two fully connected (Linear) layers.
    // nullptr indicates that the module holders start uninitialized.
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

int main() {

    auto net = std::make_shared<Net>();

    auto input = torch::randn({64, 1, 28, 28}); // Tensor dimensions: [64, 1, 28, 28]

    auto output = net->forward(input);
    
    std::cout << "Output size: "
              << output.sizes()
              << std::endl;

    return 0;
}
