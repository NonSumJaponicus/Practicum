A multilayered network for use with Fashion MNIST database.

Structure:
- Sequential model
- Dense layer with 128 neurons and ReLU activation
- Dense layer with 64 neurons and ReLU activation
- Dense layer with 64 neurons and ReLU activation

Compilation:
- Optimizer: SGD
- Loss: sparse categorical cross-entropy
- Metric: accuracy

Training and validation:
- Batch size: 300
- Epochs: 100

Final validation accuracy: 0.8767