"""
Configuration for Sequential Removal Experiments
=================================================
"""

# Training parameters (2x the original)
TRAINING_SAMPLES = 10000
TEST_SAMPLES = 1000
EPOCHS = 10
LEARNING_RATE = 0.01
LR_DECAY = 0.95
BATCH_SIZE = 32

# Model architecture
NUM_LAYERS = 2
NUM_FILTERS = 8
FILTER_SIZE = 3
POOL_SIZE = 2
DENSE_HIDDEN_SIZE = 128
NUM_CLASSES = 10

# Data
MNIST_DATA_DIR = "mnist/mnist"
