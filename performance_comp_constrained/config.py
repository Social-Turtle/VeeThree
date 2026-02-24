"""
Configuration for Constrained Dense Layer Experiments
====================================================
"""

# Training parameters
TRAINING_SAMPLES = 10000
TEST_SAMPLES = 1000
EPOCHS = 10
LEARNING_RATE = 0.01
LR_DECAY = 0.95
BATCH_SIZE = 32

# Model architecture
NUM_FILTERS_CONV1 = 4        # Fixed 4 edge filters for layer 1
NUM_FILTERS_CONV2 = 4        # 4 learned filters for layer 2
CONV2_KERNEL_SIZE = 3        # Fixed 3x3 for conv2
POOL_SIZE = 2
DENSE_LAYER_SIZES = [400, 100, 50, 10]  # Dense layer sizes to test
DENSE_HIDDEN_SIZE = 128  # Default, overridden in sweep
NUM_CLASSES = 10

# Experiment variations
KERNEL_SIZES = [3, 4, 5]     # Conv1 kernel sizes to test
N_PASS_VALUES = [1, 2, 3, 4] # For encoded models
ENCODING_TYPES = ['fixed_rank', 'fixed_vector', 'learned_rank']

# Data
MNIST_DATA_DIR = "mnist/mnist"
