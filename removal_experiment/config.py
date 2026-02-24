"""
Configuration Constants for Removal Experiment
===============================================
Training parameters are calibrated around a 3-layer, 12-filter model
to ensure fair comparison across different configurations.
"""

# =============================================================================
# TRAINING PARAMETERS (calibrated for ~15-20 min on M1 MacBook Pro)
# =============================================================================

TRAINING_SAMPLES = 5000       # Number of training images to use
TEST_SAMPLES = 1000           # Number of test images for evaluation
EPOCHS = 5                    # Training epochs
BATCH_SIZE = 100              # Mini-batch size for progress reporting
LEARNING_RATE = 0.01          # SGD learning rate
LR_DECAY = 0.95               # Learning rate decay per epoch

# =============================================================================
# NETWORK ARCHITECTURE DEFAULTS
# =============================================================================

DEFAULT_LAYERS = 3            # Default number of conv layers
DEFAULT_FILTERS = 12          # Default filters per layer (uniform)
DEFAULT_N_PASS = None         # None means "pass all" (no sparsification)

FILTER_SIZE = 3               # Conv filter dimensions (3x3)
POOL_SIZE = 2                 # Max pooling window size
DENSE_HIDDEN_SIZE = 128       # Hidden layer size before output
NUM_CLASSES = 10              # MNIST has 10 digit classes

# =============================================================================
# PATHS
# =============================================================================

MNIST_DATA_DIR = "mnist/mnist"
EXPERIMENT_BASE_DIR = "removal_experiment"

# =============================================================================
# VISUALIZATION
# =============================================================================

SAVE_ACTIVATION_EXAMPLES = True   # Save activation maps for debugging
EXAMPLES_PER_DIGIT = 1            # How many examples of each digit to save
