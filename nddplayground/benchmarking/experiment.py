import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time

# Add paths to allow imports
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # VeeThree root
sys.path.append(base_dir) # For typical_cnn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # For nddplayground package

try:
    from typical_cnn import CNN, one_hot, sigmoid, sigmoid_derivative
    from nddplayground.data_loader import load_mnist_generator
    from nddplayground.l0_layer import L0FilterBank
    from nddplayground.pooling_layer import PoolingLayer
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# Modern activation functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def cross_entropy_loss(predicted, target):
    predicted = np.clip(predicted, 1e-10, 1 - 1e-10)
    return -np.sum(target * np.log(predicted))

def cross_entropy_derivative(predicted, target):
    return predicted - target

# Adam optimizer
class AdamOptimizer:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0
        
    def update(self, params, grads, param_id):
        if param_id not in self.m:
            self.m[param_id] = np.zeros_like(params)
            self.v[param_id] = np.zeros_like(params)
        
        self.t += 1
        self.m[param_id] = self.beta1 * self.m[param_id] + (1 - self.beta1) * grads
        self.v[param_id] = self.beta2 * self.v[param_id] + (1 - self.beta2) * (grads ** 2)
        
        m_hat = self.m[param_id] / (1 - self.beta1 ** self.t)
        v_hat = self.v[param_id] / (1 - self.beta2 ** self.t)
        
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

def count_parameters(cnn):
    count = 0
    for filters in cnn.conv_layers:
        for f in filters:
            count += f.size
    count += cnn.dense_hidden['weights'].size
    count += cnn.dense_hidden['biases'].size
    count += cnn.dense_output['weights'].size
    count += cnn.dense_output['biases'].size
    return count

def load_data(max_samples=2000):
    print(f"Loading {max_samples} samples...")
    mnist_dir = os.path.join(base_dir, "mnist/mnist")
    gen = load_mnist_generator(data_dir=mnist_dir, max_samples=max_samples)
    
    images = []
    labels = []
    if gen:
        for img, lbl in gen:
            images.append(img)
            labels.append(lbl)
    
    if not images:
        raise ValueError("No data loaded. Check paths.")

    split = int(0.8 * len(images))
    return (images[:split], labels[:split]), (images[split:], labels[split:])

class NDDHybridModel:
    """NDD-Hybrid using ReLU+Adam+Cross-Entropy (modern style for ablation)"""
    def __init__(self, num_classes=10, conv2_filters=4, dense_size=16, lr=0.001):
        self.l0 = L0FilterBank(length=4)
        self.channel_limit = 2
        self.pool1 = PoolingLayer(grid_size=2, top_e=1)
        self.cnn = CNN(
            input_shape=(14, 14),
            num_classes=num_classes,
            conv_filters=[conv2_filters],
            filter_size=4,
            pool_size=2,
            dense_size=dense_size
        )
        self.optimizer = AdamOptimizer(lr=lr)
        self.param_counter = 0

    def forward_modern(self, image):
        cache = {}
        l0_out = self.l0.process(image, channel_limit=self.channel_limit)
        channels = [l0_out['horizontal'], l0_out['vertical'], l0_out['diagonal1'], l0_out['diagonal2']]
        pooled_channels = [self.pool1.process_single(c) for c in channels]
        cnn_input = np.array(pooled_channels)
        from typical_cnn import convolve_image, pool_maps
        current = cnn_input
        for filters in self.cnn.conv_layers:
            conv_out = convolve_image(current, filters, self.cnn.stride, relu)
            pooled = pool_maps(conv_out, self.cnn.pool_method, self.cnn.pool_size)
            current = pooled
        cache['flattened'] = current.flatten()
        z_hidden = np.dot(cache['flattened'], self.cnn.dense_hidden['weights']) + self.cnn.dense_hidden['biases']
        cache['hidden'] = relu(z_hidden)
        cache['z_hidden'] = z_hidden
        z_output = np.dot(cache['hidden'], self.cnn.dense_output['weights']) + self.cnn.dense_output['biases']
        cache['output'] = softmax(z_output)
        return cache['output'], cache

    def train_step(self, image, label):
        output, cache = self.forward_modern(image)
        target = one_hot(label, 10)
        loss = cross_entropy_loss(output, target)
        d_output = cross_entropy_derivative(output, target)
        d_weights_out = np.outer(cache['hidden'], d_output)
        d_biases_out = d_output
        self.cnn.dense_output['weights'] = self.optimizer.update(
            self.cnn.dense_output['weights'], d_weights_out, f'dense_out_w_{self.param_counter}'
        )
        self.cnn.dense_output['biases'] = self.optimizer.update(
            self.cnn.dense_output['biases'], d_biases_out, f'dense_out_b_{self.param_counter}'
        )
        d_hidden = np.dot(d_output, self.cnn.dense_output['weights'].T)
        d_hidden *= relu_derivative(cache['z_hidden'])
        d_weights_hidden = np.outer(cache['flattened'], d_hidden)
        d_biases_hidden = d_hidden
        self.cnn.dense_hidden['weights'] = self.optimizer.update(
            self.cnn.dense_hidden['weights'], d_weights_hidden, f'dense_hid_w_{self.param_counter}'
        )
        self.cnn.dense_hidden['biases'] = self.optimizer.update(
            self.cnn.dense_hidden['biases'], d_biases_hidden, f'dense_hid_b_{self.param_counter}'
        )
        self.param_counter += 1
        return loss

    def predict(self, image):
        out, _ = self.forward_modern(image)
        return np.argmax(out)

# --- New Model 1: CNN with NDD Gradient Preprocessing (no inhibition) ---
class NDDGradientsOnlyModel:
    """CNN using NDDs for gradients (no inhibition/top-k), rest is conventional CNN."""
    def __init__(self, num_classes=10, conv2_filters=4, dense_size=16, lr=0.001):
        self.l0 = L0FilterBank(length=4)
        self.channel_limit = 2
        self.cnn = CNN(
            input_shape=(28, 28),
            num_classes=num_classes,
            conv_filters=[conv2_filters],
            filter_size=5,
            pool_size=2,
            dense_size=dense_size
        )
        self.optimizer = AdamOptimizer(lr=lr)
        self.param_counter = 0

    def forward_modern(self, image):
        cache = {}
        l0_out = self.l0.process(image, channel_limit=self.channel_limit)
        channels = [l0_out['horizontal'], l0_out['vertical'], l0_out['diagonal1'], l0_out['diagonal2']]
        cnn_input = np.array(channels)
        # Forward through CNN conv layers
        from typical_cnn import convolve_image, pool_maps
        current = cnn_input
        for filters in self.cnn.conv_layers:
            conv_out = convolve_image(current, filters, self.cnn.stride, relu)
            pooled = pool_maps(conv_out, self.cnn.pool_method, self.cnn.pool_size)
            current = pooled
        cache['flattened'] = current.flatten()
        z_hidden = np.dot(cache['flattened'], self.cnn.dense_hidden['weights']) + self.cnn.dense_hidden['biases']
        cache['hidden'] = relu(z_hidden)
        cache['z_hidden'] = z_hidden
        z_output = np.dot(cache['hidden'], self.cnn.dense_output['weights']) + self.cnn.dense_output['biases']
        cache['output'] = softmax(z_output)
        return cache['output'], cache

    def train_step(self, image, label):
        output, cache = self.forward_modern(image)
        target = one_hot(label, 10)
        loss = cross_entropy_loss(output, target)
        d_output = cross_entropy_derivative(output, target)
        d_weights_out = np.outer(cache['hidden'], d_output)
        d_biases_out = d_output
        self.cnn.dense_output['weights'] = self.optimizer.update(
            self.cnn.dense_output['weights'], d_weights_out, f'dense_out_w_{self.param_counter}'
        )
        self.cnn.dense_output['biases'] = self.optimizer.update(
            self.cnn.dense_output['biases'], d_biases_out, f'dense_out_b_{self.param_counter}'
        )
        d_hidden = np.dot(d_output, self.cnn.dense_output['weights'].T)
        d_hidden *= relu_derivative(cache['z_hidden'])
        d_weights_hidden = np.outer(cache['flattened'], d_hidden)
        d_biases_hidden = d_hidden
        self.cnn.dense_hidden['weights'] = self.optimizer.update(
            self.cnn.dense_hidden['weights'], d_weights_hidden, f'dense_hid_w_{self.param_counter}'
        )
        self.cnn.dense_hidden['biases'] = self.optimizer.update(
            self.cnn.dense_hidden['biases'], d_biases_hidden, f'dense_hid_b_{self.param_counter}'
        )
        self.param_counter += 1
        return loss

    def predict(self, image):
        out, _ = self.forward_modern(image)
        return np.argmax(out)

# --- New Model 2: CNN with Inhibition (Top-k Pooling) ---
class InhibitionOnlyModel:
    """CNN with conventional convolutions, but first pooling layer uses inhibition (top-k)."""
    def __init__(self, num_classes=10, conv2_filters=4, dense_size=16, lr=0.001, top_k=1):
        self.cnn = CNN(
            input_shape=(28, 28),
            num_classes=num_classes,
            conv_filters=[8, conv2_filters],
            filter_size=5,
            pool_size=2,
            dense_size=dense_size
        )
        self.top_k = top_k
        self.optimizer = AdamOptimizer(lr=lr)
        self.param_counter = 0
    def forward_with_inhibition(self, image):
        cache = {'input': image, 'conv_inputs': [], 'conv_outputs': [], 'pool_outputs': []}
        current = image
        from typical_cnn import convolve_image, pool_maps
        # First conv layer
        filters1 = self.cnn.conv_layers[0]
        cache['conv_inputs'].append(current)
        conv_out1 = convolve_image(current, filters1, self.cnn.stride, relu)
        cache['conv_outputs'].append(conv_out1)
        # Inhibition pooling: for each pixel, keep only the strongest activation across all maps
        pooled1 = np.zeros_like(conv_out1)
        # conv_out1 shape: (num_maps, H, W)
        num_maps, H, W = conv_out1.shape
        for y in range(H):
            for x in range(W):
                # Find the map with the strongest activation at (y, x)
                map_idx = np.argmax(conv_out1[:, y, x])
                pooled1[map_idx, y, x] = conv_out1[map_idx, y, x]
        # Now apply 2x2 max pooling
        pooled1 = pool_maps(pooled1, 'max', self.cnn.pool_size)
        cache['pool_outputs'].append(pooled1)
        # Second conv layer
        filters2 = self.cnn.conv_layers[1]
        cache['conv_inputs'].append(pooled1)
        conv_out2 = convolve_image(pooled1, filters2, self.cnn.stride, relu)
        cache['conv_outputs'].append(conv_out2)
        pooled2 = pool_maps(conv_out2, self.cnn.pool_method, self.cnn.pool_size)
        cache['pool_outputs'].append(pooled2)
        cache['flattened'] = pooled2.flatten()
        z_hidden = np.dot(cache['flattened'], self.cnn.dense_hidden['weights']) + self.cnn.dense_hidden['biases']
        cache['hidden'] = relu(z_hidden)
        cache['z_hidden'] = z_hidden
        z_output = np.dot(cache['hidden'], self.cnn.dense_output['weights']) + self.cnn.dense_output['biases']
        cache['output'] = softmax(z_output)
        return cache['output'], cache
    def train_step(self, image, label):
        output, cache = self.forward_with_inhibition(image)
        target = one_hot(label, 10)
        loss = cross_entropy_loss(output, target)
        # Adam optimizer for dense layers
        d_output = cross_entropy_derivative(output, target)
        d_weights_out = np.outer(cache['hidden'], d_output)
        d_biases_out = d_output
        self.cnn.dense_output['weights'] = self.optimizer.update(
            self.cnn.dense_output['weights'], d_weights_out, f'dense_out_w_{self.param_counter}'
        )
        self.cnn.dense_output['biases'] = self.optimizer.update(
            self.cnn.dense_output['biases'], d_biases_out, f'dense_out_b_{self.param_counter}'
        )
        d_hidden = np.dot(d_output, self.cnn.dense_output['weights'].T)
        d_hidden *= relu_derivative(cache['z_hidden'])
        d_weights_hidden = np.outer(cache['flattened'], d_hidden)
        d_biases_hidden = d_hidden
        self.cnn.dense_hidden['weights'] = self.optimizer.update(
            self.cnn.dense_hidden['weights'], d_weights_hidden, f'dense_hid_w_{self.param_counter}'
        )
        self.cnn.dense_hidden['biases'] = self.optimizer.update(
            self.cnn.dense_hidden['biases'], d_biases_hidden, f'dense_hid_b_{self.param_counter}'
        )
        self.param_counter += 1
        return loss
    def predict(self, image):
        out, _ = self.forward_with_inhibition(image)
        return np.argmax(out)

class StandardCNNModel:
    """Standard CNN using ReLU+Adam+Cross-Entropy"""
    def __init__(self, num_classes=10, conv2_filters=4, dense_size=16, initialize_filters=False, lr=0.001):
        # Reduced capacity to match challenge
        self.cnn = CNN(
            input_shape=(28, 28),
            num_classes=num_classes,
            conv_filters=[8, conv2_filters],  # Layer 1: 8, Layer 2: reduced
            filter_size=5,
            pool_size=2,
            dense_size=dense_size  # Reduced
        )
        
        if initialize_filters:
            self._init_filters()
            
        self.optimizer = AdamOptimizer(lr=lr)
        self.param_counter = 0
            
    def _init_filters(self):
        print("  Initializing Layer 1 with directional edge filters...")
        filters = self.cnn.conv_layers[0]
        
        filters[0] = np.array([
            [ 1,  1,  1,  1],
            [ 0,  0,  0,  0],
            [ 0,  0,  0,  0],
            [-1, -1, -1, -1]
        ], dtype=float)
        filters[1] = np.array([
            [ 1,  0,  0, -1],
            [ 1,  0,  0, -1],
            [ 1,  0,  0, -1],
            [ 1,  0,  0, -1]
        ], dtype=float)
        filters[2] = np.array([
            [ 2,  1, 0.5,  0],
            [ 1,  0.5, 0,  -0.5],
            [ 0.5, 0,  -0.5,  -1],
            [ 0,  -0.5,  -1,  -2]
        ], dtype=float)
        filters[3] = np.array([
            [ 0,  0.5,  1,  2],
            [ -0.5,  0,  0.5,  1],
            [-1,  -0.5, 0,  .5],
            [-2, -1,  -0.5,  0]
        ], dtype=float)
        
        for i in range(4):
            filters[i] -= filters[i].mean()

    def forward_modern(self, image):
        cache = {'input': image, 'conv_inputs': [], 'conv_outputs': [], 'pool_outputs': []}
        
        current = image
        from typical_cnn import convolve_image, pool_maps
        for filters in self.cnn.conv_layers:
            cache['conv_inputs'].append(current)
            conv_out = convolve_image(current, filters, self.cnn.stride, relu)
            cache['conv_outputs'].append(conv_out)
            pooled = pool_maps(conv_out, self.cnn.pool_method, self.cnn.pool_size)
            cache['pool_outputs'].append(pooled)
            current = pooled
        
        cache['flattened'] = current.flatten()
        z_hidden = np.dot(cache['flattened'], self.cnn.dense_hidden['weights']) + self.cnn.dense_hidden['biases']
        cache['hidden'] = relu(z_hidden)
        cache['z_hidden'] = z_hidden
        
        z_output = np.dot(cache['hidden'], self.cnn.dense_output['weights']) + self.cnn.dense_output['biases']
        cache['output'] = softmax(z_output)
        
        return cache['output'], cache
        
    def backward_modern(self, cache, target):
        output = cache['output']
        d_output = cross_entropy_derivative(output, target)
        
        # Update output layer
        d_weights_out = np.outer(cache['hidden'], d_output)
        d_biases_out = d_output
        
        self.cnn.dense_output['weights'] = self.optimizer.update(
            self.cnn.dense_output['weights'], d_weights_out, f'dense_out_w_{self.param_counter}'
        )
        self.cnn.dense_output['biases'] = self.optimizer.update(
            self.cnn.dense_output['biases'], d_biases_out, f'dense_out_b_{self.param_counter}'
        )
        
        # Update hidden layer
        d_hidden = np.dot(d_output, self.cnn.dense_output['weights'].T)
        d_hidden *= relu_derivative(cache['z_hidden'])
        
        d_weights_hidden = np.outer(cache['flattened'], d_hidden)
        d_biases_hidden = d_hidden
        
        self.cnn.dense_hidden['weights'] = self.optimizer.update(
            self.cnn.dense_hidden['weights'], d_weights_hidden, f'dense_hid_w_{self.param_counter}'
        )
        self.cnn.dense_hidden['biases'] = self.optimizer.update(
            self.cnn.dense_hidden['biases'], d_biases_hidden, f'dense_hid_b_{self.param_counter}'
        )
        
        self.param_counter += 1

    def train_step(self, image, label):
        output, cache = self.forward_modern(image)
        target = one_hot(label, 10)
        loss = cross_entropy_loss(output, target)
        self.backward_modern(cache, target)
        return loss

    def predict(self, image):
        out, _ = self.forward_modern(image)
        return np.argmax(out)

def train_and_eval(model, train_data, test_data, name="Model", epochs=30):
    print(f"\nTraining {name}...")
    images, labels = train_data
    
    start_time = time.time()
    for epoch in range(epochs):
        epoch_loss = 0
        indices = np.random.permutation(len(images))
        
        for i, idx in enumerate(indices):
            loss = model.train_step(images[idx], labels[idx])
            epoch_loss += loss
            
        avg_loss = epoch_loss/len(images)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")
        
    duration = time.time() - start_time
    print(f"  Training took {duration:.2f}s")
    
    # Evaluation
    test_imgs, test_lbls = test_data
    correct_by_digit = defaultdict(int)
    total_by_digit = defaultdict(int)
    
    for img, lbl in zip(test_imgs, test_lbls):
        pred = model.predict(img)
        total_by_digit[lbl] += 1
        if pred == lbl:
            correct_by_digit[lbl] += 1
            
    accuracy_per_digit = {}
    total_acc = 0
    total_count = 0
    
    for d in range(10):
        total = total_by_digit[d]
        correct = correct_by_digit[d]
        acc = (correct / total * 100) if total > 0 else 0
        accuracy_per_digit[d] = acc
        total_acc += correct
        total_count += total
        
    overall = (total_acc / total_count * 100) if total_count > 0 else 0
    print(f"  Overall Accuracy: {overall:.2f}%")
    
    return accuracy_per_digit, overall



def main():
    print("=" * 70)
    print("Challenge Benchmark: Reduced Capacity + Ablations (Modern Training)")
    print("=" * 70)
    print("Configuration:")
    print("  - Layer 2: 4 filters (reduced from 8)")
    print("  - FC Layer: 16 nodes (reduced from 32)")
    print("  - All models: ReLU + Adam + Cross-Entropy + LR=0.001")
    print("  - NDD-Gradients-Only: NDDs for gradients, no inhibition")
    print("  - Inhibition-Only: Top-k pooling, conventional convs")
    print("=" * 70)

    # Load Data
    try:
        train_set, test_set = load_data(max_samples=1000)
    except Exception as e:
        print(f"Start-up Failed: {e}")
        return

    # Init Models with reduced capacity
    print("\nInitializing Models...")
    ndd_model = NDDHybridModel(conv2_filters=4, dense_size=16, lr=0.001)
    cnn_model_random = StandardCNNModel(conv2_filters=4, dense_size=16, initialize_filters=False, lr=0.001)
    cnn_model_init = StandardCNNModel(conv2_filters=4, dense_size=16, initialize_filters=True, lr=0.001)
    ndd_grad_model = NDDGradientsOnlyModel(conv2_filters=4, dense_size=16, lr=0.001)
    inhibition_model = InhibitionOnlyModel(conv2_filters=4, dense_size=16, lr=0.001, top_k=1)

    # Check Parameters
    ndd_params = count_parameters(ndd_model.cnn)
    cnn_params = count_parameters(cnn_model_random.cnn)
    ndd_grad_params = count_parameters(ndd_grad_model.cnn)
    inhibition_params = count_parameters(inhibition_model.cnn)

    print(f"\nParameter Count:")
    print(f"  NDD-Hybrid Backend: {ndd_params} params")
    print(f"  Standard CNN Total: {cnn_params} params")
    print(f"  NDD-Gradients-Only: {ndd_grad_params} params")
    print(f"  Inhibition-Only: {inhibition_params} params")

    # Train
    epochs = 30
    results = {}
    models = [
        ("NDD-Hybrid (Modern)", ndd_model),
        ("Std-CNN Random (Modern)", cnn_model_random),
        ("Std-CNN Init (Modern)", cnn_model_init),
        ("NDD-Gradients-Only", ndd_grad_model),
        ("Inhibition-Only", inhibition_model)
    ]
    for name, model in models:
        acc_dict, overall = train_and_eval(model, train_set, test_set, name, epochs=epochs)
        results[name] = (acc_dict, overall)

    # Plot
    digits = range(10)
    plt.figure(figsize=(16, 7))
    x = np.arange(len(digits))
    width = 0.16

    ndd_res = results["NDD-Hybrid (Modern)"]
    rnd_res = results["Std-CNN Random (Modern)"]
    init_res = results["Std-CNN Init (Modern)"]
    ndd_grad_res = results["NDD-Gradients-Only"]
    inhibition_res = results["Inhibition-Only"]

    ndd_scores = [ndd_res[0][d] for d in digits]
    rnd_scores = [rnd_res[0][d] for d in digits]
    init_scores = [init_res[0][d] for d in digits]
    ndd_grad_scores = [ndd_grad_res[0][d] for d in digits]
    inhibition_scores = [inhibition_res[0][d] for d in digits]

    plt.bar(x - 2*width, ndd_scores, width, label=f'NDD-Hybrid ({ndd_res[1]:.1f}%)', color='royalblue')
    plt.bar(x - width, rnd_scores, width, label=f'Std-CNN Random ({rnd_res[1]:.1f}%)', color='gray')
    plt.bar(x, init_scores, width, label=f'Std-CNN Init ({init_res[1]:.1f}%)', color='forestgreen')
    plt.bar(x + width, ndd_grad_scores, width, label=f'NDD-Gradients-Only ({ndd_grad_res[1]:.1f}%)', color='orange')
    plt.bar(x + 2*width, inhibition_scores, width, label=f'Inhibition-Only ({inhibition_res[1]:.1f}%)', color='purple')

    plt.xlabel('Digit')
    plt.ylabel('Accuracy (%)')
    plt.title('Ablation Benchmark: NDD/Gradients/Inhibition (Conv2=4, FC=4, Modern Training)')
    plt.xticks(x, digits)
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # Save (do not overwrite previous results)
    plot_dir = os.path.join(base_dir, "nddplayground/benchmarking/plots")
    os.makedirs(plot_dir, exist_ok=True)
    output_path = os.path.join(plot_dir, "benchmark_ablation_modern.png")

    plt.savefig(output_path)
    print(f"\nExperiment Complete. Plot saved to: {output_path}")

if __name__ == "__main__":
    main()
