"""Training script for the LUT-based MNIST classifier."""

import os
import sys
import numpy as np

# Resolve paths relative to this script so it runs from any working directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from lut_model import LUTModel
from main import load_mnist


def train(model, images, labels, epochs=5, print_every=500):
    """SGD training loop.

    LR follows learning_rate(t) schedule (main.c:35), with t counting
    globally across all epochs so the schedule doesn't reset each epoch.
    Prints loss + accuracy every print_every steps.
    """
    n = len(images)
    global_step = 0

    for epoch in range(epochs):
        indices = np.random.permutation(n)
        epoch_loss    = 0.0
        epoch_correct = 0

        for i, idx in enumerate(indices):
            global_step += 1
            loss, correct = model.step(images[idx], labels[idx], t=global_step)
            epoch_loss    += loss
            epoch_correct += correct

            if global_step % print_every == 0:
                avg_loss = epoch_loss / (i + 1)
                acc      = epoch_correct / (i + 1) * 100
                print(f"  epoch {epoch+1}/{epochs} step {i+1:>6}/{n} "
                      f"| loss {avg_loss:.4f} | acc {acc:.1f}%")

        avg_loss = epoch_loss / n
        acc      = epoch_correct / n * 100
        print(f"  Epoch {epoch+1}/{epochs} complete | loss {avg_loss:.4f} | acc {acc:.1f}%")

    return avg_loss, acc


def evaluate(model, images, labels):
    """Accuracy over the given set. Returns float in [0, 1]."""
    correct = 0
    for image, label in zip(images, labels):
        logits, _ = model.forward(image)
        if int(np.argmax(logits)) == label:
            correct += 1
    accuracy = correct / len(images)
    print(f"Test Accuracy: {accuracy * 100:.2f}% ({correct}/{len(images)})")
    return accuracy


def main():
    print("=" * 50)
    print("LUT Model - MNIST Training")
    print("=" * 50)

    data_dir = os.path.join(_SCRIPT_DIR, "mnist", "mnist")
    train_images, train_labels, test_images, test_labels = load_mnist(
        data_dir=data_dir, max_samples=10000
    )

    print("\nInitializing LUT model...")
    model = LUTModel()

    print("\nTraining...")
    train(model, train_images, train_labels, epochs=5)

    print("\nEvaluating on test set...")
    evaluate(model, test_images, test_labels)

    print("\n" + "=" * 50)
    print("Done!")
    print("=" * 50)


if __name__ == "__main__":
    main()
