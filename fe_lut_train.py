"""Training script for the FE→LUT hybrid MNIST classifier."""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

# Import load_mnist first, before fe_lut_model modifies sys.path (mnist_pipeline.py
# inserts feature_engineering/ at position 0 as a side-effect of being imported).
from main import load_mnist
from fe_lut_model import FELUTModel, N_CLASSES


def plot_confusion_matrix(model, images, labels, epoch):
    """Build and save a confusion matrix for the given set."""
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for image, label in zip(images, labels):
        logits, _, _, _ = model.forward(image)
        pred = int(np.argmax(logits))
        cm[label, pred] += 1

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix — FE-LUT Epoch {epoch}')
    plt.colorbar(im, ax=ax)

    thresh = cm.max() / 2
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black', fontsize=8)

    plt.tight_layout()
    out_dir = os.path.join(_SCRIPT_DIR, 'visualizations', 'fe_lut')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'confusion_epoch{epoch:02d}.png')
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"  Confusion matrix saved to {path}")
    return cm


def train(model, images, labels, epochs=5, print_every=500):
    """SGD training loop (mirrors lut_train.py)."""
    n = len(images)
    global_step = 0

    for epoch in range(epochs):
        indices = np.random.permutation(n)
        epoch_loss    = 0.0
        epoch_correct = 0
        epoch_local   = None
        epoch_global  = None
        epoch_output  = 0.0
        seq2_local    = None
        seq2_global   = None
        seq2_output   = 0.0

        for i, idx in enumerate(indices):
            global_step += 1
            loss, correct, sc, seq2 = model.step(images[idx], labels[idx], t=global_step)
            epoch_loss    += loss
            epoch_correct += correct

            if epoch_local is None:
                epoch_local  = [0.0] * len(sc['local'])
                epoch_global = [0.0] * len(sc['global'])
                seq2_local   = [0.0] * len(seq2['local'])
                seq2_global  = [0.0] * len(seq2['global'])

            for li, s in enumerate(sc['local']):
                epoch_local[li] += s
            for gi, s in enumerate(sc['global']):
                epoch_global[gi] += s
            epoch_output += sc['output']

            for li, c in enumerate(seq2['local']):
                seq2_local[li] += c
            for gi, c in enumerate(seq2['global']):
                seq2_global[gi] += c
            seq2_output += seq2['output']

            if global_step % print_every == 0:
                n_seen        = i + 1
                avg_loss      = epoch_loss / n_seen
                acc           = epoch_correct / n_seen * 100
                local_str     = '+'.join(f"{s/n_seen:.0f}" for s in epoch_local)
                global_str    = '+'.join(f"{s/n_seen:.0f}" for s in epoch_global)
                out_str       = f"{epoch_output/n_seen:.0f}"
                total         = (sum(epoch_local) + sum(epoch_global) + epoch_output) / n_seen
                seq2_l_str    = '+'.join(f"{s/n_seen:.0f}" for s in seq2_local)
                seq2_g_str    = '+'.join(f"{s/n_seen:.0f}" for s in seq2_global)
                seq2_out_str  = f"{seq2_output/n_seen:.0f}"
                seq2_total    = (sum(seq2_local) + sum(seq2_global) + seq2_output) / n_seen
                print(f"  epoch {epoch+1}/{epochs} step {i+1:>6}/{n} "
                      f"| loss {avg_loss:.4f} | acc {acc:.1f}% "
                      f"| spikes/img local[{local_str}] "
                      f"global[{global_str}] out[{out_str}] "
                      f"total={total:.0f} "
                      f"| seq2s/img local[{seq2_l_str}] "
                      f"global[{seq2_g_str}] out[{seq2_out_str}] "
                      f"total={seq2_total:.0f}")

        n_seen       = n
        avg_loss     = epoch_loss / n_seen
        acc          = epoch_correct / n_seen * 100
        local_str    = '+'.join(f"{s/n_seen:.0f}" for s in epoch_local)
        global_str   = '+'.join(f"{s/n_seen:.0f}" for s in epoch_global)
        out_str      = f"{epoch_output/n_seen:.0f}"
        total        = (sum(epoch_local) + sum(epoch_global) + epoch_output) / n_seen
        seq2_l_str   = '+'.join(f"{s/n_seen:.0f}" for s in seq2_local)
        seq2_g_str   = '+'.join(f"{s/n_seen:.0f}" for s in seq2_global)
        seq2_out_str = f"{seq2_output/n_seen:.0f}"
        seq2_total   = (sum(seq2_local) + sum(seq2_global) + seq2_output) / n_seen
        print(f"  Epoch {epoch+1}/{epochs} complete "
              f"| loss {avg_loss:.4f} | acc {acc:.1f}% "
              f"| spikes/img local[{local_str}] "
              f"global[{global_str}] out[{out_str}] "
              f"total={total:.0f} "
              f"| seq2s/img local[{seq2_l_str}] "
              f"global[{seq2_g_str}] out[{seq2_out_str}] "
              f"total={seq2_total:.0f}")

        plot_confusion_matrix(model, images, labels, epoch + 1)

    return avg_loss, acc


def evaluate(model, images, labels):
    """Accuracy over the given set."""
    correct = 0
    for image, label in zip(images, labels):
        logits, _, _, _ = model.forward(image)
        if int(np.argmax(logits)) == label:
            correct += 1
    accuracy = correct / len(images)
    print(f"Test Accuracy: {accuracy * 100:.2f}% ({correct}/{len(images)})")
    return accuracy


def main():
    print("=" * 50)
    print("FE-LUT Hybrid Model - MNIST Training")
    print("=" * 50)

    data_dir = os.path.join(_SCRIPT_DIR, "mnist", "mnist")
    train_images, train_labels, test_images, test_labels = load_mnist(
        data_dir=data_dir, max_samples=10000
    )

    print("\nInitializing FE-LUT model...")
    model = FELUTModel()

    print("\nTraining...")
    train(model, train_images, train_labels, epochs=5)

    print("\nEvaluating on test set...")
    evaluate(model, test_images, test_labels)

    print("\n" + "=" * 50)
    print("Done!")
    print("=" * 50)


if __name__ == "__main__":
    main()
