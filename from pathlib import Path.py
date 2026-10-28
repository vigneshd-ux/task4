from pathlib   import Path
import numpy as np
import torch
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
dtype = np.float32
y = np.array([[0], [1], [1], [0]], dtype=dtype)

def relux(x):
    return np.maximum(0, x)

def sigmoid(x):
    x = np.clip(x, -60.0, 60.0)
    return 1 / (1 + np.exp(-x))

def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    shifted = x - np.ma(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)

def mse(prediction, target):
    return np.mean((prediction - target) ** 2)

def cross_entropy(prediction, target, eps=1e-12):
    """Mean binary cross-entropy; prediction contains probabilities."""
    p = np.clip(np.asarray(prediction), eps, 1.0 - eps)
    t = np.asarray(target)
    return -np.mean(t * np.log(p) + (1.0 - t) * np.log(1.0 - p))

def train_xor(lr, epochs=5000, seed=0):
    rng = np.random.default_rng(seed)
    # Small random weights avoid starting all ReLU units on the same boundary.
    w1 = rng.normal(0.0, 0.8, size=(2, 4))
    b1 = np.zeros((1, 4))
    w2 = rng.normal(0.0, 0.8, size=(4, 1))
    b2 = np.zeros((1, 1))
    losses = []

    for _ in range(epochs):
        z1 = X @ w1 + b1
        hidden = relu(z1)
        prediction = sigmoid(hidden @ w2 + b2)
        losses.append(cross_entropy(prediction, y))

        # For sigmoid + binary cross-entropy, dL/dlogit = (p-y)/N.
        dlogit2 = (prediction - y) / len(X)
        dw2 = hidden.T @ dlogit2
        db2 = np.sum(dlogit2, axis=0, keepdims=True)
        dhidden = dlogit2 @ w2.T
        dz1 = dhidden * (z1 > 0.0)
        dw1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        w2 -= lr * dw2
        b2 -= lr * db2
        w1 -= lr * dw1
        b1 -= lr * db1

        if not all(np.all(np.isfinite(v)) for v in (w1, b1, w2, b2)):
            break

    hidden = relu(X @ w1 + b1)
    predictions = sigmoid(hidden @ w2 + b2).ravel()
    return np.asarray(losses), predictions

def autograd_check():
    w = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    x, target = 2.0, 10.0
    loss = (w * x + b - target) ** 2
    loss.backward()
    return w.grad.item(), b.grad.item()


def main():
    print("softmax([2.0, 1.0, 0.1]) =", np.round(softmax([2.0, 1.0, 0.1]), 3))
    print("mse([0.2, 0.8], [0, 1]) =", round(mse([0.2, 0.8], [0, 1]), 4))
    print("binary cross_entropy([0.2, 0.8], [0, 1]) =", round(cross_entropy([0.2, 0.8], [0, 1]), 4))

    # d/dw (wx+b-t)^2 = 2(wx+b-t)x; d/db = 2(wx+b-t).
    hand_w = 2 * (1 * 2 + 0 - 10) * 2
    hand_b = 2 * (1 * 2 + 0 - 10)
    grad_w, grad_b = autograd_check()
    print(f"Hand gradients: dw = {hand_w:.1f}, db = {hand_b:.1f}")
    print(f"Autograd: w.grad = {grad_w:.1f}, b.grad = {grad_b:.1f}")

    rates = [(0.01, "lr=0.01 (slow)"), (0.5, "lr=0.5 (requested)"), (5.0, "lr=5 (unstable)")]
    runs = {lr: train_xor(lr) for lr, _ in rates}
    pred = runs[0.5][1]
    print("XOR predictions after training:", np.round(pred, 2))
    print("Thresholded XOR classes:", (pred >= 0.5).astype(int))
    for lr, label in rates:
        losses = runs[lr][0]
        print(f"{label}: epochs recorded={len(losses)}, final BCE={losses[-1]:.4g}")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for lr, label in rates:
        losses = runs[lr][0]
        ax.plot(np.arange(1, len(losses) + 1), losses, label=label)
    ax.set(title="XOR training loss by learning rate", xlabel="Epoch", ylabel="Binary cross-entropy")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "task1_loss_curve.png"
    fig.savefig(out, dpi=160)
    print("Saved loss plot:", out)


    if __name__ == "__main__":
        main()