"""
Phase 2 continued: a tiny neural network, built on top of engine.py.

Structure (this mirrors PyTorch's design exactly, just tinier):
  - Neuron: one unit. Holds weights + a bias, computes weighted sum + activation.
  - Layer:  a group of Neurons that all see the same inputs.
  - MLP:    a stack of Layers ("Multi-Layer Perceptron") — a full network.

Then we train this tiny network on a toy dataset using gradient descent:
  1. Forward pass: feed inputs through the network, get predictions.
  2. Compute loss: how wrong are the predictions?
  3. Backward pass: call loss.backward() to compute every gradient.
  4. Update: nudge every weight slightly opposite its gradient (this is
     literally "learning" — moving weights in the direction that reduces
     error).
  5. Repeat.

Requires engine.py to be in the same folder (imports Value from it).
"""

from __future__ import annotations
import random
from engine import Value


class Module:
    """Base class: shared zero_grad() behavior for Neuron/Layer/MLP."""

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self) -> list[Value]:
        return []


class Neuron(Module):
    def __init__(self, num_inputs: int, nonlin: bool = True):
        # Random small initial weights — this is standard practice; the
        # network starts out "knowing nothing" and learns from there.
        self.w = [Value(random.uniform(-1, 1)) for _ in range(num_inputs)]
        self.b = Value(0.0)
        self.nonlin = nonlin  # whether to apply an activation function

    def __call__(self, x: list[Value]) -> Value:
        # Weighted sum of inputs, plus bias: w1*x1 + w2*x2 + ... + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act

    def parameters(self) -> list[Value]:
        return self.w + [self.b]

    def __repr__(self):
        kind = "TanH" if self.nonlin else "Linear"
        return f"{kind}Neuron({len(self.w)})"


class Layer(Module):
    def __init__(self, num_inputs: int, num_outputs: int, **kwargs):
        self.neurons = [Neuron(num_inputs, **kwargs) for _ in range(num_outputs)]

    def __call__(self, x: list[Value]) -> list[Value]:
        outputs = [n(x) for n in self.neurons]
        return outputs

    def parameters(self) -> list[Value]:
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    """A Multi-Layer Perceptron: a stack of Layers.

    Example: MLP(3, [4, 4, 1]) builds a network that takes 3 inputs,
    passes them through two hidden layers of 4 neurons each, and produces
    1 output. The final layer has no activation (raw output, common for
    regression/classification scores).
    """

    def __init__(self, num_inputs: int, layer_sizes: list[int]):
        sizes = [num_inputs] + layer_sizes
        self.layers = [
            Layer(sizes[i], sizes[i + 1], nonlin=(i != len(layer_sizes) - 1))
            for i in range(len(layer_sizes))
        ]

    def __call__(self, x: list[Value]) -> list[Value] | Value:
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"


if __name__ == "__main__":
    # Training demo — run this file directly:
    #   python engine/nn.py
    #
    # Toy dataset: tiny binary classification problem. Each input is a
    # 3-number vector, each label is +1 or -1. There's no real-world
    # meaning here — the point is watching the network's loss go down as
    # it learns to fit these examples.

    random.seed(42)  # reproducible results

    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]  # desired outputs

    model = MLP(3, [4, 4, 1])  # 3 inputs -> 4 -> 4 -> 1 output
    print(model)
    print(f"Total trainable parameters: {len(model.parameters())}\n")

    learning_rate = 0.05
    num_epochs = 100

    for epoch in range(num_epochs):
        # --- forward pass ---
        y_pred = [model(x) for x in xs]

        # Mean squared error loss: average of (predicted - actual)^2
        loss = sum((yp - yt) ** 2 for yp, yt in zip(y_pred, ys)) / len(ys)

        # --- backward pass ---
        model.zero_grad()  # clear old gradients before computing new ones
        loss.backward()

        # --- update weights (gradient descent) ---
        for p in model.parameters():
            p.data -= learning_rate * p.grad

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"epoch {epoch:3d}  loss = {loss.data:.4f}")

    print("\nFinal predictions vs targets:")
    for x, yt in zip(xs, ys):
        pred = model(x)
        print(f"  input={x}  target={yt:+.1f}  predicted={pred.data:+.4f}")

    print("\nIf loss dropped substantially and predictions are close to")
    print("their targets (same sign, reasonably close magnitude), training worked.")