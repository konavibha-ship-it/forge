"""
Phase 2: A tiny autograd engine, built entirely from scratch.

This is the core idea behind every deep learning framework (PyTorch,
TensorFlow, etc.): wrap numbers in an object that REMEMBERS how it was
computed, so you can automatically walk backward through those operations
to compute gradients (derivatives) — which is exactly what "training" a
neural network means: nudging numbers in the direction that reduces error.

The `Value` class below wraps a single number. Every operation (+, *, **,
etc.) creates a new Value that remembers its "parents" and how to compute
its local gradient. Calling `.backward()` on the final result walks the
whole computation graph backward (this is called "backpropagation") and
fills in `.grad` on every Value that contributed to it.

This design follows the same approach as Andrej Karpathy's "micrograd"
(free on GitHub/YouTube) — well worth watching after you get this working,
it explains the "why" behind every line here in more depth.
"""

from __future__ import annotations
import math


class Value:
    """A single scalar value that tracks its own gradient."""

    def __init__(self, data: float, _children: tuple = (), _op: str = ""):
        self.data = data
        self.grad = 0.0

        # Internal bookkeeping for backpropagation — not meant to be used
        # directly outside this class.
        self._backward = lambda: None   # how to propagate gradient to parents
        self._prev = set(_children)     # which Values created this one
        self._op = _op                  # what operation created this one (for debugging)

    def __add__(self, other: "Value | float") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(out)/d(self) = 1, d(out)/d(other) = 1
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: "Value | float") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # d(out)/d(self) = other.data, d(out)/d(other) = self.data
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, power: float) -> "Value":
        assert isinstance(power, (int, float)), "only supporting int/float powers"
        out = Value(self.data ** power, (self,), f"**{power}")

        def _backward():
            # d(out)/d(self) = power * self.data ** (power - 1)
            self.grad += (power * self.data ** (power - 1)) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> "Value":
        """ReLU activation: max(0, x). The standard nonlinearity in neural nets."""
        out = Value(0.0 if self.data < 0 else self.data, (self,), "ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Value":
        """Tanh activation, squashes values to the range (-1, 1)."""
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        """Compute gradients for every Value that led to this one."""
        # Build a topological order of the computation graph (parents
        # before children), so we process each node only after everything
        # that depends on it has already been processed.
        topo: list[Value] = []
        visited: set[Value] = set()

        def build_topo(v: "Value"):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # The gradient of the output with respect to itself is 1.
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    # --- convenience operators so Values behave like normal numbers ---
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other if isinstance(other, Value) else -other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other ** -1

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


if __name__ == "__main__":
    # Quick manual demo — run this file directly:
    #   python engine/engine.py
    #
    # We build a tiny expression: L = (a * b + c).tanh()
    # then call .backward() and confirm the gradients make sense.

    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)

    d = a * b          # d = -6
    e = d + c           # e = 4
    L = e.tanh()         # L = tanh(4) ≈ 0.9993

    print(f"L = {L.data:.4f}")

    L.backward()

    print(f"dL/da = {a.grad:.4f}")
    print(f"dL/db = {b.grad:.4f}")
    print(f"dL/dc = {c.grad:.4f}")

    # Sanity check against numerical differentiation (nudge a tiny bit,
    # see how much L changes, and confirm it roughly matches a.grad).
    h = 1e-5
    a2 = Value(2.0 + h)
    L2 = (a2 * b + c).tanh()
    numerical_grad = (L2.data - L.data) / h

    print(f"\nNumerical check: dL/da ≈ {numerical_grad:.4f} (should be close to {a.grad:.4f})")
    assert abs(numerical_grad - a.grad) < 1e-2, "Gradient check failed!"
    print("Gradient check passed — backward() is computing correct derivatives.")