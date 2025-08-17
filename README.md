MicroGrad (from scratch in Python)

A tiny autograd engine + neural net library, implemented from scratch in pure Python.  
Inspired by [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd), but written completely by me for learning purposes.

This repo shows how backpropagation actually works under the hood — no magic, just Python classes, operator overloading, and a little calculus.



##  Features
- `Value` class for scalar values with automatic differentiation
- Operator overloading: `+`, `-`, `*`, `/`, `**`, `tanh`, (and easy to extend with more)
- Reverse-mode autodiff with a minimal computation graph
- `Neuron`, `Layer`, and `MLP` classes for building simple neural networks
- Toy dataset + training loop to prove it learns

---

##  Example
```python
from micrograd import Value, MLP

# Create a tiny 3-input, 2-hidden, 1-output MLP
n = MLP(3, [4, 4, 1])

# Forward pass
x = [2.0, 3.0, -1.0]
y_pred = n(x)

# Compute a loss (mean squared error vs. target)
target = 1.0
loss = (y_pred - target) ** 2

# Backpropagation
loss.backward()

# Update params
for p in n.parameters():
    p.data -= 0.05 * p.grad
📓 Notebook
Open the notebook to see:

Step-by-step implementation of Value and backprop

Manual gradient checks

Training a neural net on a tiny dataset

bash
Copy
Edit
jupyter notebook MicroGrad.ipynb
🔬 Results
On a toy dataset with labels in { -1, 1 }, the MLP reduces loss from ~6.3 to <0.3 within a few epochs.

📦 Requirements
Python 3.8+

Jupyter Notebook (optional)

🤔 Why?
This is not meant to compete with PyTorch or TensorFlow.
It’s meant to:

Demystify backpropagation

Show how deep learning libraries work internally

Serve as a playground for experimenting with autodiff and neural nets

🛠 Future Ideas
Add ReLU, sigmoid, softmax

Add proper datasets (MNIST, etc.)

Implement Xavier/He initialization

Switch to math.tanh for numerical stability

📜 License
MIT License. Use it, hack it, learn from it.
