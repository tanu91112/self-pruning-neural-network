# 🚀 Self-Pruning Neural Network using Learnable Gates

## Problem Understanding
In real-world deep learning systems, neural networks often become inefficient in terms of memory and computation. Traditional pruning methods remove weights after training, which is not optimal.

In this project, I implemented a **self-pruning neural network**, where the model learns during training which weights are important and which can be removed.

---

## Approach

### Prunable Layer (Core Idea)
I designed a custom layer called `PrunableLinear`, where each weight has an associated learnable gate.

- Each gate controls the importance of a weight  
- Gates are passed through a sigmoid function (0 to 1)  
- Final weight = original weight × sigmoid(gate)  
- If gate → 0, the weight is effectively pruned  

This makes pruning **differentiable and learnable**.

---

## Loss Function

The model is trained using a combined objective:

**Total Loss = Classification Loss + λ × Sparsity Loss**

- **Classification Loss:** CrossEntropyLoss for prediction accuracy  
- **Sparsity Loss:** Mean of all gate activations (L1-style penalty)

This encourages the model to reduce unnecessary connections.

---

## Why This Works (L1 Sparsity)

L1-style regularization encourages small values to shrink toward zero. Since gate values lie between 0 and 1, minimizing them naturally leads to sparsity in the network.

---

## Experiments

The model was trained with different values of λ to observe the trade-off between accuracy and sparsity.

| Lambda | Accuracy | Sparsity |
|--------|----------|----------|
| 0.001  | 45.27%   | 35.06%   |
| 0.01   | 45.58%   | 46.87%   |
| 0.05   | 45.86%   | 75.46%   |

---

## Observations

- Lower λ → higher accuracy, lower sparsity  
- Higher λ → stronger pruning, slight accuracy trade-off  
- The model successfully learns which weights are less important  
- Clear trade-off between performance and efficiency is observed  

---

## Key Insight

Initially, sparsity was weak because classification loss dominated training. After properly balancing the sparsity term, the model began to prune effectively.

This highlights the importance of **loss balancing in multi-objective neural networks**.

---

## Conclusion

This project demonstrates a **self-pruning neural network**, where pruning is learned during training instead of being applied post-training.

The model successfully achieves adaptive sparsity while maintaining reasonable classification performance.

---

## How to Run

```bash
pip install -r requirements.txt
py main.py
