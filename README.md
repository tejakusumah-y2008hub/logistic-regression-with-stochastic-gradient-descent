# Logistic Regression with Stochastic Gradient Descent (From Scratch)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NumPy](https://img.shields.io/badge/Library-NumPy-orange)
![Status](https://img.shields.io/badge/Status-Complete-green)

## 📌 Project Overview
This repository contains a custom implementation of **Logistic Regression** built entirely from scratch using Python and NumPy. 

Unlike standard library implementations (like Scikit-Learn), this project manually implements the optimization logic to demonstrate a deep understanding of the underlying mathematics. Key features include:
* **Stochastic Gradient Descent (SGD):** For efficient training on large datasets.
* **Elastic Regularization:** Custom implementation of both **L1 (Lasso)** and **L2 (Ridge)** penalties to prevent overfitting.
* **Learning Rate Decay:** A simulated annealing approach (learning schedule) to ensure convergence.

## 🧠 The Mathematics
The core objective was to move beyond the "Black Box" of machine learning and implement the derivation manually.

### 1. Hypothesis (Sigmoid Function)
The model uses the sigmoid function to map predictions to probabilities between 0 and 1:
$$h_\theta(x) = \frac{1}{1 + e^{-z}}$$
Where $z = \theta^T x$.

### 2. Optimization (Stochastic Gradient Descent)
Instead of processing the entire dataset at once (Batch Gradient Descent), this implementation updates weights iteratively for **single training examples**. This allows the model to escape local minima and handle data that doesn't fit in memory.

The update rule for a single sample $(x^{(i)}, y^{(i)})$ is:
$$\theta_j := \theta_j - \eta \cdot \nabla J(\theta)$$

Where $\eta$ is the learning rate, which decays over time according to a cooling schedule:
$$\eta(t) = \frac{\eta_0}{t + t_1}$$

### 3. Regularization (The "Secret Sauce")
To prevent overfitting, I implemented regularization terms directly into the gradient calculation. This penalizes extreme weights, keeping the model generalizable.

**L2 Regularization (Ridge):**
Adds a penalty equivalent to the square of the magnitude of coefficients.
$$\text{Gradient penalty} = \lambda \cdot \theta$$

**L1 Regularization (Lasso):**
Adds a penalty equivalent to the absolute value of the magnitude of coefficients. This promotes sparsity (feature selection).
$$\text{Gradient penalty} = \lambda \cdot \text{sign}(\theta)$$

---

## 🛠️ Installation & Usage

### Prerequisites
```bash
pip install numpy matplotlib jupyter