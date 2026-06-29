# 🛡️ Support Vector Machine (SVM) Classifier

A complete tutorial on **Support Vector Machine (SVM)**, covering both the mathematical intuition behind the algorithm and its practical implementation using Python. This folder contains a step-by-step implementation of an SVM classifier from scratch, followed by applying the algorithm to a real-world diabetes dataset.

---

## 📖 Overview

Support Vector Machine (SVM) is a powerful supervised machine learning algorithm primarily used for **classification** tasks. It works by finding the optimal decision boundary, known as the **hyperplane**, that maximizes the margin between different classes. SVM is particularly effective for high-dimensional datasets and classification problems where a clear separation between classes exists.

This tutorial is designed for beginners who want to understand how SVM works internally before using machine learning libraries such as Scikit-learn.

---

## 📂 Folder Structure

```text
SUPPORT_VECTOR_MACHINE
│
├── Building SVM Classifier from scratch.ipynb
├── Implementing SVM Classifier from Scratch in Python.ipynb
├── diabetes.csv
└── README.md
```

---

## 📚 Files Description

| File                                                         | Description                                                                                                                                                                                   |
| ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Building SVM Classifier from scratch.ipynb**               | Implements the Support Vector Machine classifier from scratch using Python, explaining the mathematical concepts behind hyperplanes, support vectors, hinge loss, and parameter optimization. |
| **Implementing SVM Classifier from Scratch in Python.ipynb** | Demonstrates the practical implementation of an SVM classifier on the diabetes dataset, including preprocessing, model training, prediction, and evaluation.                                  |
| **diabetes.csv**                                             | Pima Indians Diabetes dataset used to train and evaluate the SVM classifier.                                                                                                                  |

---

## 🧠 Concepts Covered

* Supervised Machine Learning
* Classification Problems
* Binary Classification
* Support Vector Machine (SVM)
* Hyperplane
* Support Vectors
* Maximum Margin Classification
* Decision Boundary
* Hinge Loss
* Gradient Descent Optimization
* Model Training
* Prediction
* Model Evaluation

---

## 📐 Mathematical Overview

The objective of an SVM classifier is to determine the **optimal hyperplane** that maximizes the margin between two classes while minimizing classification errors.

### 1. Hyperplane Decision Boundary

The decision boundary is represented as:

```math
y = w \cdot x - b
```

where:

* **w** = Weight vector
* **b** = Bias term
* **x** = Input feature vector
* **y** = Predicted class label (−1 or 1)

---

### 2. Hinge Loss Function

To penalize incorrect classifications and maximize the decision margin, SVM minimizes the following objective function:

```math
L = \lambda \|w\|^2 + \frac{1}{m}\sum_{i=1}^{m}\max(0,\;1-y_i(w\cdot x_i-b))
```

where **λ** is the regularization parameter controlling the trade-off between margin size and classification errors.

---

### 3. Gradient Calculation

For every training sample, SVM first checks whether the sample satisfies the margin condition:

```math
y_i(w \cdot x_i - b) \ge 1
```

If the condition is satisfied:

```math
dw = 2\lambda w
```

```math
db = 0
```

Otherwise:

```math
dw = 2\lambda w - y_i x_i
```

```math
db = y_i
```

---

### 4. Parameter Updates

Using Gradient Descent, the model parameters are updated as:

```math
w = w - \alpha \cdot dw
```

```math
b = b - \alpha \cdot db
```

where **α** is the learning rate.

---

## 💻 Custom SVM Class Implementation

The custom **`SVM_classifier`** class includes the following methods:

| Method                                                            | Description                                                                                                      |
| ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **`__init__(learning_rate, no_of_iterations, lambda_parameter)`** | Initializes the learning rate, number of iterations, and regularization parameter.                               |
| **`fit(X, Y)`**                                                   | Converts labels from `{0,1}` to `{-1,1}` and trains the classifier by updating the model parameters iteratively. |
| **`update_weights()`**                                            | Computes gradients based on the hinge loss condition and updates the weight vector and bias.                     |
| **`predict(X)`**                                                  | Computes the decision function `X · w − b` and predicts the corresponding class labels.                          |

---

## 📊 Dataset

The project uses the **Pima Indians Diabetes Dataset**, which contains several medical attributes used to predict whether a patient is diabetic.

The objective is to classify each patient into one of two categories:

* Diabetic
* Non-Diabetic

---

## 🛠 Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Jupyter Notebook

---

## 👨‍💻 Author

**Shriram N**

B.Tech Artificial Intelligence and Machine Learning

Passionate about Machine Learning, Deep Learning, and Data Science.

---

⭐ If you found this tutorial helpful, consider giving the repository a star.
