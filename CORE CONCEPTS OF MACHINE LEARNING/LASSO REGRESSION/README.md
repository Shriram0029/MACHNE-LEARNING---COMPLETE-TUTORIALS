# 📉 Lasso Regression (L1 Regularization)

A complete tutorial on **Lasso Regression (L1 Regularization)**, covering both the mathematical intuition behind the algorithm and its practical implementation using Python. This folder contains a step-by-step implementation of Lasso Regression from scratch, followed by applying the algorithm to a real-world salary dataset.

---

## 📖 Overview

Lasso Regression (Least Absolute Shrinkage and Selection Operator) is a regularized version of Linear Regression that introduces an **L1 penalty** on the model coefficients. This regularization technique shrinks less important feature weights toward zero, helping reduce overfitting while also performing automatic feature selection.

This tutorial is designed for beginners who want to understand how Lasso Regression works internally before using machine learning libraries such as Scikit-learn.

---

## 📂 Folder Structure

```text
LASSO_REGRESSION
│
├── Building LASSO REGRESSION model from scratch in python.ipynb
├── Implementing Lasso Regression from Scratch.ipynb
├── salary_data.csv
└── README.md
```

---

## 📚 Files Description

| File                                                             | Description                                                                                                                                                                                                 |
| ---------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Building LASSO REGRESSION model from scratch in python.ipynb** | Implements the Lasso Regression algorithm from scratch using Python, explaining the mathematical concepts behind L1 regularization, Mean Squared Error (MSE), gradient descent, and parameter optimization. |
| **Implementing Lasso Regression from Scratch.ipynb**             | Demonstrates the practical implementation of Lasso Regression using a salary dataset, including data preprocessing, model training, prediction, and visualization.                                          |
| **salary_data.csv**                                              | Dataset containing Years of Experience and corresponding Salary values used to train and evaluate the Lasso Regression model.                                                                               |

---

## 🧠 Concepts Covered

* Supervised Machine Learning
* Regression Problems
* Linear Regression
* Lasso Regression (L1 Regularization)
* Mean Squared Error (MSE)
* L1 Regularization
* Cost Function
* Gradient Descent
* Weight Shrinkage
* Feature Selection
* Parameter Optimization
* Model Training
* Prediction

---

## 📐 Mathematical Overview

Lasso Regression extends Linear Regression by adding an **L1 regularization term** to the Mean Squared Error (MSE) loss function. This penalty encourages the model to shrink unnecessary feature weights toward zero, producing simpler and more interpretable models.

### 1. Model Equation

The prediction equation is:

$$
Y = wX + b
$$

where:

* **w** = Weight vector
* **b** = Bias term
* **X** = Input feature vector
* **Y** = Predicted output

---

### 2. Loss Function with L1 Regularization

The objective function minimized by Lasso Regression is:

$$
L = \frac{1}{m}\sum_{i=1}^{m}(\hat{y}*i-y_i)^2+\lambda\sum*{j=1}^{n}|w_j|
$$

where:

* **m** = Number of training samples
* **ŷᵢ** = Predicted value
* **yᵢ** = Actual value
* **λ** = Regularization parameter controlling the amount of weight shrinkage

---

### 3. Gradient Calculation

Since the L1 penalty depends on the absolute value of each weight, its derivative changes according to the sign of the weight.

**Weight Gradient**

For **w > 0**

$$
dw=\frac{-2(X^T(Y-\hat{Y}))+\lambda}{m}
$$

For **w ≤ 0**

$$
dw=\frac{-2(X^T(Y-\hat{Y}))-\lambda}{m}
$$

**Bias Gradient**

$$
db=-\frac{2}{m}\sum_{i=1}^{m}(y_i-\hat{y}_i)
$$

---

### 4. Parameter Updates

Using Gradient Descent, the model parameters are updated as:

$$
w=w-\alpha\cdot dw
$$

$$
b=b-\alpha\cdot db
$$

where **α** is the learning rate.

---

## 💻 Custom Lasso Regression Class

The custom **`Lasso_Regression`** class includes the following methods:

| Method                                                            | Description                                                                                                 |
| ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **`__init__(learning_rate, no_of_iterations, lambda_parameter)`** | Initializes the learning rate, number of training iterations, and L1 regularization parameter.              |
| **`fit(X, Y)`**                                                   | Trains the Lasso Regression model using Gradient Descent on the training dataset.                           |
| **`update_weights()`**                                            | Computes the gradients, applies the L1 regularization penalty, and updates the model parameters (`w`, `b`). |
| **`predict(X)`**                                                  | Computes predictions using the Lasso Regression equation **`Y = X · w + b`**.                               |

---

## 📊 Dataset

The project uses a **Salary Dataset** containing:

* Years of Experience
* Salary

The objective is to predict an employee's salary based on their years of professional experience while demonstrating the effect of **L1 Regularization** on model coefficients.

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
