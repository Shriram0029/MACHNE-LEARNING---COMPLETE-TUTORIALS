# 📉 Logistic Regression

A complete tutorial on **Logistic Regression**, covering both the mathematical intuition behind the algorithm and its practical implementation using Python. This folder contains a step-by-step implementation of Logistic Regression from scratch, followed by applying the algorithm to a real-world diabetes dataset.

---

## 📖 Overview

Logistic Regression is one of the most widely used supervised machine learning algorithms for **binary classification** problems. It predicts the probability of an input belonging to a particular class using the **Sigmoid (Logistic) Function**, making it suitable for tasks where the output is categorical.

This tutorial is designed for beginners who want to understand how Logistic Regression works internally before using machine learning libraries such as Scikit-learn.

---

## 📂 Folder Structure

```text
LOGISTIC_REGRESSION
│
├── Building Logistic Regression from scratch in python.ipynb
├── Implementing Logistic Regression from scratch in python.ipynb
├── diabetes.csv
└── README.md
```

---

## 📚 Files Description

| File                                                              | Description                                                                                                                                                                                             |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Building Logistic Regression from scratch in python.ipynb**     | Implements the Logistic Regression algorithm from scratch using Python, explaining the mathematical concepts behind the sigmoid function, binary cross-entropy loss, and gradient descent optimization. |
| **Implementing Logistic Regression from scratch in python.ipynb** | Demonstrates the practical implementation of Logistic Regression on the diabetes dataset, including preprocessing, model training, prediction, and evaluation.                                          |
| **diabetes.csv**                                                  | Pima Indians Diabetes dataset used to train and evaluate the Logistic Regression model.                                                                                                                 |

---

## 🧠 Concepts Covered

* Supervised Machine Learning
* Classification Problems
* Binary Classification
* Logistic Regression
* Sigmoid Function
* Probability Estimation
* Binary Cross-Entropy Loss
* Gradient Descent
* Decision Boundary
* Classification Threshold
* Model Training
* Prediction
* Model Evaluation

---

## 📐 Mathematical Overview

Logistic Regression estimates the probability of an input belonging to a class by passing a linear combination of input features through the **Sigmoid (Logistic) Function**.

### 1. Sigmoid Function

The Sigmoid function maps any real-valued input into a probability between 0 and 1.

```math
\sigma(z)=\frac{1}{1+e^{-z}}
```

---

### 2. Model Prediction Equation

The predicted probability is computed as:

```math
\hat{Y}=\sigma(w\cdot X+b)=\frac{1}{1+e^{-(w\cdot X+b)}}
```

where:

* **w** = Weight vector
* **b** = Bias term
* **X** = Input feature vector
* **Ŷ** = Predicted probability

---

### 3. Binary Cross-Entropy Loss

To optimize the model parameters, Logistic Regression minimizes the Binary Cross-Entropy (Log Loss) function:

```math
L=-\frac{1}{m}\sum_{i=1}^{m}\left[y_i\log(\hat{y}_i)+(1-y_i)\log(1-\hat{y}_i)\right]
```

where:

* **m** = Number of training samples
* **yᵢ** = Actual class label
* **ŷᵢ** = Predicted probability

---

### 4. Gradient Calculation

The gradients used for parameter optimization are:

**Weights Gradient**

```math
dw=\frac{1}{m}X^T(\hat{Y}-Y)
```

**Bias Gradient**

```math
db=\frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i-y_i)
```

---

### 5. Parameter Updates

The model parameters are updated using Gradient Descent:

```math
w=w-\alpha\cdot dw
```

```math
b=b-\alpha\cdot db
```

where **α** is the learning rate.

---

## 💻 Custom Logistic Regression Class

The custom **`Logistic_Regression`** class includes the following methods:

| Method                                          | Description                                                                                            |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **`__init__(learning_rate, no_of_iterations)`** | Initializes the learning rate and the number of training iterations.                                   |
| **`fit(X, Y)`**                                 | Trains the Logistic Regression model by repeatedly applying gradient descent on the training dataset.  |
| **`update_weights()`**                          | Computes predicted probabilities, calculates gradients (`dw`, `db`), and updates the model parameters. |
| **`predict()`**                                 | Predicts binary class labels by applying a threshold of **0.5** on the predicted probabilities.        |

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
