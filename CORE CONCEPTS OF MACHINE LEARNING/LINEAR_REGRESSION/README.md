# 📈 Linear Regression

A complete tutorial on **Linear Regression**, covering both the mathematical intuition behind the algorithm and its practical implementation using Python. This folder contains a step-by-step implementation of Linear Regression from scratch, followed by applying the algorithm to a real-world salary dataset.

---

## 📖 Overview

Linear Regression is one of the most fundamental supervised machine learning algorithms used for predicting continuous numerical values. It models the relationship between an independent variable (feature) and a dependent variable (target) by fitting the best possible straight line through the data.

This tutorial is designed for beginners who want to understand how Linear Regression works internally before using machine learning libraries such as Scikit-learn.

---

## 📂 Folder Structure

```text
LINEAR_REGRESSION
│
├── Building Linear Regression from scratch in python.ipynb
├── Implementing Linear Regression from scratch using python.ipynb
├── salary_data.csv
└── README.md
```

---

## 📚 Files Description

| File                                                               | Description                                                                                                                                                                                                 |
| ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Building Linear Regression from scratch in python.ipynb**        | Implements the Linear Regression algorithm from scratch using Python, explaining the mathematical concepts behind linear equations, Mean Squared Error (MSE), gradient descent, and parameter optimization. |
| **Implementing Linear Regression from scratch using python.ipynb** | Demonstrates the practical implementation of Linear Regression using a salary dataset, including preprocessing, model training, prediction, and visualization of the regression line.                       |
| **salary_data.csv**                                                | Dataset containing Years of Experience and corresponding Salary values used to train and evaluate the Linear Regression model.                                                                              |

---

## 🧠 Concepts Covered

* Supervised Machine Learning
* Regression Problems
* Simple Linear Regression
* Linear Equation
* Independent and Dependent Variables
* Mean Squared Error (MSE)
* Cost Function
* Gradient Descent
* Parameter Optimization
* Model Training
* Prediction
* Data Visualization

---

## 📐 Mathematical Overview

Linear Regression aims to find the **line of best fit** by minimizing the **Mean Squared Error (MSE)** between the predicted and actual target values.

### 1. Model Equation

The Linear Regression model is represented as:

```math
Y = wX + b
```

where:

* **w** = Weight (Slope)
* **b** = Bias (Intercept)
* **X** = Input feature
* **Y** = Predicted output

---

### 2. Loss Function (Mean Squared Error)

To measure the prediction error, Linear Regression minimizes the Mean Squared Error (MSE):

```math
L=\frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i-y_i)^2
```

where:

* **m** = Number of training samples
* **ŷᵢ** = Predicted value
* **yᵢ** = Actual value

---

### 3. Gradient Calculation

The gradients of the loss function with respect to the model parameters are calculated as:

**Weight Gradient**

```math
dw=-\frac{2}{m}X^T(Y-\hat{Y})
```

**Bias Gradient**

```math
db=-\frac{2}{m}\sum_{i=1}^{m}(y_i-\hat{y}_i)
```

---

### 4. Parameter Updates

Using Gradient Descent, the parameters are updated iteratively as:

```math
w=w-\alpha\cdot dw
```

```math
b=b-\alpha\cdot db
```

where **α** is the learning rate.

---

## 💻 Custom Linear Regression Class

The custom **`Linear_Regression`** class includes the following methods:

| Method                                          | Description                                                                        |
| ----------------------------------------------- | ---------------------------------------------------------------------------------- |
| **`__init__(learning_rate, no_of_iterations)`** | Initializes the learning rate and the number of training iterations.               |
| **`fit(X, Y)`**                                 | Trains the Linear Regression model using Gradient Descent on the training dataset. |
| **`update_weights()`**                          | Computes the gradients (`dw`, `db`) and updates the model parameters (`w`, `b`).   |
| **`predict(X)`**                                | Computes predictions using the Linear Regression equation **`Y = X · w + b`**.     |

---

## 📊 Dataset

The project uses a **Salary Dataset** containing:

* Years of Experience
* Salary

The objective is to predict an employee's salary based on their years of professional experience.

**Dataset Summary**

* **Samples:** 30
* **Feature:** `YearsExperience`
* **Target:** `Salary`
* **Train-Test Split:** 67% Training, 33% Testing

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
