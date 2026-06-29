# 🤝 K-Nearest Neighbors (KNN)

A complete tutorial on **K-Nearest Neighbors (KNN)**, covering the mathematical intuition behind the algorithm and its practical implementation using Python. This folder includes distance metric calculations, a custom KNN classifier built completely from scratch, and a comparison with Scikit-Learn using a real-world diabetes dataset.

---

## 📖 Overview

K-Nearest Neighbors (KNN) is one of the simplest and most intuitive supervised machine learning algorithms used for **classification** problems. Unlike many machine learning algorithms, KNN is a **non-parametric** and **instance-based** learning algorithm that classifies new data points based on the majority class of their nearest neighbors.

This tutorial is designed for beginners who want to understand how KNN works internally before using machine learning libraries such as Scikit-learn.

---

## 📂 Folder Structure

```text
KNN
│
├── Calculating Euclidean and Manhattan Distance.ipynb
├── K-Nearest Neighbors Classifier from scratch in Python.ipynb
├── KNN implementation from scratch.ipynb
├── KNN Classifier from SkLearn.ipynb
├── diabetes.csv
└── README.md
```

---

## 📚 Files Description

| File                                                            | Description                                                                                                                                                |
| --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Calculating Euclidean and Manhattan Distance.ipynb**          | Demonstrates the mathematical derivation and implementation of Euclidean and Manhattan distance metrics used by the KNN algorithm.                         |
| **K-Nearest Neighbors Classifier from scratch in Python.ipynb** | Implements the complete KNN classifier from scratch using Python and NumPy without relying on machine learning libraries.                                  |
| **KNN implementation from scratch.ipynb**                       | Demonstrates the practical implementation of the custom KNN classifier on the diabetes dataset, including preprocessing, prediction, and model evaluation. |
| **KNN Classifier from SkLearn.ipynb**                           | Implements the same classification problem using Scikit-Learn's `KNeighborsClassifier` for comparison.                                                     |
| **diabetes.csv**                                                | Pima Indians Diabetes dataset used to train and evaluate the KNN classifier.                                                                               |

---

## 🧠 Concepts Covered

* Supervised Machine Learning
* Classification Problems
* K-Nearest Neighbors (KNN)
* Instance-Based Learning
* Euclidean Distance
* Manhattan Distance
* Distance Metrics
* Majority Voting
* Model Training
* Prediction
* Model Evaluation

---

## 📐 Mathematical Overview

KNN classifies a new data point by finding the **K nearest training samples** using a distance metric and assigning the class that appears most frequently among those neighbors.

### 1. Euclidean Distance

Measures the straight-line distance between two points in Euclidean space.

$$
d(p,q)=\sqrt{\sum_{i=1}^{n}(p_i-q_i)^2}
$$

---

### 2. Manhattan Distance

Measures the distance by summing the absolute differences between corresponding coordinates.

$$
d(p,q)=\sum_{i=1}^{n}|p_i-q_i|
$$

---

### 3. KNN Prediction

The algorithm follows these steps:

1. Compute the distance between the query point and every training sample.
2. Sort all distances in ascending order.
3. Select the **K nearest neighbors**.
4. Determine the majority class among the selected neighbors.
5. Assign the majority class as the prediction.

---

## 💻 Custom KNN Class

The custom **`KNN_Classifier`** class includes the following methods:

| Method                                                          | Description                                                                                       |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| **`__init__(distance_metric)`**                                 | Initializes the classifier with either **Euclidean** or **Manhattan** distance.                   |
| **`get_distance_metric(training_data_point, test_data_point)`** | Computes the distance between a training sample and a test sample.                                |
| **`nearest_neighbors(X_train, test_data, k)`**                  | Computes distances for all training samples, sorts them, and returns the **K nearest neighbors**. |
| **`predict(X_train, test_data, k)`**                            | Predicts the final class label using majority voting among the nearest neighbors.                 |

---

## 📊 Dataset

The project uses the **Pima Indians Diabetes Dataset**, which contains several medical attributes used to predict whether a patient is diabetic.

**Input Features**

* Pregnancies
* Glucose
* BloodPressure
* SkinThickness
* Insulin
* BMI
* DiabetesPedigreeFunction
* Age

**Target**

* Outcome

  * **0** → Non-Diabetic
  * **1** → Diabetic

The objective is to classify each patient into one of two categories:

* Diabetic
* Non-Diabetic

---

## 🛠 Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-Learn
* Jupyter Notebook

---

## 👨‍💻 Author

**Shriram N**

B.Tech Artificial Intelligence and Machine Learning

Passionate about Machine Learning, Deep Learning, and Data Science.

---

⭐ If you found this tutorial helpful, consider giving the repository a star.
