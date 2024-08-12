
# House Price Prediction - Machine Learning Project

This repository contains a Jupyter Notebook where I have developed a machine learning project to predict house prices using the Boston Housing Dataset. The project demonstrates the entire process from understanding the data to building, evaluating, and saving the predictive model.

## Project Overview

The goal of this project is to predict the prices of houses using various features in the dataset. Although the project is built on a small dataset, the steps involved are applicable to larger datasets as well.

### Dataset

The dataset used is the Boston Housing Dataset, which contains features like crime rate, number of rooms, age of the house, and more. The dataset is loaded from a CSV file.

### Steps Involved

1. **Understanding the Dataset**:
   - Loaded the dataset and used methods like `describe()`, `head()`, and `info()` to get a better understanding of the features.

2. **Data Visualization**:
   - Visualized the features using histograms to understand their distribution.

3. **Data Splitting**:
   - Performed a train-test split using `train_test_split` from sklearn.
   - Used `StratifiedShuffleSplit` to ensure that important features are equally distributed between the train and test sets.

4. **Data Exploration**:
   - Plotted a scatter matrix to visualize the relationships between features.

5. **Data Preparation**:
   - Split the data into `housing` (features) and `housing_labels` (target variable).
   - Learned how to handle missing values using different techniques.
   - Used `SimpleImputer` from sklearn to fill in missing values.
   - Applied feature scaling using pipelines.

6. **Model Selection**:
   - Worked on three different models:
     - `LinearRegression`
     - `DecisionTreeRegressor`
     - `RandomForestRegressor`
   - Among these, `RandomForestRegressor` provided the best results.

7. **Model Evaluation**:
   - Used cross-validation to evaluate the model and prevent overfitting.

8. **Model Saving**:
   - Saved the trained model using `joblib` for future use.

9. **Model Testing**:
   - Tested the model on the test data to evaluate its performance.

10. **Model Demonstration**:
    - Demonstrated how to use the trained model for predictions on new data.

## How to Run the Project

1. Clone this repository.
2. Open the Jupyter Notebook file (`House_Price_Prediction.ipynb`).
3. Follow the steps provided in the notebook to understand the project and run the code.

## Requirements

- Python 3.x
- Jupyter Notebook
- Scikit-learn
- Pandas
- Matplotlib
- Joblib

You can install the required libraries using:

```bash
pip install -r requirements.txt
