# Restaurant Rating Prediction
Objective: Build a machine learning model to predict the aggregate rating of a restaurant based on other features.

---

This project leverages machine learning to predict restaurant ratings based on various features such as cuisine type, location, pricing, and additional services like online delivery and table booking. Using a linear regression model, I explored the impact of different factors on ratings and evaluated the model’s performance. This README details the project setup, data preprocessing, model selection, evaluation metrics, and results interpretation.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Workflow](#project-workflow)
- [Data Preprocessing](#data-preprocessing)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Results and Analysis](#results-and-analysis)
- [Future Improvements](#future-improvements)
- [Conclusions](#conclusions)

---

## Project Overview

The goal of this project was to develop a predictive model for restaurant ratings based on a variety of features. I used regression techniques to capture the relationships between the restaurant's attributes and its rating. This model could help restaurant owners, data analysts, and food industry professionals understand which factors contribute most to customer ratings.

## Dataset

The dataset contains various attributes related to restaurants, including:
- `Restaurant ID`: Unique identifier for each restaurant
- `Restaurant Name`: Name of the restaurant
- `Country Code`, `City`, `Address`: Location details
- `Cuisines`: Types of cuisine offered
- `Average Cost for Two`: Average price range for dining
- `Has Table Booking`, `Has Online Delivery`: Availability of additional services
- `Price Range`: Price tier
- `Aggregate Rating`: Target variable for predicting ratings
- `Votes`: Number of customer votes on ratings

There were a few missing values in the `Cuisines` column, which I addressed during preprocessing.

## Installation

To run this project, follow these steps:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/restaurant-rating-prediction.git
   cd restaurant-rating-prediction
   ```

2. **Install Required Libraries**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

3. **Upload the Dataset**  
   Upload the dataset to Google Colab or your local environment where you will be running the code.

## Project Workflow

1. **Data Preprocessing**: Cleaned and prepared the data for modelling.
2. **Exploratory Data Analysis (EDA)**: Gained insights into distributions and relationships.
3. **Feature Engineering**: Encoded categorical features for model compatibility.
4. **Model Training and Evaluation**: Trained and evaluated a linear regression model.
5. **Interpretation**: Analyzed the most influential features and assessed model reliability.

## Data Preprocessing

1. **Handling Missing Values**: I checked for missing values and found that the `Cuisines` column had a few missing entries. I filled these with a placeholder (`"Unknown"`) to retain the records.
   
2. **Encoding Categorical Variables**: Categorical variables like `City`, `Cuisines`, and `Has Table Booking` were encoded into numerical format using one-hot encoding to make them suitable for the regression model.

3. **Data Splitting**: I split the dataset into training and testing sets (80% for training and 20% for testing) to evaluate the model’s generalization performance.

## Modeling

I experimented with Linear Regression as an initial model. The steps included:
- **Model Selection**: Linear Regression was chosen to capture the relationship between features and the restaurant ratings.
- **Training**: I trained the model using the training dataset.
- **Hyperparameter Tuning**: I considered adding regularization techniques like Ridge Regression to address overfitting but retained the simple linear regression model for this iteration.

## Evaluation

The model’s performance was evaluated using **Mean Squared Error (MSE)** and **R-squared** on both training and testing data:

1. **Mean Squared Error (MSE)**: Measures the average squared difference between actual and predicted ratings. Lower values indicate better performance.
2. **R-squared**: Indicates how well the model explains the variance in ratings. Values close to 1 represent a good fit, while negative values indicate overfitting.

### Evaluation Results

- **Training Set**:
  - MSE: 2.58e-06
  - R-squared: 0.99999 (indicating the model fits the training data almost perfectly)

- **Testing Set**:
  - MSE: 3,387,635.09
  - R-squared: -1,488,343.85 (indicating poor generalization to unseen data)

These results show that the model fits the training data well but suffers from significant overfitting, as evidenced by the poor performance on the test set.

## Results and Analysis

To understand what influences restaurant ratings the most, I examined the model’s coefficients. Here are a few findings:

1. **Cuisine Type**: Certain types of cuisine showed strong associations with higher ratings.
2. **Price Range**: Higher price ranges were generally associated with better ratings, potentially due to perceived quality.
3. **Location (City)**: Geographical differences influenced ratings, likely due to varying customer expectations.
4. **Availability of Services**: Features like `Has Table Booking` and `Has Online Delivery` impacted ratings, with some services possibly linked to convenience and overall customer satisfaction.

These insights can help restaurant owners and analysts make strategic decisions to improve customer experience.

## Future Improvements

1. **Regularization**: Using Ridge or Lasso regression could help mitigate overfitting and produce more reliable feature importance.
2. **Feature Selection**: Reducing dimensionality by selecting only the most relevant features could improve model performance.
3. **Experimenting with Different Models**: Trying models like Decision Trees, Random Forests, or Gradient Boosting could capture more complex patterns in the data.
4. **Cross-Validation**: To ensure robust results, cross-validation could help validate the model across multiple data splits.

## Conclusions

This project was an exploratory effort to predict restaurant ratings based on multiple features. Despite overfitting in the initial model, I gained valuable insights into influential factors affecting customer ratings. With further refinement, this model could serve as a tool for restaurant analysis and customer experience improvement.

Thank you for exploring this project with me! If you have any questions, please feel free to reach out. 

--- 


