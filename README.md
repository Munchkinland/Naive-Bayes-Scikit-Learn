# Naive-Bayes-Scikit-Learn


README
This README provides an overview and documentation for the code snippets provided. Below is an explanation of each section:

Generate DataSets
This section demonstrates how to generate a synthetic dataset using scikit-learn's make_classification function. The dataset is then visualized using matplotlib. Subsequently, the data is split into training and testing sets using train_test_split.

Creating training models
Here, a Gaussian Naive Bayes model is trained on the generated dataset. The model's predictions are evaluated using metrics such as accuracy and F1 score.

Model evaluation using confusion matrix
A confusion matrix and its visualization are presented to assess the performance of the trained model. It includes metrics like True Positive, False Negative, False Positive, and True Negative.

Naive Bayes Classifier - Loading data
This part focuses on loading and exploring a real-world dataset ('loan_data.csv') using pandas. The data is then processed and visualized to understand its structure.

Processing data
Data preprocessing involves transforming categorical variables using one-hot encoding. The dataset is split into features (X) and the target variable (y) for further analysis.

Creating training models (Loan Data)
A Gaussian Naive Bayes model is trained on the loan dataset, and its performance is evaluated using accuracy and F1 score.

Model Evaluation Script (Loan Data)
A script for evaluating the performance of the Naive Bayes model on the loan dataset is provided, including metrics such as accuracy and F1 score.

True Positive, False Negative, False Positive, True Negative
This section interprets the results obtained from a confusion matrix, highlighting the meaning of True Positive, False Negative, False Positive, and True Negative instances.

Optimize GaussianNB model - Hyperparameter tuning
The Gaussian Naive Bayes model is optimized by tuning hyperparameters using GridSearchCV. The best hyperparameters and the model's improved performance are displayed.