# credit-risk-classification
Supervised Learning MOD 20 (week 20)

This Challenge will use various techniques to train and evaluate a model based on loan risk. A dataset of historical lending activity from a peer-to-peer lending services company will be used to build a model that can identify the credit worthiness of borrowers.

This challange is divided in Three sections:

1. Split the Data into Training and Testing Sets
2. Create a Logistic Regression Model with the Original Data
3. Write a Credit Risk Analysis Report

## Part 1 Split the Data into Training and Testing Sets
1. Read the lending_data.csv data from the Resources folder into a Pandas DataFrame.
2. Create the labels set (y) from the “loan_status” column, and then create the features (X) DataFrame from the remaining columns.

note
A value of 0 in the “loan_status” column means that the loan is healthy. A value of 1 means that the loan has a high risk of defaulting.

3. Split the data into training and testing datasets by using train_test_split.

## Part 2 Create a Logistic Regression Model with the Original Data
1. Fit a logistic regression model by using the training data (X_train and y_train).
2. Save the predictions for the testing data labels by using the testing feature data (X_test) and the fitted model.
3. Evaluate the model’s performance by doing the following:
    a. Generate a confusion matrix.
    b. Print the classification report.
4. How well does the logistic regression model predict both the 0 (healthy loan) and 1 (high-risk loan) labels?

## Part 3 Write a Credit Risk Analysis Report
Write a brief report that includes a summary and analysis of the performance of the machine learning models that you used in this homework. You should write this report as the README.md file included in your GitHub repository.

Structure your report by using the report template that Starter_Code.zip includes, ensuring that it contains the following:
1. An overview of the analysis: Explain the purpose of this analysis.
2. The results: Using a bulleted list, describe the accuracy score, the precision score, and recall score of the machine learning model.
3. A summary: Summarize the results from the machine learning model. Include your justification for recommending the model for use by the company. If you don’t recommend the model, justify your reasoning.