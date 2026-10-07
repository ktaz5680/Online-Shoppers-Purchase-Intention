# Online-Shoppers-Purchase-Intention

## Project Overview

This project examines online shopping behavior and whether browsing patterns can help predict purchasing intention. The analysis also looks at whether engineered engagement features improve prediction and which variables are most important.

## Dataset Information

The project uses the UCI Online Shoppers Purchasing Intention dataset. It contains 12,330 shopping sessions and 18 variables.

The target variable is Revenue, which shows whether a session resulted in a purchase.

There are 10,422 non-purchasing sessions and 1,908 purchasing sessions. The dataset has no missing values. There are also 125 rows with identical values, which were retained in the analysis.

The dataset is stored in the data folder as:

`online_shoppers_intention.csv`

## Research Questions

RQ1: What differences in browsing and engagement behavior exist between sessions that result in a purchase and those that do not?

RQ2: To what extent do engineered measures of visitor engagement improve the prediction of purchasing intention?

RQ3: Which original and engineered behavioral features contribute most to predicting purchasing intention?

## CRISP-DM Process

### Business Understanding

The goal of the project is to better understand online customer behavior and determine which browsing characteristics are associated with purchasing intention.

### Data Understanding

The dataset was examined for its size, variables, data types, missing values, identical rows, and the distribution of the Revenue variable.

### Data Preparation

Revenue was converted to a binary outcome. Weekend was converted to a binary variable. Categorical variables were one-hot encoded.

The data were divided into 80% training data and 20% testing data using stratified sampling.

Four engineered features were also created: TotalPages, TotalDuration, AverageTimePerPage, and ProductPageShare.

### Modeling

Three Random Forest models were developed.

The first model used the original features.

The second model used only the engineered features.

The third model used the original and engineered features together.

All three models used 500 trees, a random state of 42, and balanced class weights.

### Evaluation

The models were evaluated using accuracy, precision, recall, F1 score, ROC-AUC, and PR-AUC.

Permutation importance was also used to identify the most important predictors.

### Deployment

The project files, code, notebook, data, figures, and results are organized in this GitHub repository so the analysis can be reviewed and reproduced.

## Repository Structure

The repository contains the following folders:

`data` contains the dataset.

`Scripts` contains the Python scripts for the CRISP-DM steps.

`Notebooks` contains the original Jupyter notebook.

`Outputs` contains the figures and project results.

`requirements.txt` contains the Python packages needed to run the project.

`README.md` explains the project and how it is organized.

## How to Run the Project

Install the required packages using:

```bash
pip install -r requirements.txt
