Case Study: Classification Modeling, Validation, and Representation Learning in Clinical Data
Assignment Type: Individual Case Study
Deadline: April 20, 2026 (12:00 PM EST)
Submission: Upload to Canvas
Assignment Objective
This assignment evaluates your ability to apply classification methods, rigorous model validation, data leakage detection, and unsupervised representation learning to a real-world healthcare dataset.

Your analysis must demonstrate:

sound statistical reasoning

careful model evaluation

explicit data leakage prevention

thoughtful representation analysis

The goal is not simply to achieve the highest predictive accuracy, but to demonstrate methodologically sound data science practice.

Case Study Scenario
You are consulting for a healthcare analytics startup developing a predictive system to identify patients at risk of Type 2 diabetes using clinical diagnostic measurements.

The company needs an interpretable and statistically reliable model that can support clinical decision-making.

In addition to supervised classification, the company is interested in whether unsupervised representation learning reveals latent structure in the patient population.

Required Dataset
You must use the following dataset:

Pima Indians Diabetes Database (Kaggle version)

Dataset link:

https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-databaseLinks to an external site.

Dataset file:

diabetes.csv
Dataset Description
The dataset contains medical diagnostic measurements for female patients of Pima Indian heritage aged 21 or older.

Total observations:

768

Number of features:

8 predictors + 1 target variable

Variables in the Dataset
Variable	Description
Pregnancies	Number of times pregnant
Glucose	Plasma glucose concentration after oral glucose tolerance test
BloodPressure	Diastolic blood pressure (mm Hg)
SkinThickness	Triceps skin fold thickness (mm)
Insulin	2-hour serum insulin (mu U/ml)
BMI	Body mass index (weight in kg/(height in m)^2)
DiabetesPedigreeFunction	Genetic predisposition score
Age	Patient age (years)
Outcome	Diabetes diagnosis (1 = diabetic, 0 = non-diabetic)
Important Data Considerations
Several variables contain zero values that likely represent missing or invalid measurements, including:

Glucose

BloodPressure

SkinThickness

Insulin

BMI

You must decide how to handle these observations and justify your choice statistically.

Required Analytical Tasks
Your case study must include the following analyses.

1. Exploratory Data Analysis (EDA)
Perform exploratory analysis to examine:

variable distributions

potential missing or invalid values

class imbalance

correlations among predictors

Explain how these characteristics influence modeling decisions.

2. Classification Modeling
Implement at least two classification models, such as:

logistic regression

random forest

support vector machine

gradient boosting

Evaluate predictive performance using appropriate metrics, such as:

accuracy

ROC-AUC

precision-recall

F1 score

Explain why the selected metrics are appropriate for this dataset.

3. Model Validation Strategy
Design a validation procedure that properly estimates generalization performance, such as:

k-fold cross-validation

train/test split with hyperparameter tuning

Explain why your validation strategy is statistically appropriate.

4. Data Leakage Investigation
Carefully evaluate the analysis pipeline for data leakage, including:

preprocessing performed before splitting data

scaling applied to the entire dataset

target leakage through feature construction

Explain how your workflow prevents leakage.

5. Unsupervised Learning & Representation
Apply at least one unsupervised learning technique, such as:

Principal Component Analysis (PCA)

K-means clustering

t-SNE or UMAP

Investigate whether the learned representations:

reveal meaningful structure among patients

improve classification performance when used as input features

Deliverables
You must submit two components.

1. Research Report (Maximum 500 Characters)
Provide a concise research summary explaining:

the best-performing classification model

how the validation strategy was implemented

whether leakage risks were identified

insights obtained from unsupervised representation analysis

Maximum length:

500 characters (strict limit).

The report must demonstrate clear reasoning despite the length constraint.

2. Python Analysis Code
Submit a Python notebook (.ipynb) or Python script (.py) containing:

dataset loading

preprocessing and missing value handling

exploratory data analysis

classification models

validation pipeline

leakage prevention strategy

unsupervised learning analysis

performance evaluation

Your code must be clear, commented, and reproducible.

Evaluation Criteria
Criterion	Weight
Implementation of classification models	25%
Model validation design	25%
Identification and prevention of data leakage	25%
Unsupervised representation analysis	15%
Code clarity and reproducibility	10%
Academic Integrity
This assignment must be completed individually.

You may consult documentation, textbooks, and research papers, but all submitted work must be your own analysis and code.

External code must be properly cited.

Final Note
Many real-world machine learning failures arise not from weak algorithms but from incorrect validation procedures and hidden data leakage.

This assignment is designed to test your ability to conduct methodologically rigorous data science analysis.