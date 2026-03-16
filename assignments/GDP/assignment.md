Case Study: Understanding a Data Generating Process Through Modeling
Course: Foundations and Frontiers of Data Science
Assignment Type: Individual Case Study
Deadline: March 31, 2026 (12:00 PM EST)
Submission: Upload to Canvas

Assignment Objective
The goal of this assignment is to evaluate whether you can connect theoretical principles from the course to a real dataset.

You will analyze a dataset and investigate:

the data generating process (DGP)

uncertainty and statistical estimation

feature construction

bias–variance behavior

interpretability of linear models

effects of dimensionality reduction

Rather than focusing on algorithm performance alone, your analysis must demonstrate reasoning about how the data was generated and how modeling assumptions affect inference.

Case Study Scenario
You are hired as a data scientist by a policy research institute investigating factors influencing urban housing prices.

Your task is to determine whether a linear model with engineered features can provide interpretable insights about housing prices, while maintaining reasonable predictive performance.

You must also investigate how dimensionality reduction techniques alter interpretability and prediction error.

Dataset
Use the California Housing Dataset (available in sklearn.datasets.fetch_california_housing) or another public housing price dataset of comparable complexity.

Your dataset must contain:

at least 8–10 predictors

continuous variables

at least 10,000 observations

Required Analytical Tasks
Your analysis must include the following steps.

1. Data Generating Process Hypothesis
Propose a plausible data generating process of the form:


Discuss:

potential sources of noise

possible omitted variables

sampling biases

2. Exploratory Data Analysis
Perform EDA to investigate:

marginal distributions

conditional relationships

correlation structure

Include at least one statistical argument about the structure of 
.

3. Feature Engineering
Construct at least three engineered features, such as:

interaction terms

nonlinear transformations

logarithmic transformations

Evaluate whether these transformations improve model performance or interpretability.

4. Statistical Estimation and Bias–Variance Considerations
Fit at least two competing models, such as:

simple linear regression

polynomial or interaction model

regularized regression (Ridge or LASSO)

Evaluate:

training error

cross-validated error

Discuss how model complexity affects bias and variance.

5. Dimensionality Reduction
Apply Principal Component Analysis (PCA).

Evaluate:

explained variance ratio

predictive performance using reduced features

Discuss how dimensionality reduction affects interpretability.

Deliverables
You must submit two components.

1. Research Report (Maximum 500 Characters)
Your report must summarize:

the hypothesized data generating process

the most important predictors

the effect of feature engineering

the bias–variance tradeoff observed

the impact of PCA on interpretability

The report must be clear, concise, and logically argued.

Maximum length:

500 characters (strict limit).

2. Python Analysis Code
Attach a Python file or notebook containing:

data loading

EDA

feature engineering

model estimation

PCA analysis

evaluation metrics

reproducible results

Your code should be clearly commented.

Evaluation Criteria
Your work will be assessed on the following dimensions:

Criterion	Weight
Conceptual reasoning about DGP	25%
Statistical analysis and modeling	25%
Feature engineering design	20%
Bias–variance interpretation	15%
Code quality and reproducibility	15%
Academic Integrity
This assignment is individual work - NOT a team-based project.

You may consult documentation, textbooks, and research papers, but all submitted code and reasoning must be your own.

Any use of external code must be properly cited.

Final Reminder
This assignment is not about achieving the lowest prediction error.

It is about demonstrating that you understand how statistical reasoning, modeling assumptions, and feature representation interact in data science practice.