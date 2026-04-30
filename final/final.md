Option 4: Amazon Reviews 
https://www.kaggle.com/datasets/bittlingmayer/amazonreviewsLinks to an external site.

Focus: NLP, embeddings, Transformers

Dataset Selection Expectations
You must clearly justify:

why this dataset is appropriate
what problem you are solving
what assumptions you are making
Project Components 
1. Data Generating Process (DGP)
What to do
Formulate:


What to think about
What is the underlying mechanism generating the data?
What variables are missing (latent variables)?
Is your model causal or purely predictive?
What biases exist in data collection?
Expected depth
Graduate-level responses should go beyond description and include critical reasoning about assumptions.

2. Exploratory Data Analysis (EDA)
What to do
visualize distributions
analyze correlations
identify anomalies
What to think about
Does the data match your DGP assumptions?
Are relationships linear or nonlinear?
Are there subpopulations?
Common mistake to avoid
EDA is not just plotting — it must include interpretation.

3. Feature Engineering
What to do
Create:

interaction terms
nonlinear transformations
temporal features (if applicable)
aggregated features
What to think about
How does each feature affect model bias and variance?
Does this improve interpretability or harm it?
Are you introducing leakage?
Expected depth
Explain why each feature exists, not just how it is computed.

4. Unsupervised Learning & Representation
What to do
Apply at least one method:

clustering (K-means, DBSCAN, hierarchical)
representation learning (embeddings, etc.)
What to think about
What structure is being discovered?
Is it stable across parameter choices?
Does it align with domain intuition?
Advanced expectation
Connect unsupervised results to downstream supervised modeling.

5. Supervised Learning Models
Required
linear model (interpretability baseline)
one classical ML model
one neural network
What to think about
What tradeoffs exist between models?
Where do linear models fail?
When do complex models overfit?
6. Neural Networks: Training & Practice
What to do
define architecture
track loss curves
analyze training behavior
What to think about
Are you overfitting?
Is optimization stable?
How sensitive is performance to hyperparameters?
Expected depth
Discuss training dynamics, not just final accuracy.

7. Transformer Model (Required for Option 1 & 4)
What to do
implement or adapt a Transformer
use attention-based modeling
What to think about
What dependencies are being captured?
Does attention align with intuition?
Is the added complexity justified?
8. Model Evaluation & Validation
What to do
design proper train/test split
use cross-validation
select appropriate metrics
What to think about
Are your metrics aligned with the problem?
Are you measuring generalization or just fit?
Is your evaluation reproducible?
9. Data Leakage Analysis (CRITICAL)
What to do
Explicitly identify and prevent:

preprocessing leakage
feature leakage
temporal leakage
What to think about
Could future information leak into training?
Are transformations applied correctly?
Important
Failure here significantly reduces project score.

10. Comparative Analysis & Recommendation
What to do
Compare all approaches.

What to think about
Which model is best and why?
What tradeoffs exist?
What would you deploy in practice?
Deliverables
1. Team Presentation (7–12 Pages)
Must demonstrate:

reasoning
methodology
interpretation
2. In-Class Presentation
Requirements
20 minutes per team (15 min for presentation + 5 min for Q&A)
all members participate
Must include
problem framing
key decisions
results
tradeoffs
What will be evaluated
clarity
depth
ability to answer questions
Evaluation Criteria 
Criterion	Weight
DGP reasoning & assumptions	15%
Feature engineering depth	15%
Quantitative Model Development & Insight	25%
Evaluation & validation rigor	15%
Data leakage awareness	5%
Presentation quality	25%
Common Pitfalls to Avoid
treating this as a Kaggle competition
ignoring assumptions
weak validation design
lack of interpretation
over-reliance on black-box models
Final Note
A strong project will demonstrate:

technical competence
statistical thinking
clear reasoning
ability to justify decisions
Your goal is not to build the most complex model — your goal is to build the most defensible system.