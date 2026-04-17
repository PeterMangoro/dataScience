Case Study: Representation Learning and Sequence Modeling in High-Dimensional Data
Assignment Type: Individual Case Study
Deadline: May 14, 2026 
Submission: Upload to LMS

Case Study Scenario
You are consulting for a technology company developing a system to analyze user behavior sequences (e.g., browsing sessions, transaction histories, or activity logs).

The company wants to:

Discover latent behavioral patterns using unsupervised learning
Build predictive models using neural networks
Explore whether Transformer-based models capture sequence structure more effectively
Your task is to design and evaluate a modeling pipeline that addresses these goals.

Dataset (Required)
You must use the following dataset:

Online Retail Dataset (UCI / Kaggle)
https://www.kaggle.com/datasets/carrie1/ecommerce-dataLinks to an external site.

Dataset Description
The dataset contains transactional records including:

InvoiceNo
StockCode
Description
Quantity
InvoiceDate
UnitPrice
CustomerID
Country
You must construct sequence-based representations at the customer level.

Required Analytical Tasks
1. Data Preprocessing & Representation Design
Construct a dataset suitable for modeling:

Aggregate transactions into customer-level sequences
Define a representation of each transaction (e.g., item embeddings, features)
Handle missing values and inconsistencies
Explain your design choices clearly.

2. Unsupervised Learning & Representation
Apply at least one unsupervised learning method to extract structure:

Examples:

clustering (K-means / DBSCAN)
representation learning (PCA or embedding methods)
Analyze:

whether meaningful customer segments emerge
how representation quality affects downstream modeling
3. Neural Network Modeling
Design and train a neural network model for a predictive task such as:

next-item prediction
purchase behavior classification
customer segmentation prediction
Requirements:

specify architecture
analyze training behavior (loss curves, overfitting)
discuss bias–variance tradeoff
4. Transformer-Based Modeling
Implement a basic Transformer-style model or use an existing library (e.g., PyTorch, HuggingFace).

Analyze:

how attention mechanisms model sequence dependencies
performance vs neural network baseline
interpretability of attention patterns
5. Comparative Analysis
Compare:

unsupervised representation vs raw features
neural network vs Transformer
Discuss:

performance differences
computational complexity
interpretability tradeoffs
Deliverables
1. Research Report (2–3 Pages Maximum)
Your report must include:

representation design and justification
insights from unsupervised learning
neural network training analysis
Transformer model behavior
comparative conclusions
The report should emphasize:

- clear reasoning
- methodological rigor
- interpretation of results

2. Python Analysis Code
Submit a Python script containing:

data preprocessing
feature construction
unsupervised learning implementation
neural network model
Transformer model
evaluation metrics
visualizations (training curves, clusters, etc.)
Your code must be:

well-organized
clearly commented
reproducible
Evaluation Criteria
Criterion	Weight
Representation design & preprocessing	20%
Unsupervised learning analysis	20%
Neural network implementation & reasoning	20%
Transformer model understanding	20%
Comparative analysis & insights	10%
Code quality & reproducibility	10%
Academic Integrity
This assignment is individual work.

You may consult:

research papers
documentation
textbooks
But all submitted work must reflect your own analysis and implementation.

Final Note
Modern data science systems rarely rely on a single method.

They combine:

unsupervised representation learning
neural network modeling
attention-based architectures
This assignment is designed to help you think across these layers — as a practicing data scientist or researcher would.