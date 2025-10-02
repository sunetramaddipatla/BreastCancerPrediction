# BreastCancerPrediction
We gathered and cleaned a dataset of breast cancer data and trained 10 different models to determine optimal predictor of breast cancer. We achieved F1 and AUC scores of 0.99+. We used PyPlot to visualize ROC and Precision/ Recall curves

The objective of this project is to create a breast cancer prediction model using the UCI Breast Cancer Wisconsin Diagonistic dataset in order to categorize observations from a breast cancer image into benign or malignant, comparing the use of a Decision Tree model vs. a Logistic Regression model in Python. Using an accurate breast cancer prediction model can help support and guide doctors in providing better, quicker care to their patients during critical times or when patient volume is high.

📁 File

Breast_Cancer_Prediction_Team_5G.ipynb: Complete notebook including data preprocessing, visualizations, model training, and evaluation.
📊 Dataset

Source: UCI Breast Cancer Wisconsin Dataset
Samples: 570 rows
Features: 33 columns (30 feature measurements + diagnosis + ID + unnamed column)
Target: Diagnosis (M = malignant, B = benign)
Class Distribution:
Malignant (1): 212 cases (37%)
Benign (0): 357 cases (63%)
Feature Types: 10 characteristics measured in 3 ways (mean, standard error, worst values)
Data Source: Features computed from digitized image of Fine Needle Aspirate (FNA) of breast mass
🔍 Key Findings

Top Predictors: Concave points, radius, perimeter
Weakest Predictors: Fractal dimension, symmetry
Best Model: Logistic Regression with 95.614% accuracy
Clinical Impact: Model gives correct diagnosis in 95 out of 100 cases
Early Detection Benefit: Can increase survival rate from 27% to 99% through early detection
🛠️ Libraries Used

pandas
numpy
matplotlib
seaborn
scikit-learn
tensor-flow
keras
🚀 Methodology

Data Preprocessing:

Converted diagnosis column from categorical (M,B) to numerical values (1,0)
Dropped unnecessary columns (ID and NaN-only columns)
Applied data scaling for model optimization
Train-test split (80% training, 20% testing)
Model Development:

Correlation Analysis: Identified strongest and weakest predictors
Model Training: Implemented Decision Tree and Logistic Regression
Hyperparameter Optimization: Used grid-search technique for parameter tuning
Model Evaluation: Compared performance using accuracy metrics
📈 Results

Model	Accuracy
Decision Tree	94.736%
Logistic Regression	95.614%
🎯 Clinical Significance Our best model achieved 95.6% accuracy, meaning that in a clinical setting, the model provides a correct diagnosis in 95 out of 100 cases. This model could:

Prevent misdiagnosis in thousands of cases every year
Provide faster diagnostic support
Increase average survival rate from 27% to 99% through early detection
🧪 Run the Notebook To run this project locally:

Clone the repo:
git clone https://github.com/varshini-gurushankar/breast-cancer-prediction-team5G.git
cd breast-cancer-prediction-team5G
(Optional) Create a virtual environment and activate it:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:
pip install scikit-learn matplotlib pandas numpy seaborn jupyter
Launch Jupyter:
jupyter notebook
Then open Breast_Cancer_Prediction_Team_5G.ipynb.

💡 Future Improvements

Implement ensemble methods for potentially higher accuracy
Explore deep learning approaches
Add cross-validation for more robust evaluation
Integrate additional medical imaging datasets
