# 🌍 Geospatial Insights and Comprehensive EDA

This project performs an in-depth **Exploratory Data Analysis (EDA)** and geospatial analysis on an anonymized country-based dataset. The goal is to uncover key patterns, trends, and relationships in the data, followed by predictive modeling.

## 📌 Project Highlights
- **Dataset Overview:**  
  The dataset includes numerical, categorical, and date-time features, along with a binary target variable.
- **Data Preprocessing:**  
  - Handling missing values  
  - Encoding categorical variables  
  - Scaling numerical features  
- **Exploratory Data Analysis (EDA):**  
  - Distribution analysis and visualizations  
  - Correlation heatmaps  
  - Feature importance analysis  
- **Geospatial Analysis:**  
  - Mapping country-level insights  
  - Visualizing geographic trends  
- **Machine Learning Models:**  
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
  - Support Vector Machine (SVM)  
  - Gaussian Naive Bayes  
  - K-Nearest Neighbors (KNN)  
  - Gradient Boosting  
  - Multi-layer Perceptron (MLP Neural Network)  

## 📊 Model Performance
| Model                         | Accuracy | Precision | Recall | F1-Score |
|--------------------------------|----------|------------|--------|----------|
| **MLPClassifier**              | 0.978    | 0.963      | 0.978  | 0.965    |
| **Logistic Regression**        | 0.977    | 0.983      | 0.969  | 0.976    |
| **GaussianNB**                 | 0.976    | 0.986      | 0.964  | 0.975    |
| **RandomForestClassifier**     | 0.974    | 0.978      | 0.969  | 0.973    |
| **GradientBoostingClassifier** | 0.974    | 0.983      | 0.964  | 0.973    |
| **KNeighborsClassifier**       | 0.972    | 0.981      | 0.962  | 0.971    |
| **SVC**                        | 0.969    | 0.994      | 0.942  | 0.968    |
| **DecisionTreeClassifier**     | 0.959    | 0.952      | 0.967  | 0.959    |

## 📂 Dataset
This project uses an **anonymized dataset** containing various country-based metrics. The dataset is preprocessed to handle missing values, encode categorical variables, and scale numerical data.

## 🛠️ Installation
To run this project, install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn plotly kaleido graphviz pycountry openpyxl
