# 🏠 House Price Prediction using Regression Models

## 📌 Project Overview

This project is part of **Task 1 of the Prodigy InfoTech Machine
Learning Internship**.\
The objective is to implement and evaluate regression models to predict
house prices based on property features such as square footage, number
of bedrooms, bathrooms, and other housing attributes.

The project uses the **House Prices: Advanced Regression Techniques**
dataset from Kaggle and applies multiple regression techniques to
achieve accurate price prediction.

Dataset Link:\
https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

------------------------------------------------------------------------

## ✨ Features

-   Data preprocessing and handling missing values\
-   Feature encoding and normalization\
-   Implementation of multiple regression models:
    -   Ridge Regression\
    -   ElasticNet Regression\
    -   Gradient Boosting Regressor (GBR)\
-   Model evaluation using Root Mean Squared Error (RMSE)\
-   Performance comparison across models\
-   Clean and reproducible pipeline

------------------------------------------------------------------------

## 🚀 Quick Start

### 1️⃣ Clone the Repository

    git clone https://github.com/your-username/house-price-prediction.git
    cd house-price-prediction

### 2️⃣ Install Dependencies

    pip install -r requirements.txt

### 3️⃣ Download Dataset

Download the dataset from Kaggle and place: - `train.csv` - `test.csv`

inside the `data/` directory.

### 4️⃣ Run the Notebook

    jupyter notebook Task1.ipynb

Or run the training script:

    python src/train.py

------------------------------------------------------------------------

## 📂 Project Structure

    house-price-prediction/
    │
    ├── data/
    │   ├── train.csv
    │   └── test.csv
    │
    ├── notebooks/
    │   └── Task1.ipynb
    │
    ├── README.md
    └── requirements.txt

------------------------------------------------------------------------

## 🛠️ Technologies Used

-   Python 3.x\
-   Pandas & NumPy\
-   Scikit-learn\
-   Matplotlib & Seaborn\
-   Jupyter Notebook

------------------------------------------------------------------------

## 📊 Model Performance

  Model                         RMSE
  ----------------------------- ------------
  Ridge Regression              0.1145
  ElasticNet Regression         **0.1130**
  Gradient Boosting Regressor   0.1216

### 🔎 Performance Analysis

-   **ElasticNet Regression** achieved the best performance by combining
    L1 and L2 regularization.\
-   **Ridge Regression** showed strong generalization capability.\
-   **Gradient Boosting Regressor** performed slightly lower due to
    limited hyperparameter tuning.\
-   Regularized linear models performed best after preprocessing and
    feature engineering.

------------------------------------------------------------------------

## 🧠 About Prodigy InfoTech Internship

Prodigy InfoTech provides hands-on internships focused on real-world
applications of Machine Learning, Data Science, and Artificial
Intelligence.\
This task strengthens practical skills in:

-   Data preprocessing\
-   Model building\
-   Performance evaluation\
-   Comparative analysis

------------------------------------------------------------------------

## 📜 License

This project is licensed under the MIT License.\
You are free to use, modify, and distribute this project with proper
attribution.

------------------------------------------------------------------------

## 🙏 Acknowledgments

-   Kaggle for providing the House Prices dataset\
-   Scikit-learn development team\
-   Prodigy InfoTech for the internship opportunity\
-   Open-source community for continuous learning resources

------------------------------------------------------------------------

⭐ If you found this project helpful, consider giving the repository a
star!
