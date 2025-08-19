# Machine Learning Tasks

This repository contains a collection of machine learning tasks (from regression and clustering to recommendation systems, time-series forecasting, and deep learning).  
Each task is implemented in Python as an independent script.  

## Important Note about Datasets
Datasets are **not included** in this repository due to large file sizes.  
- You must manually download them and place them in a `Datasets/` folder.  
- Or use the Kaggle API to fetch them instead of the _read_csv_ function.
- Update the file paths inside each script to point to your dataset location.

### Using Kaggle APIs
```bash
kaggle datasets download -d <dataset-identifier>
```

## Setup

1. Clone this repository:
```bash
 git clone https://github.com/FarahAhmedd/Elevvo_Pathways_ML_Internship
```
```bash
 cd Elevvo_Pathways_ML_Internship
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download datasets and place them in a Datasets folder or add the Kaggle API.

## Datasets

- Task 1: [Student Performance Factors](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)  
- Task 2: [Mall Customer Dataset](https://www.kaggle.com/code/yousefmohamed20/mall-customer-segmentation-using-kmeans)  
- Task 3: [Forest Cover Type](https://archive.ics.uci.edu/dataset/31/covertype)  
- Task 4: [Loan Approval Prediction](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)  
- Task 5: [MovieLens 100K](https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset)  
- Task 6: [GTZAN Music Genre](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)  
- Task 7: [Walmart Sales Forecasting](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast)  
- Task 8: [GTSRB Traffic Sign Recognition](https://www.kaggle.com/code/alymadian/gtsrb-traffic-signs-image-classification-95-cnn)  

## Tasks Overview
### Task 1
- Description: Predict students' exam scores from study hours.
- Bonus: Polynomial regression & experimenting with different features.
- Expected Output: Regression line plots, model evaluation metrics (MSE, R²).
- Files: task1WithBonus.py.

### Task 2
- Description: Cluster mall customers by income and spending.
- Bonus: Use DBSCAN, analyze spending per cluster.
- Expected Output: Cluster visualization plots, customer segment analysis.
- Files: task2WithBonus.py.

### Task 3
- Description: Multi-class classification of forest cover types.
- Bonus: Compare Random Forest vs. XGBoost, hyperparameter tuning.
- Expected Output: Confusion matrix, feature importance plots, accuracy scores.
- Files: task3WithBonus.py.

### Task 4
- Description: Predict loan approval using imbalanced classification.
- Bonus: Use SMOTE, compare logistic regression vs. decision tree.
- Expected Output: Precision, recall, F1-score, confusion matrix.
- Files: task4.py, task4WithBonus.py.

### Task 5
- Description: Build a recommender system using user similarity.
- Bonus: Implement item-based filtering & matrix factorization (SVD).
- Expected Output: Top-k recommended movies, Precision@K.
- Files: task5WithBonus.py.

### Task 6
- Description: Classify songs into genres using audio features.
- Bonus: Compare tabular (MFCCs) vs. spectrogram CNN (transfer learning).
- Expected Output: Accuracy score, confusion matrix, CNN performance.
- Files: task6WithBonus.py.

### Task 7
- Description: Forecast Walmart sales from historical data.
- Bonus: Use rolling averages, seasonal decomposition, XGBoost/LightGBM.
- Expected Output: Forecast plots, actual vs. predicted values, error metrics.
- Files: task7WithBonus.py.

### Task 8
- Description: Classify traffic signs using deep learning (CNN).
- Bonus: Compare custom CNN vs. MobileNet, add data augmentation.
- Expected Output: Model accuracy, confusion matrix, sample predictions.
- Files: task8WithBonus.py.

## Usage
### Run each task independently
```bash
python task1WithBonus.py
```

```bash
python task2WithBonus.py
```

```bash
python task3WithBonus.py
```

```bash
python task4.py
```

```bash
python task4WithBonus.py
```

```bash
python task5WithBonus.py
```

```bash
python task6WithBonus.py
```

```bash
python task7WithBonus.py
```

```bash
python task8WithBonus.py
```
