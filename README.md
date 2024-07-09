### Cardiovascular Disease Prediction
This project involves predicting cardiovascular disease using various machine learning models. The goal is to compare the performance of different models and identify the most accurate one. The models used in this project include Logistic Regression, Support Vector Machines, k-Nearest Neighbors, Naive Bayes, Decision Trees, Random Forests, XGBoost, LightGBM, Gradient Boosting, Ridge Classifier, and Bagging Classifier.

## Project Overview
Cardiovascular diseases (CVDs) are the leading cause of death globally, making it crucial to develop reliable prediction models to identify at-risk individuals. This project leverages multiple machine learning algorithms to predict the likelihood of cardiovascular disease based on a given set of features.

## Models Implemented
- Logistic Regression
- Support Vector Machines (SVM)
- Linear SVC
- k-Nearest Neighbors (k-NN) with GridSearchCV
- Naive Bayes
- Perceptron
- Stochastic Gradient Descent (SGD)
- Decision Tree Classifier
- Random Forests with GridSearchCV
- XGB Classifier with HyperOpt
- LGBM Classifier with HyperOpt
- GradientBoostingClassifier with HyperOpt
- RidgeClassifier
- BaggingClassifier
  
## Methodology
- Data Preprocessing: The dataset is cleaned and preprocessed to handle missing values, normalize features, and encode categorical variables.
Model Training: Each model is trained on the preprocessed dataset. Hyperparameter tuning is performed for certain models using GridSearchCV and HyperOpt.
Model Evaluation: The models are evaluated based on accuracy, precision, recall, F1-score, and ROC-AUC.
- Comparison and Visualization: The performance of all models is compared using various metrics, and the results are visualized using graphs.
-  Results: The performance of each model is compared, and the Random Forest model with GridSearchCV is found to be the most accurate with an accuracy of 99.7%. The performance metrics for all models are visualized in the graphs below.


Dependencies
- Python 3.8+
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- HyperOpt
- Matplotlib
- Seaborn

## Conclusion
This project demonstrates the application of various machine learning models for predicting cardiovascular disease. The Random Forest model with GridSearchCV outperformed the other models in terms of accuracy. Future work can involve exploring more advanced techniques and larger datasets to further improve the prediction performance.



## Acknowledgements
The dataset used in this project was obtained from kaggle
Special thanks to the open-source community for providing valuable resources and libraries.
