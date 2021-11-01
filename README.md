# ecoli_datamining
This repo contains the result for the datamining final project using the ecoli dataset

## Coding Environment
### Operating System
Windows 10

### Programming Language and Version
Python 3.7.10

### Package Pre-requisites
Numpy
Pandas
Sklearn
Matplotlib
os


## Data Preprocessing
The following data preprocessing steps were taken sequentially, then cross validation is used to compare the best combination of steps to take.
* Random_State = 42
* Remove Nulls
* Normalize Data
* Standardize Data
* Remove Outliers using DBSCAN
* Feature Reduction using PCA

1. Remove Nulls
  1. Remove all nulls using `dropna` with  the function `do_nulls(ecoli_orig,all_x=True)`
  2. Remove nulls on the _Expression Level_ related features with the function `do_nulls(ecoli_orig,all_x=True)`
2. Normalize/Standardize Data
  1. Normalize the data after removing nulls `do_normalize(no_null_ecoli,ecoli_test=None)`
  2. Standardize data after removing nulls `do_standardize(no_null_ecoli,ecoli_test=None)`
3. Remove Outliers
  1. Remove Outliers from Normalized Data `do_dbscan(no_null_norm[0],ecoli_test=None)`
  2. Remove Outliers from Standardized Dta `do_dbscan(no_null_std[0],ecoli_test=None)`
    * However, this will return a null dataset due to DBSCAN only works when there is no negative values in dataset. When Standardize is used, negatives will be retained.
4. Reduce Features
  1. Reduce the number of features from the original dataset that has No Null (step 1).
  2. Reduce the number of features from the No Null (step 1) + Normalized (step 2) dataset.
  3. Reduce the number of features from the No Null (step 1) + Normalized (step 2) + No Outlier (step 3) dataset.
 
## Selection of Data Preprocessing Method
Instead of simply selecting a data preprocessing method using a single model, all models are used in the cross-validation step to compare the best output.
* _Dataset Xi, where: i = [1,2]_
* _Dataset X1_ is cross validated using each _Model k_. Each result is compared using AUC to create a model-to-model comparison for this dataset.
* _Dataset X2_ is cross validated using each _Model k_. Each result is compared using AUC to create a model-to-model comparison for this dataset.
* After the two steps above the best _Model k_ of _Dataset Xi_ is compared using _F1-score_ to ensure that the best performing model is used.

## Model Selection
1. Naive Bayes - this is done for all _preprocessed dataset i_
  1. Gaussian Naive Bayes - do_cv(_preprocessed dataset i_,'gnb',target_names)
  2. Complement Naive Bayes - do_cv(_preprocessed dataset i_,'gnb',target_names)
2. Decision Tree - this is done for all _preprocessed dataset i_
  1. do_cv_2(_preprocessed dataset i_,'clf',target_names)
3. Random Forest (RF)
  1. Standard RF - do_cv_2(_preprocessed dataset i_,'rf_s',target_names,splits=10)
  2. Balanced Class RF - do_cv_2(_preprocessed dataset i_,'rf_cw',target_names,splits=10)
  3. Balanced Subsample Class RF - do_cv_2(_preprocessed dataset i_,'rf_bcw',target_names,splits=10)
4. K-Nearest Neighbor (KNN)
  1. do_cv_2(_preprocessed dataset i_,'knn',target_names,neighbors=neighbors)
    * neighbors = Range ( 1 to 15) to check for best neighbor. Final Neighbor = 5


## Model Optimization
Output from previous will yield that the RF, and KNN methods are still inconclusive which is best. So Optimize both models
* rf_model = RandomForestClassifier()
* knn_model = KNeighborsClassifier(n_neighbors=10)
grid = {'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap}
Do the `parameter_search_CV(grid,rf_model,no_null_norm_db_pca)`

## Final Model
The final model used is the standard RF with the following parameters:
* 'n_estimators': 277
* 'min_samples_split': 20
* 'min_samples_leaf': 8
* 'max_features': 0.25
* 'max_depth': 150
* 'bootstrap': False - *HOWEVER* use True instead to ensure better performance on actual testing

## Results
Cross Validation results on training set
* accuracy: 0.959
* f1-score: 0.806
