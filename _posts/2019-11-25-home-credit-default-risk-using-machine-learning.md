---
title: Home credit default risk using Machine Learning
tags: [Data Science, Machine Learning, Banking]
style: fill
color: light
description: Machine learning model that can predict the probability of the applicant’s capability to repay his loan, using applicants data and records.
---
## Introduction

Loans have made people’s life easier. People often take loans to buy their dream house, dream car, for business and many other reasons. There are lots of key parameters that usually been checked before lending someone a loan because if the deal goes wrong the cost of it will be very high for the lender. So for the lender, it is very important to find out the appropriate applicants who are very likely to repay their loan on time.

Home credit started kaggle challenge so that kagglers can help them ensure that applicants capable of loan repayment are not rejected by using the provided data.

## Problem statement

Build a machine learning model that can predict the probability of the applicant’s capability to repay his loan, given the application data, credit information, previous application data, history of installments payment and some other data.

## Objectives and constraints

* No low-latency requirement.

* Interpretability is partially important.

* Errors can be very costly.

* The probability of a data-point belonging to each class is needed.

## Data Overview

we have 8 data files.

* application_{train|test}.csv
Static data for all applications. One row represents one loan in our data sample.

* bureau.csv
All client’s previous credits provided by other financial institutions that were reported to Credit Bureau (for clients who have a loan in our sample).

* bureau_balance.csv
Monthly balances of previous credits in the Credit Bureau.

* POS_CASH_balance.csv
Monthly balance snapshots of previous POS (point of sales) and cash loans that the applicant had with Home Credit.

* credit_card_balance.csv
Monthly balance snapshots of previous credit cards that the applicant has with Home Credit.

* previous_application.csv
All previous applications for Home Credit loans of clients who have loans in our sample.
There is one row for each previous application related to loans in our data sample.

* installments_payments.csv
Repayment history for the previously disbursed credits in Home Credit related to the loans in our sample.

* HomeCredit_columns_description.csv
This file contains descriptions for the columns in the various data files.

![](https://cdn-images-1.medium.com/max/2402/0*n7pYGwDEibj5mCNS.png)

Mapping real-world problem to machine learning problem

**Type of problem**

Here we need to predict whether the applicant is capable of loan repayment or not. ==> Binary classification problem.

**Performance metric**

This data is highly imbalanced so accuracy can not be used as a performance metric. we will use the below one.

## Exploratory Data Analysis

Let’s perform some EDA on application data. Here we will check how some of the features are distributed in terms of the loan is repaid or not.

Count and percentage of missing values in each columns.

                             Count  Percentage 
    COMMONAREA_MEDI          214865	69.872297 
    COMMONAREA_AVG           214865	69.872297 
    COMMONAREA_MODE          214865	69.872297 
    NONLIVINGAPARTMENTS_MODE 213514	69.432963 
    NONLIVINGAPARTMENTS_MEDI 213514	69.432963 
    NONLIVINGAPARTMENTS_AVG  213514	69.432963 
    FONDKAPREMONT_MODE       210295	68.386172 
    LIVINGAPARTMENTS_MEDI    210199	68.354953 
    LIVINGAPARTMENTS_MODE    210199	68.354953 
    LIVINGAPARTMENTS_AVG     210199	68.354953 
    FLOORSMIN_MEDI           208642	67.848630 
    FLOORSMIN_MODE           208642	67.848630 
    FLOORSMIN_AVG            208642	67.848630 
    YEARS_BUILD_MEDI         204488	66.497784 
    YEARS_BUILD_AVG          204488	66.497784 
    YEARS_BUILD_MODE         204488	66.497784 
    OWN_CAR_AGE              202929	65.990810 
    LANDAREA_MODE            182590	59.376738 
    LANDAREA_AVG             182590	59.376738 
    LANDAREA_MEDI            182590	59.376738

Distribution of the target column

![](https://cdn-images-1.medium.com/max/2000/0*3iyMGCDZgaWKx_yy)

As th above figure shows it is a highly imbalanced data. There are less people who have not repaied their loan.

Type of loans

![](https://cdn-images-1.medium.com/max/2000/0*7naowVVUEkKhyfuG)

This data containes cash loans and revolving loans.

Distribution of the AMT_CREDIT

![](https://cdn-images-1.medium.com/max/2000/0*Zb8WVftdatBGZv_z)

Credit dristibution is very skewed there are more number of people with low credit rate.

Distribution of the NAME_FAMILY_STATUS in terms of loan repaid or not.

![](https://cdn-images-1.medium.com/max/2000/0*f1A9UfXyrHHjcCQL)

Distribution of the NAME_EDUCATION_TYPE in terms of loan repaid or not.

![](https://cdn-images-1.medium.com/max/2000/0*VXiClIa2p1IEZ4EV)

There are more candidates with secondary education who have not repaid their loan.

Distribution of Experience in terms of loan is repaid or not.

![](https://cdn-images-1.medium.com/max/2000/0*HDTdXWgvONxE_6n_)

The less experienced candidate seems to have difficulties to repay their loan.

Distribution of AMT_CREDIT in terms of loan is repaid or not.

![](https://cdn-images-1.medium.com/max/2000/0*NMoPFrqopD42VYfW)

The credit feature does not give much information about the candidate’s loan repayment.

## Data preparation and Feature engineering

Let’s do some feature engineering and preprocessing on all the data files. There are mainly two types of features numeric and categorical.

we will do one-hot encoding for categorical features and numerical aggregation on numerical features.

Add some hand made features by taking ratio, difference and flag of the current features.

As we already seen that there are many columns with high rate of missing values, impute them using the median values.

There are some features with only 2 different values. We performed binary encoding for this features.

There are some important factors which is important for loan approval.

* percentage of employment

* percentage of credit

* income per person

* percentage of annuity to income

* payment rate based on annuity and credit

Some more feature engineering is done by including ratio features. EXT_SOURCE_1, EXT_SOURCE_2 and EXT_SOURCE_3 are most important features, so some new features are generated using them.

Create new feature using the status of the credit(active or closed) and status of the application.

**installments_payments**

There are some important factors to be considered as below:

* wether customer pays the installment on time, before the due date or after the due date

* wether customer pays the full amount of installment or less than the amount of installment

* percentage of payment to the installment amount.

Create ratio and flag features.

Combine all our prepared data into a single dataframe so we can use it letter.

    ### combine all the dataframes 
    df = df.join(bureau_agg, how='left', on='SK_ID_CURR') 
    df = df.join(prev_agg, how='left', on='SK_ID_CURR') 
    df = df.join(pos_aggr, how='left', on='SK_ID_CURR') 
    df = df.join(ins_agg, how='left', on='SK_ID_CURR') 
    df = df.join(cc_agg, how='left', on='SK_ID_CURR') 

    ### save as .csv file 
    df.to_csv('home_features.csv')

## Model tunning and selection

Here we will try Logistic regression, SGD with log loss and LGBM Classifier.

First load data and remove columns containing null value more than 75%.

    ### load feature file
    data = pd.read_csv('home_features.csv') 

    ### create train and test set 
    train = data[data['TARGET'].notnull()] 
    test = data[data['TARGET'].isnull()] 

    ### data and label variable 
    y = train['TARGET'] 
    x = train.drop(columns=['TARGET'])

Now split the data into train-test sets, impute missing values and scale using MinMaxScaler.

    ### train test split 
    x_train,x_test,y_train,y_test = train_test_split(x, y,     test_size=0.2, random_state =10) 

    ### fill missing values 
    imputer = Imputer(strategy = 'median') 
    x_train = imputer.fit_transform(x_train) 
    x_test = imputer.transform(x_test) 

    ### data scaling 
    scaler = MinMaxScaler(feature_range = (0, 1)) 
    x_train_scale = scaler.fit_transform(x_train) 
    x_test_scale = scaler.transform(x_test) 

    ### convert np.ndarray to pd.DataFrame 
    x_train_scale = pd.DataFrame(data=x_train_scale, columns=  x.columns.tolist()) 
    x_test_scale = pd.DataFrame(data=x_test_scale, columns=x.columns.tolist())

**Logistic Regression**

We will tune parameters using GridSearch to find best parameters.

    ### initialize parameters 
    param = {'penalty':['l1','l2'], 
    'C':[0.1,0.001,0.0001,0.00001]} 
    ### find best param using grid search 
    log_reg = LogisticRegression() 
    clf = GridSearchCV(log_reg,param_grid=param,n_jobs=-1,verbose=1) clf.fit(x_train_scale,y_train)

    clf.best_params_

{‘C’: 0.1, ‘penalty’: ‘l2’}

Train model with best parameters.

    ### fit model with best param 
    log_reg = LogisticRegression(C=0.1,penalty='l2') log_reg.fit(x_train_scale,y_train) 

    ### predict on test data 
    y_pred = log_reg.predict_proba(x_test_scale)[:,1] 

    ### calculate auroc score 
    score = roc_auc_score(y_test,y_pred) 

    print(score)

0.773822434554706

**SGD Classifier**

Hyperparameter tunning

    ### initialize parameter 
    param = { 'loss':['hinge','log'], 
    'penalty':['l1','l2','elasticnet'], 
    'alpha':[0.1,0.001,0.0001], 
    'fit_intercept':[True,False], 
    'learning_rate':['optimal','adaptive'] } 

    ### find best param using grid search 
    sgd = SGDClassifier(eta0=0.1) 
    clf = GridSearchCV(sgd,param_grid=param,cv=5,n_jobs=-1,verbose=1) clf.fit(x_train_scale,y_train)

    clf.best_params_

{‘alpha’: 0.0001, 
‘fit_intercept’: False,
 ‘learning_rate’: ‘adaptive’,
 ‘loss’: ‘log’, 
‘penalty’: ‘l2’}

Fit model on best parameters

    ### fit model with best param 
    sgd = SGDClassifier(loss='log', penalty='l2', learning_rate='adaptive', fit_intercept=False, alpha=0.0001, eta0=0.1, verbose=0) 
    sgd.fit(x_train_scale,y_train) 

    ### predict on test data 
    y_pred =sgd.predict_proba(x_test_scale)[:,1] 

    ### calculate auroc score 
    score = roc_auc_score(y_test,y_pred) 
    print('Auroc score: ',score)

Auroc score: 0.7719016641126538

**LGBMClassifier**

Lightgbm is faster than XGBoost so we are using Lightgbm. Here Bayesian Optimization is used for hyperparameter tuning.

    #https://github.com/hyperopt/hyperopt/issues/253#issuecomment-298960310 

    import lightgbm as lgb 
    from hyperopt import STATUS_OK 
    from hyperopt import Trials 
    from hyperopt import tpe 
    from hyperopt import hp 
    from hyperopt.pyll.base import scope 
    from hyperopt.pyll.stochastic import sample 

    N_FOLDS = 5 
    train_set = lgb.Dataset(x_train_scale, y_train) 

    def objective(params, n_folds = N_FOLDS): 
        """Objective function for Gradient Boosting Machine        Hyperparameter Tuning""" 

        ### Perform n_fold cross validation with hyperparameters 
        ### Use early stopping and evalute based on ROC AUC    cv_results = lgb.cv(params, train_set, nfold = n_folds,          num_boost_round = 10, early_stopping_rounds = 100, metrics = 'auc', seed = 50,verbose_eval=True) 

        ### Extract the best score 
        best_score = max(cv_results['auc-mean']) 

        ### Loss must be minimized 
        loss = 1 - best_score 
        
        ### Dictionary with information for evaluation 
        return {'loss': loss, 'params': params, 'status': STATUS_OK}     

    space = { 'num_leaves': sample(scope.int(hp.quniform('num_leaves', 30, 40, 1))), 'max_depth':sample(scope.int(hp.quniform('max_depth', 5,10,1))), 'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.02)), 'subsample': hp.uniform('subsample', 0.7,1.0), 'min_child_weight': hp.uniform('min_child_weight', 20, 50), 'reg_alpha': hp.uniform('reg_alpha', 0.0, 0.1), 'reg_lambda': hp.uniform('reg_lambda', 0.0, 0.1), 'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0), 'is_unbalance':[True] } 

    ### Algorithm 
    tpe_algorithm = tpe.suggest 

    ### Trials object to track progress 
    bayes_trials = Trials() 

    from hyperopt import fmin 

    MAX_EVALS = 10 
    ### Optimize 
    best = fmin(fn = objective, space = space, algo = tpe.suggest, max_evals = MAX_EVALS, trials = bayes_trials)

**Model training**

Model is trained using 5 fold cross validation. Parameter used here are found using above hyperparameter tunning.

    from lightgbm import LGBMClassifier 
    from sklearn.model_selection import KFold, StratifiedKFold

    param = { 'boosting_type': 'goss', 'n_estimators': 10000, 'learning_rate': 0.005134, 'num_leaves': 54, 'max_depth': 10, 'subsample_for_bin': 240000, 'reg_alpha': 0.436193, 'reg_lambda': 0.479169, 'colsample_bytree': 0.508716, 'min_split_gain': 0.024766, 'subsample': 1, 'is_unbalance': False, 'silent':-1, 'verbose':-1 }

    folds = KFold(n_splits= 5, shuffle=True, random_state=1001) sub_preds = np.zeros(test.shape[0]) preds = np.zeros(train.shape[0])

    for n_fold, (train_idx, valid_idx) in enumerate(  
                                       folds.split(train[predictors],
                                       train['TARGET'])):
         train_x,train_y = train[predictors].iloc[train_idx],    
                          train['TARGET'].iloc[train_idx]   
         valid_x,valid_y = train[predictors].iloc[valid_idx],     
                           train['TARGET'].iloc[valid_idx] 
         clf = LGBMClassifier(**param) 
         clf.fit(train_x, train_y, eval_set=[(train_x, train_y), 
                 (valid_x, valid_y)], eval_metric= 'auc', verbose= 400,   
                  early_stopping_rounds= 100) 
         preds[valid_idx] = clf.predict_proba(valid_x,    
                           num_iteration=clf.best_iteration_)[:, 1] 
         sub_preds += clf.predict_proba(test[predictors],   
         num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits  
         print('Fold %2d AUC : %.6f' % (n_fold + 1,  
                roc_auc_score(valid_y, preds[valid_idx]))) 
        del clf, train_x, train_y, valid_x, valid_y 
        gc.collect() 
    print('Full AUC score %.6f' % roc_auc_score(train['TARGET'], preds))

    Training until validation scores don't improve for 100 rounds. [400]	training's binary_logloss: 0.236965	training's auc: 0.801024	valid_1's binary_logloss: 0.245488	valid_1's auc: 0.772997 [800]	training's binary_logloss: 0.224045	training's auc: 0.82938	valid_1's binary_logloss: 0.239297	valid_1's auc: 0.785794 [1200]	training's binary_logloss: 0.215228	training's auc: 0.849587	valid_1's binary_logloss: 0.236786	valid_1's auc: 0.791832 [1600]	training's binary_logloss: 0.207994	training's auc: 0.866215	valid_1's binary_logloss: 0.235517	valid_1's auc: 0.795027 [2000]	training's binary_logloss: 0.201609	training's auc: 0.880665	valid_1's binary_logloss: 0.234799	valid_1's auc: 0.796835 [2400]	training's binary_logloss: 0.195764	training's auc: 0.893626	valid_1's binary_logloss: 0.234361	valid_1's auc: 0.797947 [2800]	training's binary_logloss: 0.190345	training's auc: 0.905009	valid_1's binary_logloss: 0.234069	valid_1's auc: 0.798742 [3200]	training's binary_logloss: 0.185295	training's auc: 0.915056	valid_1's binary_logloss: 0.233851	valid_1's auc: 0.799285 [3600]	training's binary_logloss: 0.180488	training's auc: 0.924116	valid_1's binary_logloss: 0.233658	valid_1's auc: 0.799773 Early stopping, best iteration is: [3684]	training's binary_logloss: 0.17951	training's auc: 0.925918	valid_1's binary_logloss: 0.233628	valid_1's auc: 0.799816 Fold 1 AUC : 0.799817 Training until validation scores don't improve for 100 rounds. [400]	training's binary_logloss: 0.237579	training's auc: 0.800652	valid_1's binary_logloss: 0.243371	valid_1's auc: 0.771569 [800]	training's binary_logloss: 0.224608	training's auc: 0.829414	valid_1's binary_logloss: 0.237195	valid_1's auc: 0.785063 [1200]	training's binary_logloss: 0.215812	training's auc: 0.849501	valid_1's binary_logloss: 0.234821	valid_1's auc: 0.790467 [1600]	training's binary_logloss: 0.208519	training's auc: 0.866436	valid_1's binary_logloss: 0.233634	valid_1's auc: 0.793247 [2000]	training's binary_logloss: 0.202071	training's auc: 0.881007	valid_1's binary_logloss: 0.232967	valid_1's auc: 0.79482 [2400]	training's binary_logloss: 0.196181	training's auc: 0.893908	valid_1's binary_logloss: 0.23246	valid_1's auc: 0.796098 [2800]	training's binary_logloss: 0.190769	training's auc: 0.905292	valid_1's binary_logloss: 0.232198	valid_1's auc: 0.796701 [3200]	training's binary_logloss: 0.185677	training's auc: 0.915467	valid_1's binary_logloss: 0.231982	valid_1's auc: 0.797248 [3600]	training's binary_logloss: 0.180872	training's auc: 0.92442	valid_1's binary_logloss: 0.231823	valid_1's auc: 0.797563 [4000]	training's binary_logloss: 0.176339	training's auc: 0.932334	valid_1's binary_logloss: 0.231743	valid_1's auc: 0.797742 Early stopping, best iteration is: [4071]	training's binary_logloss: 0.175562	training's auc: 0.933621	valid_1's binary_logloss: 0.231723	valid_1's auc: 0.797798 Fold 2 AUC : 0.797791 Training until validation scores don't improve for 100 rounds. [400]	training's binary_logloss: 0.237185	training's auc: 0.801558	valid_1's binary_logloss: 0.244709	valid_1's auc: 0.765668 [800]	training's binary_logloss: 0.224224	training's auc: 0.830057	valid_1's binary_logloss: 0.238839	valid_1's auc: 0.779433 [1200]	training's binary_logloss: 0.215497	training's auc: 0.849893	valid_1's binary_logloss: 0.236385	valid_1's auc: 0.785821 [1600]	training's binary_logloss: 0.208227	training's auc: 0.866754	valid_1's binary_logloss: 0.235072	valid_1's auc: 0.789404 [2000]	training's binary_logloss: 0.201791	training's auc: 0.881266	valid_1's binary_logloss: 0.234317	valid_1's auc: 0.791485 [2400]	training's binary_logloss: 0.195924	training's auc: 0.894101	valid_1's binary_logloss: 0.233881	valid_1's auc: 0.792763 [2800]	training's binary_logloss: 0.190512	training's auc: 0.90538	valid_1's binary_logloss: 0.233576	valid_1's auc: 0.79364 [3200]	training's binary_logloss: 0.185448	training's auc: 0.915421	valid_1's binary_logloss: 0.233366	valid_1's auc: 0.794198 [3600]	training's binary_logloss: 0.180668	training's auc: 0.924357	valid_1's binary_logloss: 0.23318	valid_1's auc: 0.79473 [4000]	training's binary_logloss: 0.176147	training's auc: 0.932237	valid_1's binary_logloss: 0.233055	valid_1's auc: 0.795059 [4400]	training's binary_logloss: 0.171833	training's auc: 0.939206	valid_1's binary_logloss: 0.232997	valid_1's auc: 0.795224 Early stopping, best iteration is: [4681]	training's binary_logloss: 0.168924	training's auc: 0.94377	valid_1's binary_logloss: 0.232941	valid_1's auc: 0.795402 Fold 3 AUC : 0.795401 Training until validation scores don't improve for 100 rounds. [400]	training's binary_logloss: 0.236536	training's auc: 0.801444	valid_1's binary_logloss: 0.247001	valid_1's auc: 0.768283 [800]	training's binary_logloss: 0.223619	training's auc: 0.830026	valid_1's binary_logloss: 0.240915	valid_1's auc: 0.781945 [1200]	training's binary_logloss: 0.214892	training's auc: 0.849725	valid_1's binary_logloss: 0.238473	valid_1's auc: 0.78789 [1600]	training's binary_logloss: 0.207625	training's auc: 0.866576	valid_1's binary_logloss: 0.237206	valid_1's auc: 0.791342 [2000]	training's binary_logloss: 0.201187	training's auc: 0.881319	valid_1's binary_logloss: 0.236454	valid_1's auc: 0.793455 [2400]	training's binary_logloss: 0.195319	training's auc: 0.894289	valid_1's binary_logloss: 0.236075	valid_1's auc: 0.794521 [2800]	training's binary_logloss: 0.189906	training's auc: 0.905677	valid_1's binary_logloss: 0.235784	valid_1's auc: 0.795415 [3200]	training's binary_logloss: 0.184872	training's auc: 0.915714	valid_1's binary_logloss: 0.23557	valid_1's auc: 0.796012 [3600]	training's binary_logloss: 0.180076	training's auc: 0.924677	valid_1's binary_logloss: 0.235437	valid_1's auc: 0.796402 [4000]	training's binary_logloss: 0.175563	training's auc: 0.932673	valid_1's binary_logloss: 0.235362	valid_1's auc: 0.79671 [4400]	training's binary_logloss: 0.171252	training's auc: 0.939824	valid_1's binary_logloss: 0.235277	valid_1's auc: 0.796962 Early stopping, best iteration is: [4418]	training's binary_logloss: 0.171072	training's auc: 0.940115	valid_1's binary_logloss: 0.235267	valid_1's auc: 0.796995 Fold 4 AUC : 0.796994 Training until validation scores don't improve for 100 rounds. [400]	training's binary_logloss: 0.236747	training's auc: 0.801775	valid_1's binary_logloss: 0.246415	valid_1's auc: 0.767425 [800]	training's binary_logloss: 0.223775	training's auc: 0.830154	valid_1's binary_logloss: 0.240443	valid_1's auc: 0.780831 [1200]	training's binary_logloss: 0.215008	training's auc: 0.850114	valid_1's binary_logloss: 0.237981	valid_1's auc: 0.786814 [1600]	training's binary_logloss: 0.207785	training's auc: 0.866723	valid_1's binary_logloss: 0.236822	valid_1's auc: 0.789616 [2000]	training's binary_logloss: 0.20133	training's auc: 0.881455	valid_1's binary_logloss: 0.236146	valid_1's auc: 0.791326 [2400]	training's binary_logloss: 0.195434	training's auc: 0.894419	valid_1's binary_logloss: 0.235709	valid_1's auc: 0.792425 [2800]	training's binary_logloss: 0.190018	training's auc: 0.905791	valid_1's binary_logloss: 0.235396	valid_1's auc: 0.793228 [3200]	training's binary_logloss: 0.184966	training's auc: 0.915785	valid_1's binary_logloss: 0.235202	valid_1's auc: 0.793738 [3600]	training's binary_logloss: 0.180179	training's auc: 0.924728	valid_1's binary_logloss: 0.235077	valid_1's auc: 0.79408 [4000]	training's binary_logloss: 0.175631	training's auc: 0.932736	valid_1's binary_logloss: 0.234991	valid_1's auc: 0.794287 Early stopping, best iteration is: [4006]	training's binary_logloss: 0.175561	training's auc: 0.932861	valid_1's binary_logloss: 0.234987	valid_1's auc: 0.794297 Fold 5 AUC : 0.794295 Full AUC score 0.796840

Here highest score is in fold 1 with 0.799817 score.

The overall score is 0.796840

Conclusion

Compare all the model’s score using prettytable.

    from prettytable import PrettyTable 
    pt=PrettyTable() 
    pt.field_names=["model","test_auc"] 
    pt.add_row(["Logistic Regression","0.7738"]) 
    pt.add_row(["SGD with log loss","0.7719"]) pt.add_row(["LGBMClassifier","0.7968"])

    print(pt)

    +---------------------------+----------+ 
    |         model             | test_auc | 
    +---------------------------+----------+
    | Logistic Regression       | 0.7738   | 
    | SGD with log loss         | 0.7719   | 
    | LGBMClassifier            | 0.7968   | 
    +---------------------------+----------+

LGBMClassier gives the best performance amongst other models and it is also faster than XGBoost.

**References:**

*Originally published at [http://csvankhede.wordpress.com](https://csvankhede.wordpress.com/2019/11/25/home-credit-default-risk-using-machine-learning/) on November 25, 2019.*
