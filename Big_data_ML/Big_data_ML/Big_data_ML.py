import sys, gc
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_selection import RFECV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, normalize
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.externals import joblib

#from pyspark.sql import SparkSession
#from pyspark.sql.functions import isnan, when, count, col
#from pyspark.ml.feature import StandardScaler, VectorAssembler
#from pyspark.ml.clustering import KMeans


from data_types import *

from knn_impute import knn_impute


def read_csv(spark,filePath):
    df = spark.read.csv(filePath, inferSchema= True, header= True)
    df.printSchema()
    return df

def DrawROC(results):
    ## prepare score-label set
    results_collect = results.collect()
    results_list = [(float(i[0][0]), 1.0-float(i[1])) for i in results_collect]
    y_test = [i[1] for i in results_list]
    y_score = [i[0] for i in results_list]

    total_data = len(y_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_score).ravel()
    false_positives = fp / total_data
    true_positives  = tp / total_data
    false_negatives = fn / total_data
    true_negatives  = tn / total_data

    print("\t\tTrue Positive\tFalse Positive")
    print("\t\t{}\t{}".format(true_positives,false_positives))
    print("\t\tFalse Negatives\tTrue Negatives")
    print("\t\t{}\t{}".format(false_negatives,true_negatives))
    print(classification_report(y_test, y_score))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
 
    y_test = [i[1] for i in results_list]
    y_score = [i[0] for i in results_list]
 
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
 
    
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve  ')
    plt.legend(loc="lower right")
    plt.show()

def free_memory():
    c = gc.collect()
    print('Free {} memory allocation..'.format(c))

def run_spark():
    spark = SparkSession.builder.\
                config("spark.executor.memory", "4g")\
                .master('local[*]') \
                .config("spark.driver.memory","18g")\
                .config("spark.executor.cores","4")\
                .config("spark.python.worker.memory","4g")\
                .config("spark.driver.maxResultSize","0")\
                .config("spark.default.parallelism","2")\
                .appName('BD_ML_project').getOrCreate()
    print('Load data ...')
    df_train = read_csv('./train.csv')
    print('Data loaded ...')
    print('Processing null data ...')
    df_train.select([count(when(isnan(c), c)).alias(c) for c in df_train.columns]).show()

    spark.stop()


def run_data_analysis():
    train = pd.read_csv("./train.csv", 
                        dtype=dtypes,
                        usecols=dtypes.keys())
    #missing_val_count_by_column = (train.isnull().sum())
    #plot = missing_val_count_by_column.plot(kind="bar", figsize=(20,10),title ='Ratio of Null values by features')
    #plt.grid = True
    #plt.show()
    #fig = plot.get_figure()
    #fig.savefig("null values.png")

    #SIZE = 8921483
    #cols_null = missing_val_count_by_column.where(lambda x: (SIZE/2) <= x).dropna()
    #plot = cols_null.plot(kind="bar", figsize=(20,10),title ='Ratio of Null values of features greater than 50% of datasize')
    #plt.grid = True
    #plt.show()
    #fig = plot.get_figure()
    #fig.savefig("null values more than 50 pre.png")
    #print(cols_null)
    train = train.drop(cols_with_missing_more_50 , axis=1)
    free_memory()

    cols = train.columns.tolist()
    plt.figure(figsize=(10,10))
    co_cols = cols[:10]
    co_cols.append('HasDetections')
    sns.heatmap(train[co_cols].corr(), cmap='RdBu_r', annot=True, center=0.0, fmt= '.3f')
    plt.title('Correlation between 1 ~ 10th columns')
    plt.show()

    co_cols = cols[10:20]
    co_cols.append('HasDetections')
    plt.figure(figsize=(10,10))
    sns.heatmap(train[co_cols].corr(), cmap='RdBu_r', annot=True, center=0.0, fmt= '.3f')
    plt.title('Correlation between 11 ~ 20th columns')
    plt.show()


    co_cols = cols[20:30]
    co_cols.append('HasDetections')
    plt.figure(figsize=(10,10))
    sns.heatmap(train[co_cols].corr(), cmap='RdBu_r', annot=True, center=0.0, fmt= '.3f')
    plt.title('Correlation between 21 ~ 30th columns')
    plt.show()

    co_cols = cols[30:40]
    co_cols.append('HasDetections')
    plt.figure(figsize=(10,10))
    sns.heatmap(train[co_cols].corr(), cmap='RdBu_r', annot=True, center=0.0, fmt= '.3f')
    plt.title('Correlation between 31 ~ 40th columns')
    plt.show()


    co_cols = cols[40:50]
    co_cols.append('HasDetections')
    plt.figure(figsize=(10,10))
    sns.heatmap(train[co_cols].corr(), cmap='RdBu_r', annot=True, center=0.0, fmt= '.3f')
    plt.title('Correlation between 41 ~ 50th columns')
    plt.show()

    co_cols = cols[50:60]
    co_cols.append('HasDetections')
    plt.figure(figsize=(10,10))
    sns.heatmap(train[co_cols].corr(), cmap='RdBu_r', annot=True, center=0.0, fmt= '.3f')
    plt.title('Correlation between 51 ~ 60th columns')
    plt.show()

    co_cols = cols[60:]
    #co_cols.append('HasDetections')
    plt.figure(figsize=(10,10))
    sns.heatmap(train[co_cols].corr(), cmap='RdBu_r', annot=True, center=0.0, fmt= '.3f')
    plt.title('Correlation between from 61th to the last columns')
    plt.show()

def exrport_model(model, filename = 'finalized_model.sav'):
    # save the model to disk
    joblib.dump(model, filename)

def time_ser_ML():
    #Time searsie
    import statsmodels.api as sm

    logreg = sm.Logit(X_train, y_train)
    logreg = logreg.fit(disp=0)
    X_train['Prob'] = logr.predict(X_train[cols])
    print("Train report...")
    print(classification_report(y_train, X_train['Prob'] ))
    #print(report)
    print('Exporting model ....')
    exrport_model(logreg,'logreg.sav')
    
    print('Predicting data ....')
    y_pred = rfe.predict(X_test)


def train_model(Sample_size, features, label, ALL_DATA = False):
    if ALL_DATA:
        train = pd.read_csv("./train.csv", 
                            dtype= col_to_load,
                            usecols= col_to_load.keys())
    else:
        train = pd.read_csv("./train.csv", 
                            dtype= col_to_load,
                            usecols= col_to_load.keys(),
                            nrows= Sample_size)
    #vv = ['AVProductStatesIdentifier','AVProductsInstalled','OsBuild',
    #   'CountryIdentifier','Processor','SmartScreen','Census_OSVersion',
    #   'Census_ProcessorCoreCount','Census_ProcessorModelIdentifier', 
    #   'Census_HasOpticalDiskDrive','Census_TotalPhysicalRAM',
    #   'Census_InternalPrimaryDisplayResolutionVertical','Census_OSInstallTypeName',
    #   'Census_PowerPlatformRoleName','Census_IsTouchEnabled',
    #   'Wdft_IsGamer']
    #features =vv
    le = LabelEncoder()
    # Convert categorty type to int
    for cols in train:
        if train[cols].dtype.name == 'category':
            train[cols] = le.fit_transform(train[cols].astype(str))
        free_memory()
    train = train.fillna(train.mean())
    train = reduce_mem_usage(train)
    free_memory()
    X_train, X_test, y_train, y_test = train_test_split(train[features], train[label], test_size=0.3, random_state=2018)

    ## Standarize features
    scaler = StandardScaler(with_mean=True)
    X_std = scaler.fit_transform(X_train)
    Y_std = scaler.fit_transform(X_test)
    

    logreg = LogisticRegression(C=0.1 ,penalty='l1', random_state= 2018, solver='liblinear',max_iter=100,verbose=True)
    rfe = RFECV(estimator=logreg, cv=4, scoring='roc_auc',n_jobs = -1)

    ##results  = cross_val_score(logreg, X_train, y_train, scoring='roc_auc', cv=3, n_jobs = -1)  
    print('Training model ....')
    X_new = rfe.fit_transform(X_std, y_train)
    
    #features_index = rfe.get_support(indices=True)
    
    #print(report)
    print('Exporting model ....')
    exrport_model(logreg,'logreg.sav')
    
    print('Predicting data ....')
    #y_pred = rfe.predict(X_test)

    Y_new = rfe.transform(Y_std)
    rfe.fit(X_new,y_train)
    y_pred = rfe.predict(Y_new)
    #y_pred = cross_val_predict(logreg, X_test, y_test, scoring='roc_auc', cv=3, n_jobs = -1)
    #print('\ntest results = {}\n'.format(y_pred))
    #print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(np.mean(results)))
    print("Test report...")
    confusionmatrix = confusion_matrix(y_test, y_pred)
    print(confusionmatrix)
    #beta = np.sort(logreg.coef_)
    #plt.plot(beta)
    #plt.ylabel('Beta Coefficients')
    #plt.show()

    print(classification_report(y_test, y_pred))
    proba = rfe.predict_log_proba(Y_new)
    #proba = cross_val_predict(logreg, X_test, y_test, scoring='roc_auc', cv=3, method='predict_proba', n_jobs = -1)
    logit_roc_auc = roc_auc_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, proba[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROCofLogReg')
    plt.show()


    # Plot number of features vs CV scores
    plt.figure()
    plt.xlabel('k')
    plt.ylabel('CV accuracy')
    plt.plot(np.arange(1, rfe.grid_scores_.size+1), rfe.grid_scores_)
    plt.show()

def train_modelRF(Sample_size, features, label, ALL_DATA = False):
    if ALL_DATA:
        train = pd.read_csv("./train.csv", 
                            dtype= col_to_load,
                            usecols= col_to_load.keys())
    else:
        train = pd.read_csv("./train.csv", 
                            dtype= col_to_load,
                            usecols= col_to_load.keys(),
                            nrows= Sample_size)
    le = LabelEncoder()
    # Convert categorty type to int
    for cols in train:
        if train[cols].dtype.name == 'category':
            train[cols] = le.fit_transform(train[cols].astype(str))
        free_memory()
    train = train.fillna(-1)
    train = reduce_mem_usage(train)
    free_memory()
    
    X_train, X_test, y_train, y_test = train_test_split(train[features], train[label], test_size=0.33, random_state=2018)

    # Standardizing the features
    X_train = StandardScaler(with_mean=True).fit_transform(X_train)
    X_test = StandardScaler(with_mean=True).fit_transform(X_test)

    X_train = normalize(X_train, norm='l1')
    X_test = normalize(X_test, norm='l1')

    cv = 10
    rf = RandomForestClassifier(random_state=2018, n_jobs = -1, verbose=True, criterion='entropy', max_depth=25, min_samples_split=10,n_estimators=1001)
    #rfe = RFECV(estimator=rf, cv=4, scoring='roc_auc',n_jobs = -1)
    score  = cross_val_score(rf, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs = -1)  
    print("Training score = {}".format(str(score.mean())))


    #y_pred = cross_val_predict(rf, X_test, y_test, cv=cv)
    ##print("Testing score = {}".format(str(prediction_cv.mean())))
    #print("Test report...")
    #confusionmatrix = confusion_matrix(y_test, y_pred)
    #print(confusionmatrix)
    ##beta = np.sort(logreg.coef_)
    ##plt.plot(beta)
    ##plt.ylabel('Beta Coefficients')
    ##plt.show()

    #print(classification_report(y_test, y_pred))
    ##proba =cross_val_predict(rf, X_test, y_test, cv=cv,method='predict_proba')
    ##proba = cross_val_predict(logreg, X_test, y_test, scoring='roc_auc', cv=3, method='predict_proba', n_jobs = -1)
    ##logit_roc_auc = roc_auc_score(y_test, y_pred)
    ##fpr, tpr, thresholds = roc_curve(y_test, proba[:,1])
    ##plt.figure()
    ##plt.plot(fpr, tpr, label='Random Forest (area = %0.2f)' % logit_roc_auc)
    ##plt.plot([0, 1], [0, 1],'r--')
    ##plt.xlim([0.0, 1.0])
    ##plt.ylim([0.0, 1.05])
    ##plt.xlabel('False Positive Rate')
    ##plt.ylabel('True Positive Rate')
    ##plt.title('Receiver operating characteristic')
    ##plt.legend(loc="lower right")
    ##plt.savefig('RandomForest_ROC')
    #plt.show()

    #print('Training model ....')
    #rf.fit(X_train, y_train)
    
    #features_index = rfe.get_support(indices=True)
    
    #print(report)
    #print('Exporting model ....')
    #exrport_model(rf,'RF.sav')
    
    print('Predicting data ....')
    #y_pred = rfe.predict(X_test)

    #X_test = rfe.transform(X_test)
    #rfe.fit(X_train, y_train)
    y_pred = cross_val_predict(rf, X_test, y_test,cv=cv)
   
    print("Test report...")
    confusionmatrix = confusion_matrix(y_test, y_pred)
    print(confusionmatrix)
   

    print(classification_report(y_test, y_pred))
    proba = cross_val_predict(rf, X_test, y_test, cv=cv, method='predict_proba', n_jobs = -1)

    rf_roc_auc = roc_auc_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, proba[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='Random Forest (area = %0.2f)' % rf_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('RandomForest_ROC')
    plt.show()


def fill_missing_data_Knn(Sample_size = 1000):

    all_features = list(col_to_load.keys())[:68]
    cols_with_missing_to_process
    labels = [r for r in cols_with_missing_to_process if r  in all_features]
    features = [r for r in all_features if r not in cols_with_missing_to_process]

    train = pd.read_csv("./train.csv", 
                        dtype= col_to_load,
                        usecols= col_to_load.keys(),
                        nrows= Sample_size
                        )
    #df_missing = train[labels].copy()
    #print("Missing data frame\n")
    print(train.info())
    #print("\n-----------------------\nNon missing data\n")
    
    #df = train[features].copy()
    #print(df.info())
    df_final = pd.DataFrame()
    cols_error =[]
    for label in labels:
        print("Processing "+label)
        if train[label].dtype.name == 'category':
            print("WARINING category type detected !!!")
            try:
                df_n  = knn_impute(target= train[label],\
                   attributes= train[features],\
                   k_neighbors=13,\
                   aggregation_method='mode',\
                  numeric_distance='euclidean',\
                  categorical_distance='hamming', missing_neighbors_threshold=0.8)
            except:
                cols_error.append(label)
                print("Error with column "+label)
                continue
        else:
            df_n  = knn_impute(target= train[label],\
               attributes= train[features],\
               k_neighbors=13,\
               aggregation_method='mean',\
              numeric_distance='euclidean',\
              categorical_distance='hamming', missing_neighbors_threshold=0.8)
        
        df = pd.DataFrame(data=df_n,columns=[label])
        #df_final = df_final.join(df_n)
        df_final = pd.concat([df_final, df_n], axis=1)
        free_memory()
        print("\n+++++++++++++++++++++++\n")
        #print(df_final.info())
        #print("\n+++++++++++++++++++++++\n")
        #count_nan = len(df_n) - df_n.count()
        #print("before {}\nafter {}".format(count_nan_brofe,count_nan))
    #df_final = df_final.drop(cols_error,axis=1)
    df_final = pd.concat([df_final, train[features]], axis=1)
    print(df_final.info())
    del train
    free_memory()
    df_final.to_csv('new_train.csv',index= None,header=True)
    return 0


def test_pararmeter(Sample_size, features, label):
    from sklearn.model_selection import GridSearchCV
    train = pd.read_csv("./train.csv", 
                            dtype= col_to_load,
                            usecols= col_to_load.keys(),
                            nrows= Sample_size)
    # Pre-process step
    print('Preprocessing ...')
    le = LabelEncoder()
    # Convert categorty type to int
    for cols in train:
       if train[cols].dtype.name == 'category':
            train[cols] = le.fit_transform(train[cols].astype(str))
            free_memory()
    train = train.fillna(0)
    train = reduce_mem_usage(train)   
    free_memory()

    X_train, X_test, y_train, y_test = train_test_split(train[features], train[label], test_size=0.3, random_state=2018)
    
    ## Standarize features
    scaler = StandardScaler(with_mean=True)
    X_std = scaler.fit_transform(X_train)
    Y_std = scaler.fit_transform(X_test)

    # set dictionary of para
    grid={"n_estimators":[1001,3001], 
          "criterion":["gini","entropy"],
          "max_depth":[17,25,71],
          "min_samples_split":[2, 5, 10]}# l1 lasso l2 ridge

    #Start testing
    print('Start test ...')
    #logreg=LogisticRegression()
    #logreg_cv=GridSearchCV(logreg, grid, cv=5, scoring='roc_auc', n_jobs = -1)
    #logreg_cv.fit(X_train,y_train)
    #print('\n------------------------\nEnd test\n')
    #print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
    #print("accuracy :",logreg_cv.best_score_)


    rf = RandomForestClassifier(random_state=2018,n_jobs = -1, verbose=True)
    rf_cv = GridSearchCV(rf, grid, cv=5, scoring='roc_auc', n_jobs = -1)
    rf_cv.fit(X_train,y_train)
    print('\n------------------------\nEnd test\n')
    print("tuned hpyerparameters :(best parameters) ",rf_cv.best_params_)
    print("accuracy :",rf_cv.best_score_)

def test_best_kmean(Sample_size, features):
    print("Load dataset ...")
    train = pd.read_csv("./train.csv", 
                            dtype= col_to_load,
                            usecols= col_to_load.keys(),
                            nrows= Sample_size)
    # Pre-process step
    print('Preprocessing ...')
    le = LabelEncoder()
    # Convert categorty type to int
    for cols in train:
       if train[cols].dtype.name == 'category':
            train[cols] = le.fit_transform(train[cols].astype(str))
            free_memory()
    train = train.fillna(-1)
    train = reduce_mem_usage(train)   
    free_memory()

    train = normalize(train, norm='l2')
    
    Sum_of_squared_distances = []

    K = range(1,35)
    for k in K:
        print("test k = {}".format(str(k)))
        km = KMeans(n_clusters=k, n_jobs=-1)
        km = km.fit(train)
        Sum_of_squared_distances.append(km.inertia_)
        print("Error = {}".format(str(km.inertia_)))
    
    print(km.cluster_centers_)
    df_centers = pd.DataFrame(km.cluster_centers_)
    df_centers.to_csv('kmean_centers.csv')
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k with sample size '+str(Sample_size))
    plt.grid()
    plt.savefig('Elbow Method')
    plt.show()

def export_newData(Sample_size, features, label):
    print("Load dataset ...")
    train = pd.read_csv("./train.csv", 
                            dtype= col_to_load,
                            usecols= col_to_load.keys(),
                            nrows= Sample_size)
    # Pre-process step
    print('Preprocessing ...')
    le = LabelEncoder()
    # Convert categorty type to int
    for cols in train:
       if train[cols].dtype.name == 'category':
            train[cols] = le.fit_transform(train[cols].astype(str))
            free_memory()
    train = train.fillna(-1)
    train = reduce_mem_usage(train)   
    free_memory()
    
    X_train, X_test, y_train, y_test = train_test_split(train[features], train[label], test_size=0.3, random_state=2018)
    # Standardizing the features
    X_train = StandardScaler(with_mean=True).fit_transform(X_train)
    X_test = StandardScaler(with_mean=True).fit_transform(X_test)
    
    X_train = normalize(X_train, norm='l1')
    X_test = normalize(X_test, norm='l1')

    print('Start Kmean ...')
    km = KMeans(n_clusters= 201, n_jobs=-1,max_iter=1500)
    km = km.fit(X_train)
    
    #distances = np.column_stack([np.sum((X_train - center)**2, axis=1)**0.5 for center in km.cluster_centers_])
    centers = km.cluster_centers_
    predictions = km.predict(X_train)
    print('Finish Kmean ...')
    free_memory()
    index = 0
    tf_0={}
    tf_1={}
    print('Prepare dataset ...')
    for idx in y_train.values:
        #if tf.keys().__contains__(predictions[idx])
        if idx == 0:
            if predictions[index] not in tf_0:
                tf_0[predictions[index]] = 1
            else:
                tf_0[predictions[index]] += 1
        else:
            if predictions[index] not in tf_1:
                tf_1[predictions[index]] = 1
            else:
                tf_1[predictions[index]]+=1
        index+=1
    
    label_0=[]
    label_1=[]
    for k, v in tf_0.items():
        if k in tf_1:
            if tf_0[k] > tf_1[k]:
                if k not in label_0:
                    label_0.append(k)
            elif tf_0[k] < tf_1[k]:
                if tf_0[k] not in label_1:
                    label_1.append(k)
            else:
                print('label {} is equals'.format(k))
        else:
            if tf_0[k] not in label_1:
                    label_1.append(k)
    X = []
    Y = []
    for l in label_0:
        X.append(centers[l])
        Y.append(0)

    for l in label_1:
        X.append(centers[l])
        Y.append(1)
    free_memory()
    print('Finish dataset ...')
    print('Start K-nn ...')
    
    ## creating odd list of K for KNN
    #myList = list(range(1,70))
    ## subsetting just the odd ones
    #neighbors = myList[0::2]#filter(lambda x: x % 2 != 0, myList)

    ## empty list that will hold cv scores
    #cv_scores = []

    ## perform 10-fold cross validation
    #for k in neighbors:
    #    print("start {} as k...".format(k))
    #    knn = KNeighborsClassifier(n_neighbors=k)
    #    scores = cross_val_score(knn, X_train, y_train, cv=4, scoring='roc_auc')
    #    cv_scores.append(scores.mean())
    #    print("finish {} as k...".format(k))
   
    ## changing to misclassification error
    #MSE = [1 - x for x in cv_scores]


    ## determining best k
    #optimal_k = neighbors[MSE.index(min(MSE))]
    #print("The optimal number of neighbors is {}".format(optimal_k))

    ## plot misclassification error vs k
    #plt.plot(neighbors, MSE)
    #plt.xlabel('Number of Neighbors K')
    #plt.ylabel('Misclassification Error')
    #plt.grid()
    #plt.show()
    cv = 10
    neigh = KNeighborsClassifier(n_neighbors=69,n_jobs=-1,algorithm='kd_tree')
    score  = cross_val_score(neigh, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs = -1)  
    print("Training score = {}".format(str(score.mean())))
    #neigh.fit(X, Y) 
    print("Predicting ...")
    y_pred = cross_val_predict(neigh, X_test, y_test,cv=cv)
    #y_pred = neigh.predict(X_test)
   
    print("Test report...")
    confusionmatrix = confusion_matrix(y_test, y_pred)
    print(confusionmatrix)
   
    free_memory()
    print(classification_report(y_test, y_pred))
    proba = cross_val_predict(neigh, X_test, y_test, cv=cv, method='predict_proba', n_jobs = -1)

    #proba = neigh.predict_proba(X_test)
    #proba = cross_val_predict(logreg, X_test, y_test, scoring='roc_auc', cv=3, method='predict_proba', n_jobs = -1)
    knn_roc_auc = roc_auc_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, proba[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='K-nn + Kmean (area = %0.2f)' % knn_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('KNN_KMEAN')
    plt.show()
    free_memory()

def main():
   #
   ks = list(col_to_load)
   features = ks[:68]
   label ='HasDetections'
   #test_best_kmean(1000000,features)
   #test_pararmeter(100000, features,label)
   #train_modelRF(300000, features,label)
   #train_model(10000,features,label)
   export_newData(300000,features,label)
   #fill_missing_data_Knn(Sample_size=10000)


if __name__ =='__main__':
    main() 