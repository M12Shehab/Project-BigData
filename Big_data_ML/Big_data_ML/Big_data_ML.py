import sys, gc
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from pyspark.sql import SparkSession
from pyspark.sql.functions import isnan, when, count, col

from data_types import *





def read_csv(filePath):
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
    
    le = LabelEncoder()
    # Convert categorty type to int
    for cols in train:
        if train[cols].dtype.name == 'category':
            train[cols] = le.fit_transform(train[cols].astype(str))
        free_memory()
    train = train.fillna(0)
    train = reduce_mem_usage(train)
    free_memory()
    X_train, X_test, y_train, y_test = train_test_split(train[features], train[label], test_size=0.3, random_state=0)
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

    confusionmatrix = confusion_matrix(y_test, y_pred)
    print(confusionmatrix)

    print(classification_report(y_test, y_pred))

    logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()


def main():
   #
   ks = list(col_to_load)
   features = ks[:68]
   label ='HasDetections'
   train_model(2000000, features,label)

if __name__ =='__main__':
    main() 