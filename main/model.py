from cmath import sqrt
import numpy as np
import pandas as pd
import math
from random import randrange

from preprocessing import samplingSMOTE, samplingSMOTEandTL, samplingTL, samplingTLandSMOTE

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import IsolationForest

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef

from bayes_opt import BayesianOptimization


"""
Compute the entropy of input series
"""
def findFlakyEntropy(input):
    entropy = 0

    unique, counts = np.unique(input, return_counts=True)
    input_dict = dict(zip(unique, counts))
    categories = input_dict.keys()
    base = len(categories)

    for category in categories:
        p_category = input_dict[category]/input.size
        entropy = entropy - p_category * math.log(p_category, base)
    
    return entropy

"""
Compute the mutual info between input series and output ndarray
"""
def findFlakyMutualInformation(input, output):
    mutualInfo = 0

    """
    categories = input.unique()
    input_dict = input.value_counts().to_dict()
    base = len(categories)
    """

    unique, counts = np.unique(input, return_counts=True)
    input_dict = dict(zip(unique, counts))
    categories = input_dict.keys()
    base = len(categories)

    unique, counts = np.unique(output, return_counts=True)
    output_dict = dict(zip(unique, counts))

    for category in categories:
        if category not in output_dict:
            output_dict[category] = 0

    for x_category in categories:
        for y_category in categories:
            index = 0 
            xy_occurrence = 0
            for i, v in enumerate(input):
                if v == x_category and output[i] == y_category:
                    xy_occurrence = xy_occurrence + 1
                index = index +1
            
            p_xy = xy_occurrence/input.size
            p_x = input_dict[x_category]/input.size
            p_y = output_dict[y_category]/input.size

            if p_xy > 0:
                mutualInfo = mutualInfo + p_xy * math.log(p_xy/(p_x*p_y), base)

    return mutualInfo

"""
Compute the flaky detection capacity from input series and output ndarray
Based on intrusion detection capacity
"""
def findFlakyDetectionCapacity(input, output):
    mutualInfo = findFlakyMutualInformation(input, output)
    entropy = findFlakyEntropy(input)

    return mutualInfo/entropy

def findCategoricalFlakyDetectionCapacity(input, output):
    return


"""
Calculate the accuracy of knn classifier with given setting
"""
def getKnnAccuracy(neighbour, fold, embedding, output, metricsOption):
    knn = KNeighborsClassifier(n_neighbors=neighbour)
    return runKFold(knn, fold, embedding, output, metricsOption)

"""
Calculate the accuracy of svm classifier with given setting
"""
def getSvmAccuracy(svmKernel, fold, embedding, output, metricsOption):
    svm_classifier = svm.SVC(kernel=svmKernel, random_state=0)
    return runKFold(svm_classifier, fold, embedding, output, metricsOption)

"""
Calculate the accuracy of random forest classifier with given setting
"""
def getRandomForestAccuracy(maxDepth, minLeaf, minSplit, estimator, fold, embedding, output, metricsOption):
    rfc = RandomForestClassifier(max_depth=maxDepth, min_samples_leaf=minLeaf, min_samples_split=minSplit, n_estimators=estimator, random_state=0)        
    return runKFold(rfc, fold, embedding, output, metricsOption)

"""
Calculate the accuracy of gbdt classifier with given setting
"""
def getGBDTAccuracy(estimator, learnRate, maxDepth, minLeaf, minSplit, fold, embedding, output, metricsOption):
    gbdt = GradientBoostingClassifier(n_estimators=estimator, learning_rate=learnRate,max_depth=maxDepth, min_samples_leaf=minLeaf, min_samples_split=minSplit, random_state=0)
    return runKFold(gbdt, fold, embedding, output, metricsOption)


def getKnnCategoryAccuracy(fold, sampling, embedding, output, metrics, k):
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    return runCategorySpecificKFold(knn, fold, sampling, embedding, output, metrics)

def getSvmCategoryAccuracy(fold, sampling, embedding, output, metrics, kernel, c):
    svm_classifier = svm.SVC(kernel=kernel, C=c, random_state=0)
    return runCategorySpecificKFold(svm_classifier, fold, sampling, embedding, output, metrics)

def getRFCategoryAccuracy(fold, sampling, embedding, output, metrics, config):

    rfc = RandomForestClassifier(
        max_leaf_nodes=config['max_leaf_nodes'],
        max_samples=config['max_samples'],
        min_impurity_decrease=config['min_impurity_decrease'],
        min_samples_leaf=config['min_samples_leaf'],
        min_samples_split=config['min_samples_split'],
        min_weight_fraction_leaf=config['min_weight_fraction_leaf'],
        n_estimators=config['n_estimators'],
        random_state=0)

    return runCategorySpecificKFold(rfc, fold, sampling, embedding, output, metrics)

X_tuning = 0
y_tuning = 0

def getRFCategoryAccuracyWithBO(fold, sampling, embedding, output, metrics):
    global X_tuning 
    global y_tuning
    criterionList = ["gini", "entropy", "log_loss"]
    
    X_tuning, X_test, y_tuning, y_test = train_test_split(embedding, output, test_size=0.33, random_state=0)

    rf_bo = BayesianOptimization(bo_params_rf, {
                                                'criterion':(0,2.999),
                                                'max_samples':(0.5,1),
                                                'max_depth':(1,200),
                                                'min_impurity_decrease':(0,0.5),
                                                'min_samples_leaf':(1,200),
                                                'min_samples_split':(2,400),
                                                'n_estimators':(100,200),
                                                'min_weight_fraction_leaf':(0,0.05),
                                                'max_leaf_nodes':(2,400)
                                            })

    boResult = rf_bo.maximize(n_iter=500, init_points=20,acq='ei')
    params = rf_bo.max['params']
    params['criterion']=  criterionList[int(params['criterion'])]
    params['max_depth']=  int(params['max_depth'])
    params['min_samples_leaf']=  int(params['min_samples_leaf'])
    params['min_samples_split']=  int(params['min_samples_split'])
    params['n_estimators']= int(params['n_estimators'])
    params['max_leaf_nodes']= int(params['max_leaf_nodes'])

    print(params)
    #{'max_depth': 30, 'max_leaf_nodes': 125, 'max_samples': 0.5578022123369274, 'min_impurity_decrease': 0.004788117051408303, 'min_samples_leaf': 41, 'min_samples_split': 227, 'min_weight_fraction_leaf': 0.0018379269493552176, 'n_estimators': 172}
    rf_optimal = RandomForestClassifier(criterion=params['criterion'],max_samples=params['max_samples'],max_depth=params['max_depth'],min_samples_leaf=params['min_samples_leaf'],min_samples_split=params['min_samples_split'],n_estimators=params['n_estimators'],min_weight_fraction_leaf=params['min_weight_fraction_leaf'],max_leaf_nodes=params['max_leaf_nodes'],random_state=0)

    return runCategorySpecificKFold(rf_optimal, fold, sampling, embedding, output, metrics)


def getGBDTCategoryAccuracy(fold, sampling, embedding, output, metrics, config):
    
    gbdt = GradientBoostingClassifier(
        learning_rate=config['learning_rate'],
        max_depth=config['max_depth'],
        min_samples_leaf=config['min_samples_leaf'],
        min_samples_split=config['min_samples_split'],
        n_estimators=config['n_estimators'],
        random_state=0)
    return runCategorySpecificKFold(gbdt, fold, sampling, embedding, output, metrics)

def getGBDTCategoryAccuracyWithBO(fold, sampling, embedding, output, metrics):

    global X_tuning 
    global y_tuning

    X_tuning, X_test, y_tuning, y_test = train_test_split(embedding, output, test_size=0.33, random_state=0)

    gbdt_bo = BayesianOptimization(bo_params_gbdt, {
                                              'learning_rate':(0.001,10),
                                              'max_depth':(1,200),
                                              'min_samples_leaf':(1,200),
                                              'min_samples_split':(2,400),
                                              'n_estimators':(100,200)
                                             })

    results = gbdt_bo.maximize(n_iter=200, init_points=20,acq='ei')
    params = gbdt_bo.max['params']
    params['max_depth']=  int(params['max_depth'])
    params['min_samples_leaf']=  int(params['min_samples_leaf'])
    params['min_samples_split']=  int(params['min_samples_split'])
    params['n_estimators']= int(params['n_estimators'])



    print(params)
    gbdt_optimal = GradientBoostingClassifier(learning_rate=params['learning_rate'],max_depth=params['max_depth'],min_samples_leaf=params['min_samples_leaf'],min_samples_split=params['min_samples_split'],n_estimators=params['n_estimators'],random_state=0)

    return runCategorySpecificKFold(gbdt_optimal, fold, sampling, embedding, output, metrics)


"""
The most atomic run of a given classifier and shuffle and repeat
metricsOption 0 - F1 score
1 - Flaky Detection Capacity
2 - Matthews Correlation Coefficient
"""
def runKFold(classifier, fold, sampling, embedding, output, metrics):
    sum = 0

    kf = KFold(n_splits=fold)
    result_dict = {}

    if 'precision' in metrics:
        result_dict['precision'] = []
    if 'recall' in metrics:
        result_dict['recall'] = []
    if 'f1s' in metrics:
        result_dict['f1s'] = []
    if 'mcc' in metrics:
        result_dict['mcc'] = []
    if 'fdc' in metrics:
        result_dict['fdc'] = []

    for train_index, test_index in kf.split(embedding):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = embedding[train_index], embedding[test_index]
        y_train, y_test = output[train_index], output[test_index]

        if sampling == 0:
            X_sampled, y_sampled = X_train, y_train
        elif sampling == 1:
            X_sampled, y_sampled = samplingTL(X_train, y_train)
        elif sampling == 2:
            X_sampled, y_sampled = samplingSMOTE(X_train, y_train)
        elif sampling == 3:
            X_sampled, y_sampled = samplingTLandSMOTE(X_train, y_train)
        elif sampling == 4:
            X_sampled, y_sampled = samplingSMOTEandTL(X_train, y_train)

        classifier.fit(X_sampled,y_sampled)
        y_pred = classifier.predict(X_test)
        
        if metricsOption == 0:
            sum += f1_score(y_test, y_pred, average='weighted')
        elif metricsOption == 1:
            sum += findFlakyDetectionCapacity(y_test, y_pred)
        elif metricsOption == 2:
            sum += matthews_corrcoef(y_test, y_pred)

    return sum/fold

def runCategorySpecificKFold(classifier, fold, sampling, embedding, output, metrics):

    kf = KFold(n_splits=fold, shuffle=True, random_state=randrange(1000))
    result_dict = {}
    unique, counts = np.unique(output, return_counts=True)
    for category in unique:
        result_dict[category] = {}
        if 'precision' in metrics:
            result_dict[category]['precision'] = []
        if 'recall' in metrics:
            result_dict[category]['recall'] = []
        if 'f1s' in metrics:
            result_dict[category]['f1s'] = []
        if 'mcc' in metrics:
            result_dict[category]['mcc'] = []
        if 'fdc' in metrics:
            result_dict[category]['fdc'] = []

    for train_index, test_index in kf.split(embedding):
        X_train, X_test = embedding[train_index], embedding[test_index]
        y_train, y_test = output[train_index], output[test_index]

        unique_train, counts_train = np.unique(y_train, return_counts=True)

        # avoid missing minority class
        while len(unique_train) < len(unique):
            kf_reshuffle = KFold(n_splits=fold, shuffle=True, random_state=randrange(1000))
            train_index, test_index = list(kf_reshuffle)[0]

            X_train, X_test = embedding[train_index], embedding[test_index]
            y_train, y_test = output[train_index], output[test_index]

            unique_train, counts_train = np.unique(y_train, return_counts=True)


        """
        df_describe = pd.DataFrame(X_train)
        print(df_describe.value_counts())
        df_describe = pd.DataFrame(y_train)
        print(df_describe.value_counts())
        """

        if sampling == 0:
            X_sampled, y_sampled = X_train, y_train
        elif sampling == 1:
            X_sampled, y_sampled = samplingTL(X_train, y_train)
        elif sampling == 2:
            X_sampled, y_sampled = samplingSMOTE(X_train, y_train)
        elif sampling == 3:
            X_sampled, y_sampled = samplingTLandSMOTE(X_train, y_train)
        elif sampling == 4:
            X_sampled, y_sampled = samplingSMOTEandTL(X_train, y_train)

        classifier.fit(X_sampled,y_sampled)
        y_pred = classifier.predict(X_test)
        
        cm = confusion_matrix(y_test,y_pred,labels=unique)

        FP = np.sum(cm, axis=0) - np.diag(cm)
        FN = np.sum(cm, axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm[:].sum() - (FP + FN + TP)

        TP = np.asarray([1 if x==0 else x for x in TP], dtype=np.int)
        
        if 'precision' in metrics:
            precision = TP/(TP+FN)
        if 'recall' in metrics:
            recall = TP/(TP+FP)
        if 'f1s' in metrics:
            f1score = TP/(TP+0.5*(FP+FN))
        if 'mcc' in metrics:
            mcc = (TP*TN-FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        if 'fdc' in metrics:
            fdc = findFlakyDetectionCapacity(y_test, y_pred)

        index = 0
        for category in unique:
            if 'precision' in metrics:
                result_dict[category]['precision'].append(precision[index])
            if 'recall' in metrics:
                result_dict[category]['recall'].append(recall[index])
            if 'f1s' in metrics:
                result_dict[category]['f1s'].append(f1score[index])
            if 'mcc' in metrics:
                result_dict[category]['mcc'].append(mcc[index])
            if 'fdc' in metrics:
                result_dict[category]['fdc'].append(fdc)

            index = index + 1
        
    for category in unique:
        for metric in metrics:
            result_dict[category][metric] = sum(result_dict[category][metric])/len(result_dict[category][metric])
    
    result_dict['average'] = {}
    for metric in metrics:
        result_dict['average'][metric] = 0
        for category in unique:
            result_dict['average'][metric] += result_dict[category][metric]
        result_dict['average'][metric] = result_dict['average'][metric] / len(unique)

    return result_dict

def stratified_kfold_score(clf,X,y,n_fold):
    X,y = X,y
    strat_kfold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=1)
    accuracy_list = []

    for train_index, test_index in strat_kfold.split(X, y):
        x_train_fold, x_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        # for now just do smote then tl. add other options later
        X_sampled, y_sampled = samplingSMOTEandTL(x_train_fold, y_train_fold)

        clf.fit(X_sampled, y_sampled)

        preds = clf.predict(x_test_fold)
        #accuracy_test = f1_score(preds,y_test_fold, average='macro')
        accuracy_test = findFlakyDetectionCapacity(y_test_fold, preds)
        accuracy_list.append(accuracy_test)
    
    return np.array(accuracy_list).mean()

    
def bo_params_rf(criterion, max_samples,n_estimators,min_impurity_decrease,min_samples_leaf,min_samples_split,max_depth,min_weight_fraction_leaf,max_leaf_nodes):
    criterionList = ["gini", "entropy", "log_loss"]
    params = {
        'criterion': criterionList[int(criterion)],
        'max_samples': max_samples,
        'max_depth': int(max_depth),
        'min_impurity_decrease':min_impurity_decrease,
        'min_samples_leaf':int(min_samples_leaf),
        'min_samples_split':int(min_samples_split),
        'n_estimators':int(n_estimators),
        'min_weight_fraction_leaf':min_weight_fraction_leaf,
        'max_leaf_nodes':int(max_leaf_nodes)

    }
    clf = RandomForestClassifier(criterion=params['criterion'], max_samples=params['max_samples'],max_depth=params['max_depth'],min_impurity_decrease=params['min_impurity_decrease'],min_samples_leaf=params['min_samples_leaf'],min_samples_split=params['min_samples_split'],n_estimators=params['n_estimators'],min_weight_fraction_leaf=params['min_weight_fraction_leaf'],max_leaf_nodes=params['max_leaf_nodes'])
    score = stratified_kfold_score(clf,X_tuning, y_tuning,5)
    #print(score)
    return score

def bo_params_gbdt(n_estimators,learning_rate,max_depth,min_samples_leaf,min_samples_split):
        
    params = {
        'learning_rate': learning_rate,
        'max_depth':int(max_depth),
        'min_samples_leaf':int(min_samples_leaf),
        'min_samples_split':int(min_samples_split),
        'n_estimators':int(n_estimators)
    }
    clf = GradientBoostingClassifier(learning_rate=params['learning_rate'],max_depth=params['max_depth'],min_samples_leaf=params['min_samples_leaf'],min_samples_split=params['min_samples_split'],n_estimators=params['n_estimators'])
    score = stratified_kfold_score(clf,X_tuning, y_tuning,5)
    #print(score)
    return score