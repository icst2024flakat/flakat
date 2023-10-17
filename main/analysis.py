import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import csv
import random

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef

from model import getKnnAccuracy
from model import getSvmAccuracy
from model import getRandomForestAccuracy
from model import getGBDTAccuracy


"""
generate the 2d projection of reduced embedding
"""
def generateScatterPlot(outputDir, outputName, vector, result, categoryName, colors, figSize):
    lw = 2
    plt.figure(figsize=figSize)
    for color, i, category_names in zip(colors, categoryName, categoryName):
        plt.scatter(
            vector[result == i, 0], vector[result == i, 1], color=color, alpha=0.5, lw=lw, label=category_names
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    #plt.title("PCA of flaky tests dataset")
    plt.savefig(outputDir+outputName)
    return

def generateScatterPlotMatrix(outputDir, outputName, vectorMatrix, vectorName, result, categoryName, colors, figSize):
    fig, axs = plt.subplots(len(vectorMatrix), len(vectorMatrix[0]), figsize=figSize)
    lw = 2
    for x in range(len(vectorMatrix)):
        for y in range(len(vectorMatrix[0])):
            for color, i, category_names in zip(colors, categoryName, categoryName):
                axs[x, y].scatter(
                    vectorMatrix[x][y][result['category'] == i, 0], vectorMatrix[x][y][result['category'] == i, 1], color=color, alpha=0.5, lw=lw, label=category_names
                )

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    
    fig.savefig(outputDir+outputName)
    return

def show3dScatterPlot(outputName, vector, result, categoryName, colors, figSize):
    fig = plt.figure(outputName, figsize = figSize)
    
    ax = fig.add_subplot(111,projection='3d')
    lw = 2

    for color, i, category_names in zip(colors, categoryName, categoryName):
        ax.scatter(
            vector[result['category'] == i, 0], vector[result['category'] == i, 1], vector[result['category'] == i, 2], color=color, alpha=0.5, lw=lw, label=category_names
        )

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
    return


def getPredictionAccuracyBySVM(kernelList, fold, embeddingList ,output, metricsOption):
    accuracyMatrix = []
    for k in kernelList:
        accuracyList = []
        for embedding in embeddingList:
            accuracyList.append(getSvmAccuracy(k,fold, embedding ,output, metricsOption))
        accuracyMatrix.append(accuracyList)
    return accuracyMatrix

def getPredictionAccuracyByRandomForest(minLeafList, fold, embeddingList ,output, metricsOption):
    accuracyMatrix = []
    for m in minLeafList:
        accuracyList = []
        for embedding in embeddingList:
            # all default other than minimum sample leaf size
            accuracyList.append(getRandomForestAccuracy(None, m, 2, 10, fold, embedding ,output, metricsOption))
        accuracyMatrix.append(accuracyList)
    return accuracyMatrix

"""
Run sequential model-based optimization on random forest classifier
Given a grid of hyperparameter space, picking
"""
def sequentialModelBasedOptimizationRandomForest(grid, iteration, fileDirectory, fold, embedding, output, metricsOption):
    with open(fileDirectory, 'w', encoding='UTF8', newline='') as f:
        
        accuracy = 0
        hpList = list(grid.keys())
        csvColumns = hpList
        csvColumns.append("accuracy")

        writer = csv.writer(f)
        writer.writerow(csvColumns)

        i = 0
        while i < iteration:
            print("SMBO iteration " + str(i))
            if i == 0:
                numEstimators = grid['n_estimators'][random.randrange(len(grid['n_estimators']))]
                maxDepth = grid['max_depth'][random.randrange(len(grid['max_depth']))]
                minSamplesSplit = grid['min_samples_split'][random.randrange(len(grid['min_samples_split']))]
                minSamplesLeaf = grid['min_samples_leaf'][random.randrange(len(grid['min_samples_leaf']))]

                accuracy = getRandomForestAccuracy(maxDepth, minSamplesLeaf, minSamplesSplit, numEstimators, fold, embedding ,output, metricsOption)

                data = []
                data.append(numEstimators)
                data.append(maxDepth)
                data.append(minSamplesSplit)
                data.append(minSamplesLeaf)
                data.append(accuracy)

                writer.writerow(data) 
            else:
                hyperparameterToUpdate = random.randrange(len(hpList))
                if hyperparameterToUpdate == 0:
                    temp = numEstimators
                    while temp == numEstimators:
                        numEstimators = grid['n_estimators'][random.randrange(len(grid['n_estimators']))]
                elif hyperparameterToUpdate == 1:
                    temp = maxDepth
                    while temp == maxDepth:
                        maxDepth = grid['max_depth'][random.randrange(len(grid['max_depth']))]
                elif hyperparameterToUpdate == 2:
                    temp = minSamplesSplit
                    while temp == minSamplesSplit:
                        minSamplesSplit = grid['min_samples_split'][random.randrange(len(grid['min_samples_split']))]
                elif hyperparameterToUpdate == 3:
                    temp = minSamplesLeaf
                    while temp == minSamplesLeaf:
                        minSamplesLeaf = grid['min_samples_leaf'][random.randrange(len(grid['min_samples_leaf']))]
                temp_acc = accuracy
                accuracy = getRandomForestAccuracy(maxDepth, minSamplesLeaf, minSamplesSplit, numEstimators, fold, embedding ,output, metricsOption)

                data = []
                data.append(numEstimators)
                data.append(maxDepth)
                data.append(minSamplesSplit)
                data.append(minSamplesLeaf)
                data.append(accuracy)

                writer.writerow(data) 

                if temp_acc > accuracy:
                    if hyperparameterToUpdate == 0:
                        numEstimators = temp
                    elif hyperparameterToUpdate == 1:
                        maxDepth = temp
                    elif hyperparameterToUpdate == 2:
                        minSamplesSplit = temp
                    elif hyperparameterToUpdate == 3:
                        minSamplesLeaf = temp

            i = i+1

    return 

def getPredictionAccuracyByGBDT(minLeafList, fold, embeddingList ,output, metricsOption):
    accuracyMatrix = []
    for m in minLeafList:
        accuracyList = []
        for embedding in embeddingList:
            # all default other than minimum sample leaf size
            accuracyList.append(getGBDTAccuracy(100, 0.1, 3, m, 2,fold, embedding ,output, metricsOption))
        accuracyMatrix.append(accuracyList)
    return accuracyMatrix

"""
Running knn and collect accuracy of various k to
determing how well local and global structure are preserved
"""
def getPredictionAccuracyByKnn(kList, fold, embeddingList ,output, metricsOption):
    accuracyMatrix = []
    for k in kList:
        accuracyList = []
        for embedding in embeddingList:
            accuracyList.append(getKnnAccuracy(k,fold, embedding ,output, metricsOption))
        accuracyMatrix.append(accuracyList)
    return accuracyMatrix

"""
Print computed reduction on console
"""
def prettyPrintReductionQuality(qualityMatrix, embeddingNames):
    row = ""
    for name in embeddingNames:
        row = row + " " + name
    print(row)
    for qualityList in qualityMatrix:
        row = ""
        for quality in qualityList:
            row = row + " " + str(quality)
        print(row)
    return

"""
Write the computed result to csv
"""
def writeMatrixToCsv(matrix, names, fileDirectory):
    with open(fileDirectory, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        writer.writerow(names)
        writer.writerows(matrix)

    return

"""
Plot the accuracy matrix for a specific classifier and a specific metric
"""
def plotMatrix(matrix, names, xAxis, classifierName ,metricName, outputDir):
    plt.figure(figsize=[10,5])

    xRange = range(len(xAxis))

    accuracyMatrix = list(reversed(list(zip(*matrix))[::-1]))
    for i, v in enumerate(names):
        plt.plot(xRange, accuracyMatrix[i], label = v, marker = "o")

    plt.xticks(xRange,xAxis)

    if classifierName == "knn":
        plt.xlabel("k")
    if classifierName == "svm":
        plt.xlabel("kernel type")
    if classifierName == "rf":
        plt.xlabel("minimum leaf node size")
    if classifierName == "gbdt":
        plt.xlabel("minimum leaf node size") 
    
    if metricName == "fdc":
        plt.ylabel("FDC")
    if metricName == "f1s":
        plt.ylabel("f1 score")


    plt.legend()
    fileDirectory = outputDir + classifierName + "_" + metricName + ".png"
    plt.savefig(fileDirectory)

    return
