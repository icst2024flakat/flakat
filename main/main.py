from select import KQ_NOTE_LINK
import numpy as np
import pandas as pd
import os

from loadData import loadRaw
from loadData import loadCode2vecEmbedding
from loadData import loadTfidfEmbedding
from loadData import loadDoc2vecEmbedding

from reduction import pcaReduction
from reduction import ldaReduction
from reduction import isomapReduction
from reduction import tsneReduction
from reduction import umapReduction

from model import getKnnCategoryAccuracy
from model import getSvmCategoryAccuracy
from model import getRFCategoryAccuracy
from model import getRFCategoryAccuracyWithBO
from model import getGBDTCategoryAccuracy
from model import getGBDTCategoryAccuracyWithBO

from analysis import generateScatterPlot
from analysis import generateScatterPlotMatrix
from analysis import prettyPrintReductionQuality
from analysis import writeMatrixToCsv
from analysis import plotMatrix
from analysis import show3dScatterPlot

from analysis import getPredictionAccuracyByKnn
from analysis import getPredictionAccuracyBySVM
from analysis import getPredictionAccuracyByRandomForest
from analysis import getPredictionAccuracyByGBDT
from analysis import sequentialModelBasedOptimizationRandomForest




def gridSearchReductions(vectorDf, outputDf, outputDir, prefix, rawDf, reducedDimension,categoryNames, colors, size):

    embeddingList = []
    embeddingNames = []

    reduced = pcaReduction(vectorDf,reducedDimension)
    embeddingList.append(reduced)
    embeddingNames.append("PCA")
    if reducedDimension == 2: 
        generateScatterPlot(outputDir, prefix + '_PCA_2d.png',reduced, rawDf, categoryNames, colors, size)
    elif reducedDimension == 3: 
        show3dScatterPlot(prefix + '_PCA_3d', reduced, rawDf, categoryNames, colors, size)

    reduced = ldaReduction(vectorDf, outputDf, reducedDimension)
    embeddingList.append(reduced)
    embeddingNames.append("LDA")
    if reducedDimension == 2: 
        generateScatterPlot(outputDir, prefix + '_LDA_2d.png',reduced, rawDf, categoryNames, colors, size)
    elif reducedDimension == 3: 
        show3dScatterPlot(prefix + '_LDA_3d', reduced, rawDf, categoryNames, colors, size) 

    n_neighbour = [2,5,10,20,50]
    for n in n_neighbour:
        reduced = isomapReduction(vectorDf, outputDf, reducedDimension, n)
        embeddingList.append(reduced)
        tempName = "isomap_n_"+str(n)
        if reducedDimension == 2: 
            generateScatterPlot(outputDir, prefix + '_' + tempName + '_2d.png',reduced, rawDf, categoryNames, colors, size)
        elif reducedDimension == 3: 
            show3dScatterPlot(prefix + '_' + tempName + '_3d', reduced, rawDf, categoryNames, colors, size)
        embeddingNames.append(tempName)

    perp = [10,25,50,100]
    for p in perp:
        reduced = tsneReduction(vectorDf, reducedDimension, p, 1000)
        embeddingList.append(reduced)
        tempName = "tsne_p_"+str(p)
        if reducedDimension == 2: 
            generateScatterPlot(outputDir, prefix + '_' + tempName + '_2d.png',reduced, rawDf, categoryNames, colors, size)
        elif reducedDimension == 3: 
            show3dScatterPlot( prefix + '_' + tempName + '_3d', reduced, rawDf, categoryNames, colors, size)
        embeddingNames.append(tempName)

    n_neighbour = [5,20,80,320]
    min_dist = [0.05,0.2,0.5,0.8]
    for n in n_neighbour:
        for d in min_dist:
            reduced = umapReduction(vectorDf, reducedDimension, n, d)
            embeddingList.append(reduced)
            tempName = "umap_n_"+str(n)+"_d_"+str(d)
            if reducedDimension == 2: 
                generateScatterPlot(outputDir, prefix + '_' + tempName + '_2d.png',reduced, rawDf, categoryNames, colors, size)
            elif reducedDimension == 3: 
                show3dScatterPlot(prefix + '_' + tempName + '_3d', reduced, rawDf, categoryNames, colors, size)
            embeddingNames.append(tempName)

def generateScatterPlotForOptimal(doc2vecDf, code2vecDf, tfidfDf, outputDf, outputDir, rawDf, categoryNames, colors, size):
    doc2vecReducedList = []
    code2vecReducedList = []
    tfidf2vecReducedList = []

    doc2vecReducedList.append(pcaReduction(doc2vecDf,2))
    doc2vecReducedList.append(ldaReduction(doc2vecDf,outputDf,2))
    doc2vecReducedList.append(isomapReduction(doc2vecDf, outputDf, 2, 20))
    doc2vecReducedList.append(tsneReduction(doc2vecDf, 2, 10, 1000))
    doc2vecReducedList.append(umapReduction(doc2vecDf, 2, 5, 0.2))

    code2vecReducedList.append(pcaReduction(code2vecDf,2))
    code2vecReducedList.append(ldaReduction(code2vecDf,outputDf,2))
    code2vecReducedList.append(isomapReduction(code2vecDf, outputDf, 2, 20))
    code2vecReducedList.append(tsneReduction(code2vecDf, 2, 10, 1000))
    code2vecReducedList.append(umapReduction(code2vecDf, 2, 5, 0.2))
    
    tfidf2vecReducedList.append(pcaReduction(tfidfDf,2))
    tfidf2vecReducedList.append(ldaReduction(tfidfDf,outputDf,2))
    tfidf2vecReducedList.append(isomapReduction(tfidfDf, outputDf, 2, 20))
    tfidf2vecReducedList.append(tsneReduction(tfidfDf, 2, 10, 1000))
    tfidf2vecReducedList.append(umapReduction(tfidfDf, 2, 5, 0.2))

    reducedMatrix = []
    reducedMatrix.append(doc2vecReducedList)
    reducedMatrix.append(code2vecReducedList)
    reducedMatrix.append(tfidf2vecReducedList)

    generateScatterPlotMatrix(outputDir, 'multiple.png',reducedMatrix, None, rawDf, categoryNames, colors, size)

# generate the accuracy plots for both f1s and fdc using the optimal reduced emebdding
def generateAccuracyPlots(doc2vecDf, code2vecDf, tfidfDf, outputDf, outputDir):
    embeddingList = []
    embeddingNames = []

    doc2vecReduced = ldaReduction(doc2vecDf, outputDf,2)
    embeddingList.append(doc2vecReduced)
    embeddingNames.append("doc2vec")

    code2vecReduced = ldaReduction(code2vecDf, outputDf, 2)
    embeddingList.append(code2vecReduced)
    embeddingNames.append("code2vec")

    tfidfReduced = ldaReduction(tfidfDf, outputDf, 2)
    embeddingList.append(tfidfReduced)
    embeddingNames.append("tfidf")

    fold = 10

    kList = [2,5,10,20,50,100,200]

    knnResult = getPredictionAccuracyByKnn(kList, fold, embeddingList, outputDf.values, True)
    writeMatrixToCsv(knnResult, embeddingNames, outputDir+'knn_fdc.csv')
    plotMatrix(knnResult,embeddingNames,kList, "knn", "fdc", outputDir)

    knnResult = getPredictionAccuracyByKnn(kList, fold, embeddingList, outputDf.values, False)
    writeMatrixToCsv(knnResult, embeddingNames, outputDir+'knn_f1s.csv')
    plotMatrix(knnResult,embeddingNames,kList, "knn", "f1s", outputDir)

    kernelList = ['linear', 'poly', 'rbf', 'sigmoid']

    svmResult = getPredictionAccuracyBySVM(kernelList, fold, embeddingList, outputDf.values, True)
    writeMatrixToCsv(svmResult, embeddingNames, outputDir+'svm_fdc.csv')
    plotMatrix(svmResult,embeddingNames, kernelList, "svm", "fdc", outputDir)

    svmResult = getPredictionAccuracyBySVM(kernelList, fold, embeddingList, outputDf.values, False)
    writeMatrixToCsv(svmResult, embeddingNames, outputDir+'svm_f1s.csv')
    plotMatrix(svmResult,embeddingNames, kernelList, "svm", "f1s",outputDir)
    
    minLeafList = [1,2,5,10,20,50,100,200,300,500]

    rfResult = getPredictionAccuracyByRandomForest(minLeafList, fold, embeddingList, outputDf.values, True)
    writeMatrixToCsv(rfResult, embeddingNames, outputDir+'randomForest_fdc.csv')
    plotMatrix(rfResult,embeddingNames, minLeafList, "rf", "fdc", outputDir)

    rfResult = getPredictionAccuracyByRandomForest(minLeafList, fold, embeddingList, outputDf.values, False)
    writeMatrixToCsv(rfResult, embeddingNames, outputDir+'randomForest_f1s.csv')
    plotMatrix(rfResult,embeddingNames, minLeafList, "rf", "f1s",outputDir)

    minLeafList = [1,2,5,10,20,50,100,200,300,500]

    gbdtResult = getPredictionAccuracyByGBDT(minLeafList, fold, embeddingList, outputDf.values, True)
    writeMatrixToCsv(gbdtResult, embeddingNames, outputDir+'GBDT_fdc.csv')
    plotMatrix(gbdtResult,embeddingNames, minLeafList, "gbdt", "fdc", outputDir)

    gbdtResult = getPredictionAccuracyByGBDT(minLeafList, fold, embeddingList, outputDf.values, False)
    writeMatrixToCsv(gbdtResult, embeddingNames, outputDir+'GBDT_f1s.csv')
    plotMatrix(gbdtResult,embeddingNames, minLeafList, "gbdt", "f1s", outputDir)



def main():
    # location of raw data
    rawCsv = 'data/input/extracted-all-projects.csv'
    # location of vectorized result from code2vec
    code2vecCsv = 'data/input/vector.csv'
    outputDir = 'data/output/'
    features = ['feature_0','feature_1','feature_2','feature_3','feature_4','feature_5','feature_6','feature_7','feature_8','feature_9','feature_10','feature_11','feature_12','feature_13','feature_14','feature_15','feature_16','feature_17','feature_18','feature_19','feature_20','feature_21','feature_22','feature_23','feature_24','feature_25','feature_26','feature_27','feature_28','feature_29','feature_30','feature_31','feature_32','feature_33','feature_34','feature_35','feature_36','feature_37','feature_38','feature_39','feature_40','feature_41','feature_42','feature_43','feature_44','feature_45','feature_46','feature_47','feature_48','feature_49','feature_50','feature_51','feature_52','feature_53','feature_54','feature_55','feature_56','feature_57','feature_58','feature_59','feature_60','feature_61','feature_62','feature_63','feature_64','feature_65','feature_66','feature_67','feature_68','feature_69','feature_70','feature_71','feature_72','feature_73','feature_74','feature_75','feature_76','feature_77','feature_78','feature_79','feature_80','feature_81','feature_82','feature_83','feature_84','feature_85','feature_86','feature_87','feature_88','feature_89','feature_90','feature_91','feature_92','feature_93','feature_94','feature_95','feature_96','feature_97','feature_98','feature_99','feature_100','feature_101','feature_102','feature_103','feature_104','feature_105','feature_106','feature_107','feature_108','feature_109','feature_110','feature_111','feature_112','feature_113','feature_114','feature_115','feature_116','feature_117','feature_118','feature_119','feature_120','feature_121','feature_122','feature_123','feature_124','feature_125','feature_126','feature_127','feature_128','feature_129','feature_130','feature_131','feature_132','feature_133','feature_134','feature_135','feature_136','feature_137','feature_138','feature_139','feature_140','feature_141','feature_142','feature_143','feature_144','feature_145','feature_146','feature_147','feature_148','feature_149','feature_150','feature_151','feature_152','feature_153','feature_154','feature_155','feature_156','feature_157','feature_158','feature_159','feature_160','feature_161','feature_162','feature_163','feature_164','feature_165','feature_166','feature_167','feature_168','feature_169','feature_170','feature_171','feature_172','feature_173','feature_174','feature_175','feature_176','feature_177','feature_178','feature_179','feature_180','feature_181','feature_182','feature_183','feature_184','feature_185','feature_186','feature_187','feature_188','feature_189','feature_190','feature_191','feature_192','feature_193','feature_194','feature_195','feature_196','feature_197','feature_198','feature_199','feature_200','feature_201','feature_202','feature_203','feature_204','feature_205','feature_206','feature_207','feature_208','feature_209','feature_210','feature_211','feature_212','feature_213','feature_214','feature_215','feature_216','feature_217','feature_218','feature_219','feature_220','feature_221','feature_222','feature_223','feature_224','feature_225','feature_226','feature_227','feature_228','feature_229','feature_230','feature_231','feature_232','feature_233','feature_234','feature_235','feature_236','feature_237','feature_238','feature_239','feature_240','feature_241','feature_242','feature_243','feature_244','feature_245','feature_246','feature_247','feature_248','feature_249','feature_250','feature_251','feature_252','feature_253','feature_254','feature_255','feature_256','feature_257','feature_258','feature_259','feature_260','feature_261','feature_262','feature_263','feature_264','feature_265','feature_266','feature_267','feature_268','feature_269','feature_270','feature_271','feature_272','feature_273','feature_274','feature_275','feature_276','feature_277','feature_278','feature_279','feature_280','feature_281','feature_282','feature_283','feature_284','feature_285','feature_286','feature_287','feature_288','feature_289','feature_290','feature_291','feature_292','feature_293','feature_294','feature_295','feature_296','feature_297','feature_298','feature_299','feature_300','feature_301','feature_302','feature_303','feature_304','feature_305','feature_306','feature_307','feature_308','feature_309','feature_310','feature_311','feature_312','feature_313','feature_314','feature_315','feature_316','feature_317','feature_318','feature_319','feature_320','feature_321','feature_322','feature_323','feature_324','feature_325','feature_326','feature_327','feature_328','feature_329','feature_330','feature_331','feature_332','feature_333','feature_334','feature_335','feature_336','feature_337','feature_338','feature_339','feature_340','feature_341','feature_342','feature_343','feature_344','feature_345','feature_346','feature_347','feature_348','feature_349','feature_350','feature_351','feature_352','feature_353','feature_354','feature_355','feature_356','feature_357','feature_358','feature_359','feature_360','feature_361','feature_362','feature_363','feature_364','feature_365','feature_366','feature_367','feature_368','feature_369','feature_370','feature_371','feature_372','feature_373','feature_374','feature_375','feature_376','feature_377','feature_378','feature_379','feature_380','feature_381','feature_382','feature_383']
    col = ['id','name','feature_0','feature_1','feature_2','feature_3','feature_4','feature_5','feature_6','feature_7','feature_8','feature_9','feature_10','feature_11','feature_12','feature_13','feature_14','feature_15','feature_16','feature_17','feature_18','feature_19','feature_20','feature_21','feature_22','feature_23','feature_24','feature_25','feature_26','feature_27','feature_28','feature_29','feature_30','feature_31','feature_32','feature_33','feature_34','feature_35','feature_36','feature_37','feature_38','feature_39','feature_40','feature_41','feature_42','feature_43','feature_44','feature_45','feature_46','feature_47','feature_48','feature_49','feature_50','feature_51','feature_52','feature_53','feature_54','feature_55','feature_56','feature_57','feature_58','feature_59','feature_60','feature_61','feature_62','feature_63','feature_64','feature_65','feature_66','feature_67','feature_68','feature_69','feature_70','feature_71','feature_72','feature_73','feature_74','feature_75','feature_76','feature_77','feature_78','feature_79','feature_80','feature_81','feature_82','feature_83','feature_84','feature_85','feature_86','feature_87','feature_88','feature_89','feature_90','feature_91','feature_92','feature_93','feature_94','feature_95','feature_96','feature_97','feature_98','feature_99','feature_100','feature_101','feature_102','feature_103','feature_104','feature_105','feature_106','feature_107','feature_108','feature_109','feature_110','feature_111','feature_112','feature_113','feature_114','feature_115','feature_116','feature_117','feature_118','feature_119','feature_120','feature_121','feature_122','feature_123','feature_124','feature_125','feature_126','feature_127','feature_128','feature_129','feature_130','feature_131','feature_132','feature_133','feature_134','feature_135','feature_136','feature_137','feature_138','feature_139','feature_140','feature_141','feature_142','feature_143','feature_144','feature_145','feature_146','feature_147','feature_148','feature_149','feature_150','feature_151','feature_152','feature_153','feature_154','feature_155','feature_156','feature_157','feature_158','feature_159','feature_160','feature_161','feature_162','feature_163','feature_164','feature_165','feature_166','feature_167','feature_168','feature_169','feature_170','feature_171','feature_172','feature_173','feature_174','feature_175','feature_176','feature_177','feature_178','feature_179','feature_180','feature_181','feature_182','feature_183','feature_184','feature_185','feature_186','feature_187','feature_188','feature_189','feature_190','feature_191','feature_192','feature_193','feature_194','feature_195','feature_196','feature_197','feature_198','feature_199','feature_200','feature_201','feature_202','feature_203','feature_204','feature_205','feature_206','feature_207','feature_208','feature_209','feature_210','feature_211','feature_212','feature_213','feature_214','feature_215','feature_216','feature_217','feature_218','feature_219','feature_220','feature_221','feature_222','feature_223','feature_224','feature_225','feature_226','feature_227','feature_228','feature_229','feature_230','feature_231','feature_232','feature_233','feature_234','feature_235','feature_236','feature_237','feature_238','feature_239','feature_240','feature_241','feature_242','feature_243','feature_244','feature_245','feature_246','feature_247','feature_248','feature_249','feature_250','feature_251','feature_252','feature_253','feature_254','feature_255','feature_256','feature_257','feature_258','feature_259','feature_260','feature_261','feature_262','feature_263','feature_264','feature_265','feature_266','feature_267','feature_268','feature_269','feature_270','feature_271','feature_272','feature_273','feature_274','feature_275','feature_276','feature_277','feature_278','feature_279','feature_280','feature_281','feature_282','feature_283','feature_284','feature_285','feature_286','feature_287','feature_288','feature_289','feature_290','feature_291','feature_292','feature_293','feature_294','feature_295','feature_296','feature_297','feature_298','feature_299','feature_300','feature_301','feature_302','feature_303','feature_304','feature_305','feature_306','feature_307','feature_308','feature_309','feature_310','feature_311','feature_312','feature_313','feature_314','feature_315','feature_316','feature_317','feature_318','feature_319','feature_320','feature_321','feature_322','feature_323','feature_324','feature_325','feature_326','feature_327','feature_328','feature_329','feature_330','feature_331','feature_332','feature_333','feature_334','feature_335','feature_336','feature_337','feature_338','feature_339','feature_340','feature_341','feature_342','feature_343','feature_344','feature_345','feature_346','feature_347','feature_348','feature_349','feature_350','feature_351','feature_352','feature_353','feature_354','feature_355','feature_356','feature_357','feature_358','feature_359','feature_360','feature_361','feature_362','feature_363','feature_364','feature_365','feature_366','feature_367','feature_368','feature_369','feature_370','feature_371','feature_372','feature_373','feature_374','feature_375','feature_376','feature_377','feature_378','feature_379','feature_380','feature_381','feature_382','feature_383']

    #categoryNames = ['ID','OD','UD','OD-Vic','OD-Brit','NOD','NDOD','NDOI','NonFlaky']
    #colors = ["blue", "red", "indigo", "green", "yellow", "silver", "maroon", "peru", "aqua"]
    categoryNames = ['ID','OD','UD','OD-Vic','OD-Brit','NOD','NDOD','NDOI']
    colors = ["blue", "red", "indigo", "green", "yellow", "silver", "maroon", "peru"]

    rawDf = loadRaw(rawCsv, True)
    outputDf = rawDf['category']
    code2vecDf = loadCode2vecEmbedding(code2vecCsv, col, rawDf, features, True)
    tfidfDf = loadTfidfEmbedding(rawDf, 384, 5, 0.7)
    doc2vecDf = loadDoc2vecEmbedding(rawDf, 384, 10, 1)

    sampling = 0

    reductionDimention = 2
    IsomapN = 5
    tSNEp = 50
    tSNEi = 1000
    UMAPn = 320
    UMAPd = 0.05

    nameEmbedding = ""
    nameSampling = ""
    nameReduction = ""
    nameScatter = ""
    nameClassifier = ""
    nameOption = ""

    fold = 10
    metrics = ['precision', 'recall', 'f1s', 'mcc', 'fdc']

    knnK=[500]
    svmKernel=['linear','poly', 'rbf', 'sigmoid']
    svmC = [0.1, 1, 10, 100]

    rfConfig={'max_depth': 30, 'max_leaf_nodes': 125, 'max_samples': 0.5578022123369274, 'min_impurity_decrease': 0.004788117051408303, 'min_samples_leaf': 41, 'min_samples_split': 227, 'min_weight_fraction_leaf': 0.0018379269493552176, 'n_estimators': 172}
    # doc2vec
    #{'max_depth': 116, 'max_leaf_nodes': 286, 'max_samples': 0.6008105922992728, 'min_impurity_decrease': 0.008335864754482336, 'min_samples_leaf': 34, 'min_samples_split': 217, 'min_weight_fraction_leaf': 0.041835274717841445, 'n_estimators': 146}
    # code2vec
    #{'max_depth': 30, 'max_leaf_nodes': 125, 'max_samples': 0.5578022123369274, 'min_impurity_decrease': 0.004788117051408303, 'min_samples_leaf': 41, 'min_samples_split': 227, 'min_weight_fraction_leaf': 0.0018379269493552176, 'n_estimators': 172}
    # tfidf
    #{'max_depth': 53, 'max_leaf_nodes': 324, 'max_samples': 0.5096968938687001, 'min_impurity_decrease': 0.020294613947313078, 'min_samples_leaf': 70, 'min_samples_split': 203, 'min_weight_fraction_leaf': 0.00522892296925322, 'n_estimators': 133}
   
    gbdtConfig = {
        'learning_rate': 0.5433990284529154, 
        'max_depth': 17, 
        'min_samples_leaf': 88, 
        'min_samples_split': 305, 
        'n_estimators': 187
        }
    
    while True:
        cmd = input('Enter command\n')
        
        if cmd == 'exit':
            print('Exit program')
            break

        elif len(cmd) != 6:
            print('Invalid command')
        
        else:
            """
            1. embedding to use 1-doc2vec 2-code2vec 3-tfidf
            2. sampling 0-none 1-TL 2-SMOTE 3-TL&SMOTE 4-SMOTE&TL
            3. reduction 1-PCA 2-LDA 3-Isomap 4-t-SNE 5-UMAP
            4. visualization of reduced embedding 0-none 1-2d 2-3d
            5. classifier 1-KNN 2-SVM 3-RF 4-RF with tuning 5-GBDT 6-GBDT with tuning
            6. result 0-csv 1-no log
            """
            for i, c in enumerate(cmd):
                if i == 0:
                    if c == '1':
                        vectorEmbedding = doc2vecDf
                        nameEmbedding = "doc2vec"
                    elif c == '2':
                        vectorEmbedding = code2vecDf
                        nameEmbedding = "code2vec"
                    elif c == '3':
                        vectorEmbedding = tfidfDf
                        nameEmbedding = "tfidf"
                
                elif i == 1:
                    vector_val, vector_cat = vectorEmbedding, outputDf.values
                    if c == '0':
                        sampling = 0
                        nameSampling = "NoSampling"
                    elif c == '1':
                        sampling = 1
                        nameSampling = "TL"
                    elif c == '2':
                        sampling = 2
                        nameSampling = "SMOTE"
                    elif c == '3':
                        sampling = 3
                        nameSampling = "TL+SMOTE"
                    elif c == '4':
                        sampling = 4
                        nameSampling = "SMOTE+TL"

                elif i == 2:
                    if int(cmd[3]) == 0:
                        reductionDimention = 2
                    else:
                        reductionDimention = int(cmd[3])+1
                        
                    if c =='1':
                        reducedEmbedding = pcaReduction(vector_val, reductionDimention)
                        nameReduction = "PCA"
                    elif c == '2':
                        reducedEmbedding = ldaReduction(vector_val, vector_cat, reductionDimention)
                        nameReduction = "LDA"
                    elif c == '3':
                        reducedEmbedding = isomapReduction(vector_val, vector_cat, reductionDimention, IsomapN)
                        nameReduction = "Isomap"
                    elif c == '4':
                        reducedEmbedding = tsneReduction(vector_val, reductionDimention, tSNEp, tSNEi)
                        nameReduction = "tSNE"
                    elif c == '5':
                        reducedEmbedding = umapReduction(vector_val, reductionDimention, UMAPn, UMAPd)
                        nameReduction = "UMAP"

                elif i == 3:
                    if c == '0':
                        pass
                        #print("No scatter plots")
                    elif c == '1':
                        generateScatterPlot(outputDir,nameEmbedding+"_"+nameReduction+"_2d.png", reducedEmbedding, vector_cat, categoryNames, colors, [10,10])
                        nameScatter = "2"
                    elif c == '2':
                        show3dScatterPlot(nameEmbedding+"_"+nameReduction+"_3d", reducedEmbedding, vector_cat, categoryNames, colors, [10,10])
                        nameScatter = "3"

                elif i == 4:
                    if c == '1':

                        for k in knnK:
                            result = getKnnCategoryAccuracy(fold, sampling, reducedEmbedding, vector_cat, metrics, k)
                            nameClassifier = "KNN"
                            print(result['average']['f1s'])

                            #resultCsv = outputDir + nameEmbedding+"_"+nameSampling+"_"+nameReduction+"_"+nameScatter+"_"+nameClassifier+"_k="+str(k)+".csv"
                            #df = pd.DataFrame(result).transpose()
                            #df.to_csv(resultCsv)
                    elif c == '2':
                        for k in svmKernel:
                            for c in svmC:
                                result = getSvmCategoryAccuracy(fold, sampling, reducedEmbedding, vector_cat, metrics, k, c)
                                nameClassifier = "SVM"
                                
                                resultCsv = outputDir + nameEmbedding+"_"+nameSampling+"_"+nameReduction+"_"+nameScatter+"_"+nameClassifier+"_k="+str(k)+"_c="+str(c)+".csv"
                                df = pd.DataFrame(result).transpose()
                                df.to_csv(resultCsv)
                    elif c == '3':
                        result = getRFCategoryAccuracy(fold, sampling, reducedEmbedding, vector_cat, metrics, rfConfig)
                        nameClassifier = "RF"
                    elif c == '4':
                        result = getRFCategoryAccuracyWithBO(fold, sampling, reducedEmbedding, vector_cat, metrics)
                        nameClassifier = "RF_BO"
                    elif c == '5':
                        result = getGBDTCategoryAccuracy(fold, sampling, reducedEmbedding, vector_cat, metrics, gbdtConfig)
                        nameClassifier = "GBDT"
                    elif c == '6':
                        result = getGBDTCategoryAccuracyWithBO(fold, sampling, reducedEmbedding, vector_cat, metrics)
                        nameClassifier = "GBDT_BO"
                elif i == 5:
                    if c == '0':
                        
                        resultCsv = outputDir + nameEmbedding+"_"+nameSampling+"_"+nameReduction+"_"+nameScatter+"_"+nameClassifier+".csv"
                        df = pd.DataFrame(result).transpose()
                        df.to_csv(resultCsv)
                    elif c == '1':
                        pass
                    
        
        """
        # grid search and save results for all combinations into files
        if cmd == '1':

            vectorDf = doc2vecDf
            prefix = 'doc2vec'
            gridSearchReductions(vectorDf, outputDf, outputDir, prefix, rawDf, 2, categoryNames, colors, [15,15])

            vectorDf = code2vecDf
            prefix = 'code2vec'
            gridSearchReductions(vectorDf, outputDf, outputDir, prefix, rawDf, 2, categoryNames, colors, [15,15])

            vectorDf = tfidfDf
            prefix = 'tfidf'
            gridSearchReductions(vectorDf, outputDf, outputDir, prefix, rawDf, 2, categoryNames, colors, [15,15])

        # generate multiple 2d scatter plot for visualing reduction between each embedding and each technique
        elif cmd == '2':
            generateScatterPlotForOptimal(doc2vecDf, code2vecDf, tfidfDf, outputDf, outputDir, rawDf, categoryNames, colors, [15,10])

        # generate prediction accuracy
        elif cmd == '3':
            generateAccuracyPlots(doc2vecDf, code2vecDf, tfidfDf, outputDf, outputDir)
        
        # generate 3d scatter plots
        elif cmd == '4':

            vectorDf = doc2vecDf
            prefix = 'doc2vec'
            gridSearchReductions(vectorDf, outputDf, outputDir, prefix, rawDf, 3, categoryNames, colors, [10,10])

            vectorDf = code2vecDf
            prefix = 'code2vec'
            gridSearchReductions(vectorDf, outputDf, outputDir, prefix, rawDf, 3, categoryNames, colors, [10,10])

            vectorDf = tfidfDf
            prefix = 'tfidf'
            gridSearchReductions(vectorDf, outputDf, outputDir, prefix, rawDf, 3, categoryNames, colors, [10,10])
        
        elif cmd == '5':

            doc2vecReduced = ldaReduction(doc2vecDf, outputDf,2)
            code2vecReduced = ldaReduction(code2vecDf, outputDf,2)
            tfidfReduced = ldaReduction(tfidfDf, outputDf,2)

            # Number of trees in random forest
            n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 20)]
            # Maximum number of levels in tree
            max_depth = [int(x) for x in np.linspace(5, 100, num = 20)]
            max_depth.append(None)
            # Minimum number of samples required to split a node
            min_samples_split = [2, 5, 10, 20, 50, 100, 200, 500]
            # Minimum number of samples required at each leaf node
            min_samples_leaf = [1, 2, 5, 10, 20, 50, 100, 200]

            random_grid = {'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf}
                        
            sequentialModelBasedOptimizationRandomForest(random_grid, 100, outputDir+"doc_SMDO_FDC_result.csv", 5, doc2vecReduced, outputDf.values, 1)
            sequentialModelBasedOptimizationRandomForest(random_grid, 100, outputDir+"code_SMDO_FDC_result.csv", 5, code2vecReduced, outputDf.values, 1)
            sequentialModelBasedOptimizationRandomForest(random_grid, 100, outputDir+"tfidf_SMDO_FDC_result.csv", 5, tfidfReduced, outputDf.values, 1)

        elif cmd == 'exit':
            print('Exit program')
            break

        else:
            print('Invalid command')
        """
    

if __name__ == "__main__":
    main()