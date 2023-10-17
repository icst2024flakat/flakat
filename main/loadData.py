import pandas as pd
import numpy as np

import nltk
from tqdm import tqdm
from sklearn import utils
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import doc2vecPreprocess

"""
Input: csvLocation, onlyFlaky
Output: dataframe
Load the raw test case from parsed datasets
"""
def loadRaw(csvLocation, onlyFlaky):
    dataCSV = pd.read_csv(csvLocation)
    #dataCSV = dataCSV['raw']
    dataCSV['id']=range(1, len(dataCSV) + 1)
    # rows without category label, fill with NonFlaky
    dataCSV['category'].replace(np.nan, 'NonFlaky', inplace=True)
    
    # only pick rows with the desired flaky category
    if onlyFlaky:
        dataCSV = dataCSV[(dataCSV['category'] == 'ID') | (dataCSV['category'] == 'UD') |(dataCSV['category'] == 'OD') | (dataCSV['category'] == 'OD-Vic') | (dataCSV['category'] == 'OD-Brit') | (dataCSV['category'] == 'NOD') | (dataCSV['category'] == 'NDOD') | (dataCSV['category'] == 'NODI')]
    else:
        dataCSV = dataCSV[(dataCSV['category'] == 'ID') | (dataCSV['category'] == 'UD') |(dataCSV['category'] == 'OD') | (dataCSV['category'] == 'OD-Vic') | (dataCSV['category'] == 'OD-Brit') | (dataCSV['category'] == 'NOD') | (dataCSV['category'] == 'NDOD') | (dataCSV['category'] == 'NODI') | (dataCSV['category'] == 'NonFlaky')]

    return dataCSV

"""
Input: csvLocation, features, categoryDf
Output: dataframe
Load code2vec result from manually generated csv
"""
def loadCode2vecEmbedding(csvLocation, features, categoryDf, outputFeatures, onlyFlaky):
    dataCSV = pd.read_csv(csvLocation)
    dataCSV['id']=range(1, len(dataCSV) + 1)
    
    formatedDf = dataCSV[features]
    formatedDf['category'] = categoryDf['category']
    #formatedDf['category'].replace(np.nan, 'NonFlaky', inplace=True)

    # only pick rows with the desired flaky category
    if onlyFlaky:
        formatedDf = formatedDf[(formatedDf['category'] == 'ID') | (formatedDf['category'] == 'UD') |(formatedDf['category'] == 'OD') | (formatedDf['category'] == 'OD-Vic') | (formatedDf['category'] == 'OD-Brit') | (formatedDf['category'] == 'NOD') | (formatedDf['category'] == 'NDOD') | (formatedDf['category'] == 'NODI')]
    else:
        formatedDf = formatedDf[(formatedDf['category'] == 'ID') | (formatedDf['category'] == 'UD') |(formatedDf['category'] == 'OD') | (formatedDf['category'] == 'OD-Vic') | (formatedDf['category'] == 'OD-Brit') | (formatedDf['category'] == 'NOD') | (formatedDf['category'] == 'NDOD') | (formatedDf['category'] == 'NODI') | (formatedDf['category'] == 'NonFlaky')]

    return formatedDf[outputFeatures]

"""
Input: rawDataframe
Output: dataframe
Generate tfidf embedding and return it
"""
def loadTfidfEmbedding(rawDf, numFeatures, minDf, maxDf):
    tfidfDf = rawDf[['raw','category']]
  
    # use the sklearn tfidf vectorizer with number of features consistent with code2vec
    # minDf and maxDf keeping the same convention as previous flaky test course project
    tfidfconverter = TfidfVectorizer(max_features=numFeatures, min_df=minDf, max_df=maxDf)
    formatedDf = tfidfconverter.fit_transform(tfidfDf['raw'])

    return formatedDf.toarray()

"""
Tokenize raw test cases
"""
def doc2vecTokenizeText(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens

"""
Extract vectorized embedding
"""
def doc2vecVecForLearning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words)) for doc in sents])
    return targets, regressors

"""
Input: rawDataframe
Output: dataframe
Generate doc2vec embedding and return it
"""
def loadDoc2vecEmbedding(rawDf, numFeatures, window, minCount):
    docDf = rawDf[['raw','category']]
    docDf['raw'] = docDf['raw'].apply(doc2vecPreprocess)
    doc_tagged = docDf.apply(lambda r: TaggedDocument(words=doc2vecTokenizeText(r['raw']), tags=[r.category]), axis=1)
    
    tqdm.pandas(desc="progress-bar")
    model_dmm = Doc2Vec(dm=1, dm_mean=1, vector_size=numFeatures, window=window, negative=5, min_count=minCount, workers=5, alpha=0.065, min_alpha=0.065)
    model_dmm.build_vocab([x for x in tqdm(doc_tagged.values)])

    for epoch in range(30):
        model_dmm.train(utils.shuffle([x for x in tqdm(doc_tagged.values)]), total_examples=len(doc_tagged.values), epochs=1)
    model_dmm.alpha -= 0.002
    model_dmm.min_alpha = model_dmm.alpha

    doc_y, doc_X = doc2vecVecForLearning(model_dmm, doc_tagged)
    return doc_X