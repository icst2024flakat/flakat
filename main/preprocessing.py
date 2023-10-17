import re

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

"""
Preprocessing raw test cases 
Remove special characters and split camel case names to words
"""
def doc2vecPreprocess(code):
    code = re.sub(r"(?<=\w)([A-Z])", r" \1", code)
    code = re.sub(r'[^A-Za-z0-9 ]+', '', code)
    code = code.lower()

    return code

def samplingTL(vector_val, vector_cat):
    tl = TomekLinks()
    X_tl, y_tl = tl.fit_resample(vector_val, vector_cat)

    return X_tl, y_tl

def samplingSMOTE(vector_val, vector_cat):
    smote = SMOTE(k_neighbors=3)
    X_smote, y_smote = smote.fit_resample(vector_val, vector_cat)

    return X_smote, y_smote

def samplingTLandSMOTE(vector_val, vector_cat):
    tl = TomekLinks()
    X_tl, y_tl = tl.fit_resample(vector_val, vector_cat)

    smote = SMOTE(k_neighbors=3)
    X_smote, y_smote = smote.fit_resample(X_tl, y_tl)

    return X_smote, y_smote

def samplingSMOTEandTL(vector_val, vector_cat):
    smote = SMOTE(k_neighbors=3)
    X_smote, y_smote = smote.fit_resample(vector_val, vector_cat)

    tl = TomekLinks()
    X_tl, y_tl = tl.fit_resample(X_smote, y_smote)

    return X_tl, y_tl