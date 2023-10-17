from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import Isomap
import umap

"""
Input: vector embedding, number of components
Output: reduced vector embeddding at lower dimension
"""
def pcaReduction(vector, components):
    pca = PCA(n_components=components , random_state=0)
    return pca.fit_transform(vector)

"""
Input: vector embedding, category result, number of components
Output: reduced vector embeddding at lower dimension
"""
def ldaReduction(vector, result, components):
    lda = LinearDiscriminantAnalysis(n_components=components)
    return lda.fit_transform(vector, result)

"""
Input: vector embedding, category result, number of components, number of neighbours to consider
Output: reduced vector embeddding at lower dimension
"""
def isomapReduction(vector, result, components, neighbour):
    isomap = Isomap(n_neighbors=neighbour, n_components=components)
    return isomap.fit_transform(vector, result)

"""
Input: vector embedding, number of components, perplexity, number of iteration
Output: reduced vector embeddding at lower dimension
"""
def tsneReduction(vector, components, perp, iteration):
    tsne = TSNE(n_components=components, verbose=1, perplexity=perp, n_iter=iteration)
    return tsne.fit_transform(vector)
    
"""
Input: vector embedding, number of components, number of neighbour, minimum distance
Output: reduced vector embeddding at lower dimension
"""
def umapReduction(vector, components, neighbour, distance):
    umap_reducer = umap.UMAP(n_neighbors=neighbour, n_components=components, min_dist=distance)
    return umap_reducer.fit_transform(vector)
