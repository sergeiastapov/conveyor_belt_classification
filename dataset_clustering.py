import os
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

features = [
           "stft",  # Magnitude spectrum applying STFT
           "fbank", # Mel-filterbank energies
           "ssc",   # Spectral Subband Centroids
           "mfcc"   # Mel Frequency Cepstral Coefficients
           ]

exp_name = "1st_visit"
# path to dataset folder
path_data = os.path.join("./datasets", exp_name)

if __name__ == "__main__":
    
    feat_name = "mfcc"
    dataset = np.load(os.path.join(path_data, feat_name+".npy"))
    
#    # choose 20% of dataset vectors for test set
#    n = int(0.2 * dataset.shape[0])
#    np.random.seed(123)
#    index_test = np.random.choice(dataset.shape[0], n, replace=False)
#    index_train = list(set(range(dataset.shape[0])) - set(index_test))
#    
#    feats_train = dataset[index_train,:-1]
#    feats_train_refclass = np.int32(dataset[index_train, -1])     
#    feats_eval = dataset[index_test,:-1]
#    feats_eval_refclass = np.int32(dataset[index_test, -1])
    
    # use whole dataset for clustering
    feats = dataset[:,:-1]
    feats_refclass = np.int32(dataset[:, -1])
    
    # reduce markup to 2 classes
    feats_refclass[feats_refclass > 1] = 1
    feats_refclass[feats_refclass > 1] = 1
    
    num_classes = np.max(feats_refclass)+1
    
    # apply PCA to get 2 or 3 principal components
    feats = (feats - feats.min())/(feats.max() - feats.min())
    pca = PCA(n_components=num_classes)
    lda = LDA(n_components=num_classes)
#    pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)])
    feats_pca = pca.fit_transform(feats)
#    feats_pca = lda.fit_transform(feats, feats_refclass)
    
    
    # perform clustering
    kmeans = KMeans(n_clusters=num_classes, random_state=0)
    feats_cluster = kmeans.fit_predict(feats_pca)
#    hull = list()
#    hull.append(ConvexHull(feats_pca[feats_cluster==0]))
#    hull.append(ConvexHull(feats_pca[feats_cluster==1]))
    
    # plot pca and clusters
    fig, ax = plt.subplots(1, figsize=(12,8))
    plt.scatter(feats_pca[feats_refclass==0,0], feats_pca[feats_refclass==0,1], c='r', alpha = 0.3, s=1)
    plt.scatter(feats_pca[feats_refclass==1,0], feats_pca[feats_refclass==1,1], c='b', alpha = 0.3, s=1)
    
    
#    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))
##    ax1.scatter(feats_pca[:,0], feats_pca[:,1], c=feats_refclass, alpha = 0.3, s=1)
#    ax1.scatter(feats_pca[feats_refclass==0,0], feats_pca[feats_refclass==0,1], c='r', alpha = 0.3, s=1)
#    ax1.scatter(feats_pca[feats_refclass==1,0], feats_pca[feats_refclass==1,1], c='b', alpha = 0.3, s=1)
#    
#    ax2.scatter(feats_pca[feats_cluster==0,0], feats_pca[feats_cluster==0,1], c='r', alpha = 0.6, s=5)
#    ax2.scatter(feats_pca[feats_cluster==1,0], feats_pca[feats_cluster==1,1], c='b', alpha = 0.6, s=5)
    
            
            
            
            
            