import os
import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

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
    
    feat_name = "stft"
    dataset = np.load(os.path.join(path_data, feat_name+".npy"))
    
    # choose 20% of dataset vectors for test set
    n = int(0.2 * dataset.shape[0])
    np.random.seed(123)
    index_test = np.random.choice(dataset.shape[0], n, replace=False)
    index_train = list(set(range(dataset.shape[0])) - set(index_test))
    
    feats_train = dataset[index_train,:-1]
    feats_train_refclass = np.int32(dataset[index_train, -1])     
    feats_test = dataset[index_test,:-1]
    feats_test_refclass = np.int32(dataset[index_test, -1])
    
    # reduce markup to 2 classes
    feats_train_refclass[feats_train_refclass > 1] = 1
    feats_test_refclass[feats_test_refclass > 1] = 1
    
    
    ####################################################################
    feats_train_refclass = to_categorical(feats_train_refclass)
    feats_test_refclass = to_categorical(feats_test_refclass)
    
    class_num = feats_train_refclass.shape[1]
    
    # define the keras model
    model = Sequential()
    model.add(Dense(120, input_dim=feats_train.shape[1], activation='relu')) # 120
    #model.add(Dropout(0.1))
    model.add(Dense(10, activation='relu')) # 30
    model.add(Dense(class_num, activation='softmax'))
    
    # compile the keras model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    #model.summary()
    
    # fit the keras model on the dataset
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='min')
    checkpointer = ModelCheckpoint(filepath="best_weights.hdf5", verbose=1, \
                                   save_best_only=True, monitor='val_accuracy', mode='max')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, \
                                       verbose=1, min_delta=1e-4, mode='min')
    
    history = model.fit(feats_train, feats_train_refclass, epochs=100, batch_size=10, \
                        validation_data=(feats_test, feats_test_refclass), verbose=1, \
                        callbacks=[early_stopping, checkpointer, reduce_lr_loss], validation_split=0.25)
    
    model.load_weights('./best_weights.hdf5')
            
    predicted = model.predict(feats_test)
    predicted = np.argmax(predicted, axis=1)
    target = np.argmax(feats_test_refclass, axis=1)
    confmat = confusion_matrix(target, predicted)
    acc = accuracy_score(target, predicted)
    print(pd.crosstab(pd.Series(target), pd.Series(predicted), \
                      rownames=['True'], colnames=['Predicted'], margins=True))
    print('\naccuracy = {:.2f}\n'.format(acc*100))
    print(classification_report(target, predicted))
            
    ####################################################################
#    clf = MLPClassifier(solver='adam', alpha=1e-4, hidden_layer_sizes=(120, 10), \
#                        max_iter=1000,activation = 'relu', random_state=1)
#    clf_train = clf.fit(feats_train, feats_train_refclass)
#    clf_test = clf.predict(feats_test)
#    clf_confmat = confusion_matrix(feats_test_refclass, clf_test)
#    clf_acc = accuracy_score(feats_test_refclass, clf_test)
#    print(pd.crosstab(pd.Series(feats_test_refclass), pd.Series(clf_test), \
#                      rownames=['True'], colnames=['Predicted'], margins=True))
#    print('accuracy = {:.2f}'.format(clf_acc*100))
#    print(classification_report(feats_test_refclass, clf_test))
            
#    joblib.dump(clf, 'weights_mlp_stft.joblib')
            
            
            
            
            