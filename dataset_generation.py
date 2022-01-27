import os
import numpy as np

#import matplotlib.pyplot as plt

# paths to feature sets of class 1, 2, 3, WAV files, accordingly
paths_feat = [
             r"./feature_sets/belt_normal",
             r"./feature_sets/belt_fault",
             r"./feature_sets/under_fault"
             ]

features = [
           "stft",  # Magnitude spectrum applying STFT
           "fbank", # Mel-filterbank energies
           "ssc",   # Spectral Subband Centroids
           "mfcc"   # Mel Frequency Cepstral Coefficients
           ]

exp_name = "1st_visit"

# path to dataset folder
path_out = os.path.join("./datasets", exp_name)

if __name__ == "__main__":
    
    # ensure out path exists
    os.makedirs(path_out, exist_ok=True)
    
    for feature in features:
        dataset = list()
        
        for i in range(len(paths_feat)):
            path_feat = os.path.join(paths_feat[i], feature)
            # Create list of numpy files in the specific directory
            list_of_files_in = [os.path.join(path_feat, f) \
                                for f in os.listdir(path_feat) if f.lower().endswith('.npy')]
            
            for fname in list_of_files_in:
                feat_data = np.load(fname)
                feat_data = np.hstack((feat_data, i*np.ones((feat_data.shape[0], 1))))
                dataset.append(feat_data)
            
        dataset_arr = np.vstack(dataset)
        np.save(os.path.join(path_out, feature), dataset_arr)
                

            
            
            
            
            
            
            
            
            
            
            
            
            