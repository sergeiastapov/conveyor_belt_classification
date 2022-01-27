import os
import numpy as np
import soundfile as sf
import python_speech_features as spf

from utils.stft_calc import stft

#import matplotlib.pyplot as plt

# paths to class 1, 2, 3, WAV files, accordingly
paths_in = [
           r"../First visit 11.01.2022/SM4 sensoriga salvestatud/Location 2 sild normaalne heli",
           r"../First visit 11.01.2022/SM4 sensoriga salvestatud/Location 3 sild vigane rullik",
           r"../First visit 11.01.2022/SM4 sensoriga salvestatud/Location 4 katte all vigane rullik"
           ]

# paths to feature sets of class 1, 2, 3, WAV files, accordingly
paths_out = [
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

frame_length = 0.032 # seconds
overlap = 2

if __name__ == "__main__":
    
    for i in range(len(paths_in)):
        path_in = paths_in[i]
        path_out = paths_out[i]
        # ensure out path exists
        os.makedirs(path_out, exist_ok=True)
    
        # Create list of WAV files in all subdirectories of root
        list_of_wav_in = list()
        for (dirpath, dirnames, filenames) in os.walk(path_in):
            list_of_wav_in += [os.path.join(dirpath, file) for file in filenames if file.lower().endswith('.wav')]
        #os.path.basename(f)[:-15]
        
        # extract features from each WAV
        for wav_path in list_of_wav_in:
            # read WAV
            audio, sampling_rate = sf.read(wav_path) #, always_2d = True)
            #audio = audio.T
            fftsize = int(frame_length * sampling_rate)
            
            ## ~~~ DO STFT ~~~ ##
            audio_stft = stft(audio, fftsize, overlap) # complex 2D spectrum
            audio_stft_abs = np.abs(audio_stft) # magnitude spectrum
            # save to numpy array
            path_stft = os.path.join(path_out, "stft")
            os.makedirs(path_stft, exist_ok=True)
            np.save(os.path.join(path_stft, os.path.basename(wav_path)[:-4]), audio_stft_abs)
            
            ## ~~~ DO Fbank energies ~~ ##
            audio_fbank = spf.fbank(audio, samplerate=sampling_rate, \
                          winlen=frame_length, winstep=frame_length/2, \
                          nfilt=40, nfft=fftsize, lowfreq=80, highfreq=None, \
                          preemph=0, winfunc=np.hamming)[0]
            # save to numpy array
            path_fbank = os.path.join(path_out, "fbank")
            os.makedirs(path_fbank, exist_ok=True)
            np.save(os.path.join(path_fbank, os.path.basename(wav_path)[:-4]), audio_fbank)
            
            ## ~~~ DO Spectral Subband Centroids ~~ ##
            audio_ssc = spf.ssc(audio, samplerate=sampling_rate, \
                        winlen=frame_length, winstep=frame_length/2, \
                        nfilt=40, nfft=fftsize, lowfreq=80, highfreq=None, \
                        preemph=0, winfunc=np.hamming)
            # save to numpy array
            path_ssc = os.path.join(path_out, "ssc")
            os.makedirs(path_ssc, exist_ok=True)
            np.save(os.path.join(path_ssc, os.path.basename(wav_path)[:-4]), audio_ssc)
            
            ## ~~~ DO MFCC ~~ ##
            audio_mfcc = spf.mfcc(audio, samplerate=sampling_rate, \
                         winlen=frame_length, winstep=frame_length/2, \
                         numcep=20, nfilt=40, nfft=fftsize, lowfreq=80, highfreq=None, \
                         preemph=0, ceplifter=22, appendEnergy=True, winfunc=np.hamming)
            # save to numpy array
            path_mfcc = os.path.join(path_out, "mfcc")
            os.makedirs(path_mfcc, exist_ok=True)
            np.save(os.path.join(path_mfcc, os.path.basename(wav_path)[:-4]), audio_mfcc)
            
            
            
            
            
            
            
            
            
            
            
            
            