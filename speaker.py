#Art by Ankit

import librosa
from librosa.feature import mfcc
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

#load data for first person
y0, sr0 = librosa.load('chappie_sound/chaipe0.wav')
y1, sr1 = librosa.load('chappie_sound/chaipe1.wav')
y2, sr2 = librosa.load('chappie_sound/chaipe2.wav')
y3, sr3 = librosa.load('chappie_sound/chaipe3.wav')
y4, sr4 = librosa.load('chappie_sound/chaipe4.wav')
y5, sr5 = librosa.load('chappie_sound/chaipe5.wav')
y6, sr6 = librosa.load('chappie_sound/chaipe6.wav')

#extrac feature for first person
mfccs0 = librosa.feature.mfcc(y=y0, sr=sr0, n_mfcc=40)
mfccs1 = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=40)
mfccs2 = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=40)
mfccs3 = librosa.feature.mfcc(y=y3, sr=sr3, n_mfcc=40)
mfccs4 = librosa.feature.mfcc(y=y4, sr=sr4, n_mfcc=40)
mfccs5 = librosa.feature.mfcc(y=y5, sr=sr5, n_mfcc=40)
mfccs6 = librosa.feature.mfcc(y=y6, sr=sr6, n_mfcc=40)

#load data for second person
y0, sr0 = librosa.load('gp/gp0.wav')
y1, sr1 = librosa.load('gp/gp1.wav')
y2, sr2 = librosa.load('gp/gp2.wav')
y3, sr3 = librosa.load('gp/gp3.wav')
y4, sr4 = librosa.load('gp/gp4.wav')
y5, sr5 = librosa.load('gp/gp5.wav')
y6, sr6 = librosa.load('gp/gp6.wav')

#extrac feature for second person
mfcc0 = librosa.feature.mfcc(y=y0, sr=sr0, n_mfcc=40)
mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=40)
mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=40)
mfcc3 = librosa.feature.mfcc(y=y3, sr=sr3, n_mfcc=40)
mfcc4 = librosa.feature.mfcc(y=y4, sr=sr4, n_mfcc=40)
mfcc5 = librosa.feature.mfcc(y=y5, sr=sr5, n_mfcc=40)
mfcc6 = librosa.feature.mfcc(y=y6, sr=sr6, n_mfcc=40)

#combine all them in X for input
X = [mfccs0[0, :50], mfccs1[0, :50], mfccs2[0, :50], mfccs3[0, :50], mfccs4[0, :50], mfccs5[0, :50], mfcc0[0, :50], mfcc1[0, :50], mfcc2[0, :50], mfcc3[0, :50], mfcc4[0, :50], mfcc5[0, :50]]
#label them for speaker recognition
y = ['ankit', 'ankit', 'ankit', 'ankit', 'ankit', 'ankit', 'gp', 'gp', 'gp', 'gp', 'gp', 'gp']
#lable them for their age
y1 = ['20', '20','20', '20','20', '20', '17', '17', '17', '17', '17', '17']

#train the model
neighSpeaker = KNeighborsClassifier(n_neighbors=3)
neighAge = KNeighborsClassifier(n_neighbors=3)
neighSpeaker.fit(X, y)
neighAge.fit(X, y1)

#predict the output
print(neighSpeaker.predict([mfccs6[0, :50]]))
print(neighAge.predict([mfcc6[0, :50]]))


