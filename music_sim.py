import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding( ' utf-8 ' )

df = pd.read_csv('./data/Concentration.csv')
#df = df.sample(n=10)
pdf = df.drop(['duration_ms', 'Unnamed: 0','Unnamed: 0.1', 'song_title', 'artist'], axis = 1)

#print pdf
some = euclidean_distances(pdf,pdf)

some = (some - np.min(some))/np.ptp(some)
print df.song_title
plt.imshow(some, cmap='Blues', interpolation='nearest')
plt.colorbar()
tick_marks = [i for i in range (15)]
plt.xticks(tick_marks, df.song_title, rotation = 'vertical')
plt.yticks(tick_marks, df.song_title)
plt.show()
