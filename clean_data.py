import pandas as pd
from sklearn import preprocessing

data = pd.read_csv('./data/top-tracks.csv')
data = data.drop(['key', 'mode', 'time_signature', 'id'], axis = 1)

data = data.dropna(axis = 0)

min_max_scaler = preprocessing.MinMaxScaler()
data[['tempo', 'loudness', 'acousticness',
        'instrumentalness', 'danceability', 'valence',
        'speechiness', 'energy', 'liveness']] = min_max_scaler.fit_transform(data[['tempo', 'loudness', 'acousticness',
                                                                                    'instrumentalness', 'danceability', 'valence',
                                                                                    'speechiness', 'energy', 'liveness']])


print data.head()

data.to_csv("./data/top_tracks_clean.csv", sep=',')
