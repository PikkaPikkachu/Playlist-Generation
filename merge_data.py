import pandas as pd

def find_songs_by_word(word, data):
    return data.loc[data['song_title'].str.contains(word, regex=True, na=False)]



df = pd.read_csv('./data/songs_clean.csv')
pdf = pd.read_csv('./data/top_tracks_clean.csv')
df = df.drop(['Unnamed: 0.1'], axis = 1)

pdf.rename(columns={'name': 'song_title', 'artists': 'artist'}, inplace=True)

words = ['Mozart', 'Mineur', 'Major', 'Minor', 'flat', 'Nocturne']

for w in words:
    pdf = pdf.append(find_songs_by_word(w, df), ignore_index=True)



#data = df.append(pdf, ignore_index=True)
pdf.to_csv("./data/pdata.csv", sep=',')
