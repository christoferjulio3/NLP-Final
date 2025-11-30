from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os
import shutil
import numpy as np
import librosa
import math

def audio_sampling(audio_path, song, sampling_rate, n_mels,cut_rate, model='ast'):
  '''
  This is the code to sample the audio based on the requirement of 2 models: audio spectrogram transformer 
  and whisper. 
  '''

  # load mp3 file with librosa
  y, _ = librosa.load(os.path.join(audio_path, song))

  # sampling
  sample = librosa.feature.melspectrogram(y=y, n_mels=n_mels, sr=sampling_rate, n_fft=400, hop_length=160, pad_mode='constant')
  sample_log = librosa.power_to_db(sample, ref=np.max)

  # finding the possible amount of sample of one song based on time resolution
  x = math.ceil(sample_log.shape[-1] / cut_rate)

  # padding with zero
  if sample_log.shape[-1] < (x * cut_rate):
    length_dev = abs(sample_log.shape[-1] - (x * cut_rate))
    width = [(0, 0)] * (sample_log.ndim - 1) + [(0, length_dev)]
    sample_log = np.pad(sample_log, width, mode= "constant")
  else: sample_log = sample_log

  # stacking the samples on the first dimension to simplify processing
  sample_split_log = np.hsplit(sample_log, x)
  sample_split_log_stack = np.stack(sample_split_log, 0)

  # swap the second and third dimension as the input of Audio spectrogram Transformer
  if model == 'ast':
    sample_split_log_stack = np.einsum('abc -> acb', sample_split_log_stack)

  return sample_split_log_stack

# save audio file
def save_audio(audio_path, name, files):
  filename = name + '.npy'
  np.save(os.path.join(audio_path, filename), files)


def similarity(sentence1, sentence2, vectorizer):

    tfidf_matrix = vectorizer.transform([sentence1, sentence2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

def get_similar_sentences(df, queries):
    '''
    This is the code for finding similarity between two sentences
    in this case, mp3 files name and the name in the dataset are vastly different, making adjusting the name 
    with cleaning the dataset alone is hard.
    Then, TF-IDF is used.
    '''
    vectorizer = TfidfVectorizer().fit(df['billboard'].str.lower())
    df_copy = df.copy()
    results = {}

    for i, query in enumerate(queries):
        query = query.lower()
        similarities = df['billboard'].apply(lambda x: similarity(query, x.lower(), vectorizer))
        most_similar_idx = similarities.idxmax()
        most_similar_score = similarities.max()
        results[query] = [(df.iloc[most_similar_idx]['billboard'], most_similar_score)]
        df_copy.at[most_similar_idx, 'billboard'] = query
        print(i)
        print(query)
        print(results[query])

    return results, df_copy


def move_used_files(destination_path, path, csv):

    count = 0
    count_fl = 0
    ls = list(csv['song_name'])
    for x in os.listdir(path):
      if x.lower() in ls:
        count += 1
        shutil.move(os.path.join(path,x), os.path.join(destination_path,x))
        print(f"Moved {x} to {destination_path}")
      else:
        count_fl += 1

    print(f'total of {count} files are kept')
    print(f'total of {count_fl} files are moved')

def separate_data(path, path_save, df, save=True):
    data = []

    explicit_labels = df.set_index('song_name')['explicit'].to_dict()

    for x in os.listdir(path):
        y = x.lower()
        name, ext = os.path.splitext(y)
        song = np.load(os.path.join(path, x))
        save_name = os.path.join(path_save, name)

        if y in explicit_labels:
            label = explicit_labels[y]
        else:
            print(f"song error, check the name: {y}")
            label = None

        print(f'now splitting {x}')
        for i, part in enumerate(song):
            new_name = f"{name}_part_{i}"
            data.append({'song_name': new_name, 'part': i, 'explicit': label})
            np.save(f"{save_name}_part_{i}", part)
            print(f'finished saving {x} part {i}')

    new_df = pd.DataFrame(data)

    if save == True:
        new_df.to_csv(os.path.join(path_save, 'ast_df_final_sep_1024.csv'), index=False)
        print('all are saved')
        return new_df
    else:
       return new_df


if __name__ == '__main__':
    
    # pth = r'D:/NLP final project/unused_data.csv'
    # pth_all_csv = r'D:/NLP final project/balanced_data_new_clean.csv'
    # pth_dst = r'D:/NLP final project/used_np_false'
    # pth_src = r'D:/NLP final project\songs\whisper\whisper_dataset_np'
    # pth_basic =r'D:/NLP final project'
    df = pd.read_csv(pth)
    data = []
    for i, x in enumerate(df['billboard']):
        if x.endswith('.npy'):
            data.append({'song_name': x, 'explicit': df['explicit'][i]})
    
    c = 0
    new_df = pd.DataFrame(data)
    new_df = new_df.iloc[1:, :]
    new_df = new_df[new_df['explicit']== False]

    df_all = pd.read_csv(pth_all_csv)
    df_all = df_all.drop(['song_id', "song_type", 'popularity', 'artists', 'billboard'], axis=1)
    
    df_all_new = pd.concat([new_df, df_all])
    df_all_new = df_all_new.reset_index()
    df_all_new = df_all_new.iloc[:,1:]
    
    df_all_new.to_csv(os.path.join(pth_basic, 'df_final.csv'), index=False)


 
