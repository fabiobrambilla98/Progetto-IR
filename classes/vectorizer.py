import numpy as np
from gensim.models import Word2Vec
import time
import pickle
from keras.preprocessing.text import Tokenizer
from keras.utils.data_utils import pad_sequences


class Vectorizer:
    def __init__(self):
        with open('/Users/fabio/Documents/Progetto IR/models/tokenizer', 'rb') as w:
            self.tokenizer = pickle.load(w)

    def create_w2v(self, tokens, output_folder="models/", size=1000,
                   window=3,
                   min_count=1,
                   workers=3,
                   sg=1):
        out_file = output_folder + "w2v_" + str(size) + ".model"
        start_time = time.time()
        self.w2v_model = Word2Vec(
            tokens, min_count=min_count, vector_size=size, workers=workers, window=window, sg=sg)
        print("Time taken to train word2vec model: " +
              str(time.time() - start_time))
        self.w2v_model.save(out_file)
        return self.w2v_model

    def load_w2v(self, folder):
        self.w2v_model = Word2Vec.load(folder)
        return self.w2v_model

    def vectorize_w2v_docs(self, docs):
        vectorized_docs = []
        for doc in docs:
            mean = (np.mean([self.w2v_model.wv[token]
                    for token in doc if token in self.w2v_model.wv], axis=0)).tolist()
            vectorized_docs.append(mean)
        return vectorized_docs

    def vectorize_keras_df(self, df, col_name, max_len=300):
        X = self.tokenizer.texts_to_sequences(df[col_name])
        return pad_sequences(X, maxlen=max_len, padding='post')
