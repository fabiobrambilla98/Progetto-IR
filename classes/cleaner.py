from typing import List
import numpy as np
import pandas as pd
from gensim.utils import simple_preprocess
from gensim.parsing.porter import PorterStemmer
import re
import contractions
import text_hammer as th
from tqdm import tqdm


class Cleaner:
    def __init__(self):
        self.tokenized_docs = []
        self.porter_stemmer = PorterStemmer()

    def __remove_contractions(self, doc):
        doc = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b',
                     '', doc, flags=re.MULTILINE)
        expanded_words = []
        for word in doc.split():
            fixed_word = contractions.fix(word)
            expanded_words.append(fixed_word)

        return ' '.join(expanded_words)

    def clean_and_tokenize_docs(self, docs: List):
        for doc in docs:
            cleaned_doc = self.__remove_contractions(doc)
            cleaned_doc = simple_preprocess(cleaned_doc)
            cleaned_doc = [self.porter_stemmer.stem(
                word) for word in cleaned_doc]
            self.tokenized_docs.append(cleaned_doc)
        return self.tokenized_docs

    def text_preprocessing(self, df, col_name, new_col):
        column = col_name
        df[new_col] = tqdm(df[column].apply(lambda x: str(x).lower()))
        df[new_col] = tqdm(df[new_col].apply(lambda x: th.cont_exp(x)))
        df[new_col] = tqdm(df[new_col].apply(lambda x: th.remove_emails(x)))
        df[new_col] = tqdm(df[new_col].apply(lambda x: th.remove_html_tags(x)))
        df[new_col] = tqdm(df[new_col].apply(
            lambda x: th.remove_special_chars(x)))
        df[new_col] = tqdm(df[new_col].apply(
            lambda x: th.remove_accented_chars(x)))
        df[new_col] = tqdm(df[new_col].apply(lambda x: th.make_base(x)))
        return (df)
