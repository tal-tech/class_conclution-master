import numpy as np
import json
import pandas as pd
import pickle
import jieba
import re


class EmbeddingFeature():
    def __init__(self, embedding_path, embedding_dim=200):
        self.embedding_dim = embedding_dim
        self.default_embedding = np.array(
            [0.] * embedding_dim, dtype=np.float32)
        self.w2v = pickle.load(open(embedding_path, 'rb'))

    def get_w2v(self, word):
        embedding = self.w2v.get(word, [])
        return embedding

    def get_sentence_embedding_mean(self, sentence, max_len):
        words = jieba.lcut(sentence)[:max_len]
        embedding_list = []
        for word in words:
            embedding = self.get_w2v(word)
            if len(embedding) != 0:
                embedding_list.append(embedding)
            else:
                embedding_list.append(self.default_embedding)
        if len(embedding_list) == 0:
            return self.default_embedding
        else:
            return np.array(embedding_list).mean(axis=0)

    def get_sentence_list_embedding_mean(self, text_list, max_len):
        if len(text_list) == 0:
            raise Exception("text_list is empty")
        s2w = {}
        for sentence in set(text_list):
            s2w[sentence] = self.get_sentence_embedding_mean(sentence, max_len)
        embedding_list = []
        for sentence in text_list:
            embedding_list.append(s2w[sentence])
        return np.array(embedding_list).reshape(len(embedding_list), self.embedding_dim)


class ConcludeDetector():

    def __init__(self, model_path, patterns, pattern2index_path, embedding_path):
        self.model = pickle.load(open(model_path, 'rb'))
        self.patterns = patterns
        self.pattern2index = pickle.load(open(pattern2index_path, 'rb'))
        self.emb = EmbeddingFeature(embedding_path)
        self.re_words = re.compile('讲了|总结一下|学了|今天')
        self.re_conclude = re.compile('|'.join(self.patterns))
        self.max_len = 50

    def find_pattern(self, text):
        hit = False
        for pattern in self.patterns:
            if len(re.findall(pattern, text)) != 0:
                hit = True
                break
        if hit:
            return pattern
        else:
            return 'eoo'

    def extract_line(self, text, sjt=0.8):
        f_w2v = self.emb.get_sentence_embedding_mean(text, self.max_len)
        f_words = len(self.re_words.findall(text))
        f_time = sjt
        f_pattern = self.pattern2index.get(self.find_pattern(text), 0)
        f_text = np.hstack([f_w2v, [f_time, f_pattern, f_words]])
        return f_text

    def predict(self, feature):
        y_pred = self.model.predict(feature)
        return y_pred


class DfConcludeDetector(ConcludeDetector):
    def __init__(self, model_path, pattern_path, pattern2index_path, embedding_path):
        ConcludeDetector.__init__(
            self, model_path, pattern_path, pattern2index_path, embedding_path)

    def extract_df(self, df):
        text_list = df['text'].tolist()
        sjt_list = df['时间条'].tolist()
        feature_list = []
        for text, sjt in zip(text_list, sjt_list):
            feature = self.extract_line(text, sjt)
            feature_list.append(feature)
        return np.vstack(feature_list)

    def re_detector(self, df):
        keywords = df.text.apply(lambda x: self.re_conclude.findall(str(x)))
        keywordId = keywords.apply(lambda x: False if len(x) == 0 else True)
        df['keywords'] = keywords
        df_re = df[keywordId].copy()
        df_re['时间条'] = df_re['sentence_id']/df.shape[0]
        return df_re

    def predict_df(self, df):
        feature = self.extract_df(df)
        y_pred = self.predict(feature)
        return y_pred

    def get_result(self, df):
        df_re = self.re_detector(df)
        if df_re.shape[0] == 0:
            return False, None
        y_pred = self.predict_df(df_re)
        df_model = df_re[y_pred == 1].copy()
        if df_model.shape[0]==0:
            return False,None
        return True, df_model


def init_conclude_model(model_path, patterns, pattern2index_path, embedding_path):
    return DfConcludeDetector(model_path, patterns, pattern2index_path, embedding_path)


def convert_list_to_json(item):
    uniqueKeys = set(item)
    result = []
    for i in uniqueKeys:
        result.append({'keyword': i[:20], 'word_count': item.count(i)})
    return result


def conclude_detector(text_list,df_cd):
    if len(text_list)==0:
        return []
    df = pd.DataFrame(text_list)
    df['sentence_id'] = range(1, df.shape[0]+1)
    state,df_model = df_cd.get_result(df)
    if state:
        data = df_model.apply(lambda x: {'begin_time': x['begin_time'], 'end_time': x['end_time'],
                       'sentence': x['text'],
                       'keyword_list': convert_list_to_json(x['keywords'])}, axis=1).tolist()
    else:
        data = []
    return data


if __name__ == '__main__':
    pass
