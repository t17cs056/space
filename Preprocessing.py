from nltk import tokenize, stem
import glob
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif, chi2 #特徴選択手法のMI (相互情報量) , CHI(カイ2乗統計量) のライブラリ
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
import os
import csv #csvファイルを読み書きするライブラリ
from tqdm import tqdm #進捗バー表示ライブラリ
from info_gain import info_gain #IG (情報利得) ライブラリ
from operator import itemgetter
from sklearn.model_selection import train_test_split #訓練セットとテストセットに分割するために利用

class preprocesser():

    def __init__(self):
        self.ds = None
        self.cl = None
        self.ds_train = None
        self.cl_train = None
        self.res = None
        self.boolean = None #パラメータサーチを行うか、実験を行うかの処理を決定する

    def doc_load(self, boolean):

        documents = [] #文書リスト
        stemmer = stem.PorterStemmer() 
        filenamelist = [] #ファイル名リスト
        labellist = [] #分野リスト
        self.boolean = boolean #パラメータサーチを行うか、実験を行うかの処理を決定する

        #ストップワードリストを読み込む
        with open('StopWordList', 'r') as f:
            stopwords = [s.strip() for s in f.readlines() if s != '\n']

        #データセットの文書を読み込む
        for l in tqdm(glob.glob('/home/watanabe/study/webkb/*/*'), desc='documents load'):
            with open(l, 'r', encoding='charmap') as f: #データセットごとに適切なencodingを設定する #適切でないと文字化けする
                texts = f.read()
                filenamelist.append(os.path.basename(l)) #ファイル名取得
                labellist.append(os.path.basename(os.path.dirname(l))) #直上のディレクトリ名取得 #basenameがファイル名、dirnameがフォルダ名を取得
                words = tokenize.wordpunct_tokenize(texts) #単語ごとに分ける
                words = [stemmer.stem(Word.lower()) for Word in words if Word.isalpha() and stemmer.stem(Word.lower()) not in stopwords] #記号とストップワードを除く #大文字を小文字に直す #単語の語幹化
                documents.append(' '.join(words)) #文書に含まれてる単語群をひとまとまりの要素として、リストに追加 #documentsの要素が文書となっている

        cv = CountVectorizer()
        x = cv.fit_transform(documents) #文書内に出現する単語をカウント
        self.ds = pd.DataFrame(x.toarray(), columns=cv.get_feature_names(), index=filenamelist)
        self.cl = pd.DataFrame(labellist, columns=['Category'], index=filenamelist) #データセットの分野
        self.ds_train, ds_test, self.cl_train, cl_test = train_test_split(self.ds, self.cl, test_size=0.2, train_size=0.8, shuffle=True, random_state=0, stratify=self.cl) #random_stateを固定しているので分け方は毎回同じ

        if self.boolean == 'search': #パラメータサーチ用
            #df = np.count_nonzero(self.ds, axis=0) #文書頻度による特徴選択
            mi = mutual_info_classif(x, labellist) #相互情報量(MI)による特徴選択
            #chi2_, pval = chi2(x, labellist) #カイ2乗統計量(CHI)による特徴選択
            """
            ###情報利得(IG)による特徴選択###
            ig = np.empty(len(cv.get_feature_names()), dtype=np.float64)

            for i, c in enumerate(tqdm(cv.get_feature_names(), desc='IG')):
                #res.setdefault(c, info_gain.info_gain(df_cv[c], label_df['Category']))
                ig[i] = info_gain.info_gain(self.ds[c], self.cl['Category']) #各文書における単語cの出現回数とカテゴリからIGを計算
            """
            self.res = dict(zip(cv.get_feature_names(), mi)) #各単語の重要度を辞書型で保存

        elif self.boolean == 'exp': #実験用
            #df = np.count_nonzero(self.ds, axis=0) #文書頻度による特徴選択
            #mi = mutual_info_classif(x, labellist) #相互情報量(MI)による特徴選択
            chi2_, pval = chi2(x, labellist) #カイ2乗統計量(CHI)による特徴選択
            """
            ###情報利得(IG)による特徴選択###
            ig = np.empty(len(cv.get_feature_names()), dtype=np.float64)

            for i, c in enumerate(tqdm(cv.get_feature_names(), desc='IG')):
                #res.setdefault(c, info_gain.info_gain(df_cv[c], label_df['Category']))
                ig[i] = info_gain.info_gain(self.ds[c], self.cl['Category']) #各文書における単語cの出現回数とカテゴリからIGを計算
            """
            self.res = dict(zip(cv.get_feature_names(), chi2_)) #各単語の重要度を辞書型で保存

    def preprocess(self, word_num): #単語削減処理
        
        if self.boolean == 'search': #パラメータサーチ用
            res_sorted = sorted(self.res.items(), key=itemgetter(1), reverse=True)[0:word_num] #特徴選択において上位X語を抽出
            keylist = [v[0] for v in res_sorted] #抽出した上位X語のキーをリストにする
            df_cv = self.ds_train[keylist] #訓練セットから上位X語の単語を抽出
            df_binary = (df_cv > 0) * 1 #文書表現をbag of wordsで表現する  #()内は真理値が返ってくる
            """
            for w in tqdm(list(self.res.keys()), desc='representation'):
                df_cv[w] *= res_sorted[w] #各文書の単語の出現回数と特徴選択で得られた単語の重要度をかけて文書を表現する
            """
            return df_binary, self.cl_train
            #return df_cv, self.cl_train

        elif self.boolean == 'exp': #実験用
            res_sorted = sorted(self.res.items(), key=itemgetter(1), reverse=True)[0:word_num] #特徴選択において上位X語を抽出
            keylist = [v[0] for v in res_sorted] #抽出した上位X語のキーをリストにする
            df_cv = self.ds[keylist] #データセットから上位X語の単語を抽出
            df_binary = (df_cv > 0) * 1 #文書表現をbag of wordsで表現する  #()内は真理値が返ってくる
            """
            for w in tqdm(list(self.res.keys()), desc='representation'):
                df_cv[w] *= res_sorted[w] #各文書の単語の出現回数と特徴選択で得られた単語の重要度をかけて文書を表現する
            """
            return df_binary, self.cl
            #return df_cv, self.cl

"""
###tfidfを利用した特徴選択###
tfidf = TfidfVectorizer() 
y = tfidf.fit_transform(documents)
df_tfidf = pd.DataFrame(y.toarray(), columns=tfidf.get_feature_names(), index=filenamelist)
tfidf_sorted = df_tfidf.sum().sort_values(ascending=False)[0:4000] #上位X語を取得
#print(tfidf_sorted)
res = tfidf_sorted.to_dict()
#print(res)
df_cv = df_cv[list(res.keys())]
    
for w in tqdm(list(res.keys()), desc='representation'):
    df_cv[w] *= res[w] #各文書の単語の出現回数と特徴選択で得られた単語の重要度をかける

#print(df_cv)
"""
"""
pre = preprocesser()
pre.doc_load('exp')

ds, cl = pre.preprocess(5431)
#print(type(ds))

ds.to_csv('./web_data_CHI.csv')
cl.to_csv('./web_category_CHI.csv')
"""