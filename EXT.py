import pandas as pd
import numpy as np
import CategoryCentroid as CC
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from tqdm import tqdm #進捗バー表示ライブラリ
from sklearn.model_selection import StratifiedKFold, train_test_split
import random
import time
import Cos_neigh as cn
import cos_sim as cs
from bayes_opt import BayesianOptimization
import faiss
import gc
import Preprocessing

class Cent_ext():
    
    def __init__(self, ds, cl, c):
        self.ds = ds
        self.cl = cl
        self.crd = pd.DataFrame(CC.CategoryCentroid(self.ds, self.cl).calCentroid()) #TFIDF表現されたデータから求めたカテゴリ重心
        #self.Mcent = [None] * (len(self.ds.index) * len(self.crd.index)) #元の入力データのTFIDF表現を、重心とのコサイン類似度(距離)に置き換えたデータ(Cos_cent)
        self.Mcent = np.empty((len(self.ds.index), len(self.crd.index)), dtype=np.float32)
        self.clf = LinearSVC(C=c, max_iter=10000, random_state=0)
    """
    def cos_sim(self, v1, v2): #コサイン類似度を計算
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    """

    def fit(self):
        Cent_ext = np.empty((len(self.ds.index), len(self.ds.index)), dtype=np.float32)
        """
        for i in tqdm(range(self.ds_len), desc='Mcent'): #１つの文書に対して各重心とのコサイン類似度(距離)を計算(Cos_cent)
            for j in range(self.crd_len):
                #self.Mcent[i*self.crd_len + j] = self.cos_sim(self.ds.iloc[i], self.crd.iloc[j])
                self.Mcent[i][j] = self.cos_sim(self.ds.iloc[i].values, self.crd.iloc[j].values)

        #self.Mcent = np.array(self.Mcent).reshape(self.ds_len, self.crd_len)
        #Cent_ext = [None] * ((self.ds_len) ** 2)
        #Cent_ext = np.empty((self.ds_len, self.ds_len), dtype=np.float32)

        for i in tqdm(range(self.ds_len), desc='Cent_ext'):
            for j in range(self.ds_len):
                #Cent_ext[i*self.ds_len + j] = np.linalg.norm(self.Mcent[j] - self.Mcent[i]) #1つの文書から近いカテゴリに含まれている各文書とのユークリッド距離を求める(Cos_neigh)
                self.Cent_ext[i][j] = np.linalg.norm(self.Mcent[j] - self.Mcent[i])
        """
        self.Mcent = cs.cos_sim(self.ds.values, self.crd.values)
        Cent_ext = cn.cos_neigh(self.Mcent, self.Mcent)

        #Cent_ext = np.array(Cent_ext).reshape(self.ds_len, self.ds_len) #行列の形の変換
        self.clf.fit(Cent_ext, np.ravel(self.cl))

    def predict(self, target):
        #test_Mcent = [None] * self.crd_len
        #test_Cent_ext = [None] * (target_len * self.ds_len)
        test_Mcent = np.empty((len(target.index), len(self.crd.index)), dtype=np.float32)
        test_Cent_ext = np.empty((len(target.index), len(self.ds.index)), dtype=np.float32)
        """
        for i in tqdm(range(len(target.index)), desc='predict'): #ターゲット文書と各重心とのコサイン類似度を計算(Cos_cent)
            for j in range(len(self.crd.index)):
                test_Mcent[j] = self.cos_sim(target.iloc[i].values, self.crd.iloc[j].values)

            for j in range(self.ds_len):
                #test_Cent_ext[i*self.ds_len + j] = np.linalg.norm(self.Mcent[j] - test_Mcent) #各文書とのユークリッド距離を求める(Cos_neigh)
                test_Cent_ext[i][j] = np.linalg.norm(self.Mcent[j] - test_Mcent)
        """
        test_Mcent = cs.cos_sim(target.values, self.crd.values)
        test_Cent_ext = cn.cos_neigh(test_Mcent, self.Mcent)

        #test_Cent_ext = np.array(test_Cent_ext).reshape(target_len, self.ds_len)
        test_pred = self.clf.predict(test_Cent_ext)

        return test_pred

class Orig_ext():

    def __init__(self, ds, cl, c, k):
        self.ds = ds
        self.cl = cl
        self.crd = pd.DataFrame(CC.CategoryCentroid(self.ds, self.cl).calCentroid()) #TFIDF表現されたデータから求めたカテゴリ重心
        self.hwc = [LinearSVC(C=c, random_state=0, max_iter=100000) for i in range(len(self.crd.index))] #カテゴリの種類数だけSVMを宣言
        self.XMFextend = None
        self.k = k
        self.knn = faiss.IndexFlatIP(len(self.ds.columns))
        #self.Mcent = np.empty((len(self.ds.index), len(self.crd.index)), dtype=np.float32) #改良手法1

    """    
    def cos_sim(self, v1, v2): #コサイン類似度を計算
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    """
    def sigmoid_func(self, x):    # シグモイド関数の定義
        return 1 / (1 + np.exp(-x))

    def fit(self):
        
        #Cos_cent = [None] * (self.ds_len * self.crd_len) #１つの文書に対して各重心とのコサイン類似度(距離)を計算
        #Cos_neigh = [None] * (self.ds_len ** 2) #1つの文書と各カテゴリに含まれる文書とのユークリッド距離を計算
        Cos_cent = np.empty((len(self.ds.index), len(self.crd.index)), dtype=np.float32)
        #Cos_neigh = np.empty((len(self.ds.index), len(self.ds.index)), dtype=np.float32) #従来手法
        Cos_neigh = np.empty((len(self.ds.index), self.k), dtype=np.float32) #手法2
        tmpDS = self.ds.values.copy(order='C').astype(np.float32)
        faiss.normalize_L2(tmpDS)
        self.knn.add(tmpDS)

        """
        for i in tqdm(range(self.ds_len), desc='fit'): 
            ###Cos_neighを生成###
            for j in range(self.ds_len):
                #Cos_neigh[i*self.ds_len + j] = np.linalg.norm(self.ds.iloc[j] - self.ds.iloc[i]) #1つの文書から近いカテゴリに含まれている各文書とのコサイン類似度を求める(Cos_neigh)
                Cos_neigh[i][j] = self.cos_sim(self.ds.iloc[i].values, self.ds.iloc[j].values)
            
            ###Cos_centを生成###
            for j in range(self.crd_len):
                #Cos_cent[i*self.crd_len + j] = self.cos_sim(self.ds.iloc[i], self.crd.iloc[j]) #1つの文書に対して各重心とのコサイン類似度(距離)を計算(Cos_cent)
                Cos_cent[i][j] = self.cos_sim(self.ds.iloc[i].values, self.crd.iloc[j].values)
        """
        
        print('fit')
        ###Cos_neighを生成###
        Cos_cent = cs.cos_sim(self.ds.values, self.crd.values)
        
        """
        ###Cos_neighを生成###
        Cos_neigh = cs.cos_sim(self.ds.values, self.ds.values)
        """
        for i, d in enumerate(tqdm(self.ds.values, desc='cos_neigh')):
            D, I = self.knn.search(d, self.k + 1)
            D = np.delete(D, 0, 1)
            print(D)
            Cos_neigh[i] = D[0]

        ###Cos_neighを生成###
        #Cos_neigh = cn.cos_neigh(self.ds.values, self.ds.values) #1つの文書から近いカテゴリに含まれている各文書とのユークリッド距離を求める(Cos_neigh)
        #Cos_cent = np.array(Cos_cent).reshape(self.ds_len, self.crd_len) #行列の形に変換
        #Cos_neigh = np.array(Cos_neigh).reshape(self.ds_len, self.ds_len)
        self.XMFextend = np.concatenate([self.ds.values, Cos_cent, Cos_neigh], 1) #XMextendを生成 #従来手法
        """
        self.Mcent = cs.cos_sim(self.ds.values, self.crd.values) #改良手法1
        cent_ext = cn.cos_neigh(self.Mcent, self.Mcent)
        #print(cent_ext.shape)
        
        self.XMFextend = np.concatenate([self.ds.values, cent_ext], 1) #XMextendを生成 #改良手法1
        #print(self.XMFextend.shape)
        
        del cent_ext
        gc.collect()
        """

    def predict(self, target):
        test_XMFextend = None
        
        #test_Cos_cent = [None] * (target_len * self.crd_len) #１つの文書に対して各重心とのコサイン類似度(距離)を計算
        #test_Cos_neigh = [None] * (target_len * self.ds_len) #1つの文書と各カテゴリに含まれる文書とのユークリッド距離を計算
        test_Cos_cent = np.empty((len(target.index), len(self.crd.index)), dtype=np.float32)
        test_Cos_neigh = np.empty((len(target.index), len(self.ds.index)), dtype=np.float32)
        
        """
        for i in tqdm(range(target_len), desc="test"): #ターゲット文書に対して各重心とのコサイン類似度(距離)を計算(Cos_cent)
            ###Cos_neighを生成###
            for j in range(self.ds_len): 
                #test_Cos_neigh[i*self.ds_len + j] = np.linalg.norm(self.ds.iloc[j] - target.iloc[i]) #各文書とのコサイン類似度を求める(Cos_neigh)
                test_Cos_neigh[i][j] = self.cos_sim(target.iloc[i].values, self.ds.iloc[j].values)

            ###Cos_centを生成###
            for j in range(self.crd_len):      
                #test_Cos_cent[i*self.crd_len + j] = self.cos_sim(target.iloc[i], self.crd.iloc[j])
                test_Cos_cent[i][j] = self.cos_sim(target.iloc[i].values, self.crd.iloc[j].values)
        """
        
        print('test')
        ###Cos_neighを生成###
        test_Cos_cent = cs.cos_sim(target.values, self.crd.values)

        ###Cos_neighを生成###
        test_Cos_neigh = cs.cos_sim(target.values, self.ds.values)
        
        ###Cos_neighを生成###
        #test_Cos_neigh = cn.test_cos_neigh(target.values, self.ds.values) #各文書とのユークリッド距離を求める(Cos_neigh)
        #test_Cos_cent = np.array(test_Cos_cent).reshape(target_len, self.crd_len)
        #test_Cos_neigh = np.array(test_Cos_neigh).reshape(target_len, self.ds_len)
        """
        test_Mcent = cs.cos_sim(target.values, self.crd.values) #改良手法1
        test_cent_ext = cn.cos_neigh(test_Mcent, self.Mcent)
        #print(test_cent_ext.shape)
        """
        test_XMFextend = np.concatenate([target.values, test_Cos_neigh, test_Cos_cent], 1) #test_XMextendを生成 #従来手法
        #test_XMFextend = np.concatenate([target.values, test_cent_ext], 1) #test_XMextendを生成 #改良手法1
        #print(test_XMFextend.shape)
        
        orig_ext = pd.DataFrame(columns=self.crd.index, index=target.index) #予測したカテゴリを格納
        """
        del self.Mcent
        del test_Mcent
        del test_cent_ext
        gc.collect()
        """
        for i in tqdm(range(len(self.hwc)), desc='hwc fit'): #カテゴリの種類数だけ超平面を生成
            Bool_cl_train = (self.cl['Category'] == self.crd.index[i]).values #fitに与えるカテゴリをTrueとFalseの2値にする
            self.hwc[i].fit(self.XMFextend, Bool_cl_train) #カテゴリごとに学習 #二値分類

        del self.XMFextend
        gc.collect()

        for i in tqdm(range(len(self.crd.index)), desc='predict'):
            orig_ext[self.crd.index[i]] = [self.sigmoid_func(i) for i in self.hwc[i].decision_function(test_XMFextend)] #テスト文書と超平面の間の距離を取得し、シグモイド関数で正規化 #Orig_ext行列へ縦方向に代入していく

        test_pred = orig_ext.astype(np.float32).idxmax(axis=1) #ターゲット文書が持つ特徴量が最大のカテゴリを取得

        return test_pred

pre = Preprocessing.preprocesser()
pre.doc_load('search') #クラスPreprocessing内でデータを読み込み分割、単語の重要度を算出する #'search'でパラメータサーチ、'exp'でテスト
#ds_train, ds_test, cl_train, cl_test = Preprocessing.doc_load() #訓練データとテストデータに分割
"""
###パラメータサーチ始め###
params = { #パラメータサーチ用の引数
    'W': (2000, 8000),
    'C': (-5, 5), #正則化パラメータ
    #'K': (3, 10), #近傍文書の取得数
}

def ext_cv(W, C): #パラメータサーチで使う関数
    #print(W)
    #print(C)
    #print(K)
    score = 0.0
    ds, cl = pre.preprocess(int(W)) #特徴選択による単語削除
    cl.drop(index=ds[ds.isnull().any(axis=1)].index, inplace=True) #欠損値を含む文書のカテゴリ情報を削除 
    ds.dropna(inplace=True) #欠損値を含む文書(行)を削除
    sp_ds_train, sp_ds_test, sp_cl_train, sp_cl_test = train_test_split(ds, cl, test_size=0.2, train_size=0.8, shuffle=True, stratify=cl)
    #print(sp_ds_train.shape)
    #print(sp_cl_train.shape)
    ext = Orig_ext(sp_ds_train, sp_cl_train, 2**int(C))
    ext.fit()
    ext_pred = ext.predict(sp_ds_test)
    score = f1_score(sp_cl_test, ext_pred, average='micro')

    
    skf = StratifiedKFold(n_splits=5, shuffle=True) #5分割交差検証

    for train_index, test_index in skf.split(ds, cl):
        syn = SYN(ds.iloc[train_index], cl.iloc[train_index], int(K), 2**int(C))
        syn.fit()
        syn_pred = syn.predict(ds.iloc[test_index])
        score += f1_score(cl.iloc[test_index], syn_pred, average='micro')
    

    return score

start = time.time()
syn_cv_bo = BayesianOptimization(ext_cv, params, random_state=0) #ベイズ最適化を使用したパラメータサーチ
syn_cv_bo.maximize(init_points=3, n_iter=15) #パラメータサーチ実行
end = time.time()
print('max score:', syn_cv_bo.max['target'])
print('W:', int(syn_cv_bo.max['params']['W']))
#print('K:', int(syn_cv_bo.max['params']['K']))
print('C:', 2**int(syn_cv_bo.max['params']['C']))
print("search time:", (end - start)) # 実行時間計測
###パラメータサーチ終わり###
"""

MacroF1 = 0
MicroF1 = 0
c = 0.125 #正則化パラメータ
#k = 0 #k-NNにより取得する近傍文書の数
#word_num = 2000 #特徴選択で取得する上位単語の数
ds = pd.read_csv('./web_data_MI2000BoW.csv', index_col=0)
cl = pd.read_csv('./web_category.csv', index_col=0)
#ds, cl = pre.preprocess(word_num) #特徴選択による単語削除
cl.drop(index=ds[ds.isnull().any(axis=1)].index, inplace=True) #欠損値を含む文書のカテゴリ情報を削除 
ds.dropna(inplace=True) #欠損値を含む文書(行)を削除
print(ds.shape)
print(cl.shape)

skf = StratifiedKFold(n_splits=5, shuffle=True) #5分割交差検証

for train_index, test_index in skf.split(ds, cl):
    start = time.time()
    ext = Orig_ext(ds.iloc[train_index], cl.iloc[train_index], c)
    ext.fit()
    ext_pred = ext.predict(ds.iloc[test_index])
    end = time.time()
    MacroF1 += f1_score(cl.iloc[test_index], ext_pred, average='macro')
    MicroF1 += f1_score(cl.iloc[test_index], ext_pred, average='micro')
    print('-------------------')
    print("score: {}".format(f1_score(cl.iloc[test_index], ext_pred, average='micro')))
    print("time: {}".format(end - start)) # 実行時間計測
    print('-------------------')

#MacroF1とMicroF1の平均を出力する
print('macroF1: {}'.format(MacroF1 / 5))
print('microF1: {}'.format(MicroF1 / 5))

