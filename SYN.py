import pandas as pd
import numpy as np
import random
from sklearn.svm import LinearSVC #線形SVM
from sklearn.neighbors import NearestNeighbors
import CategoryCentroid as CC#モジュール名にはファイル名を書く
from sklearn.model_selection import StratifiedKFold, train_test_split # StratifiedKFold は5分割交差検証, train_test_split は訓練セットとテストセットに分割するとき使用
from tqdm import tqdm #進捗バー表示ライブラリ
from sklearn.metrics import f1_score
import faiss #k-NN法で使用
import time #実行時間計測に使用
from bayes_opt import BayesianOptimization #パラメータサーチのベイズ最適化
import gc
import synthetic as sy #SDR生成プログラム
import Preprocessing #特徴選択等の前処理プログラム

class SYN():

    def __init__(self, ds_train, cl_train, k, c, a): #重み付け手法
    #def __init__(self, ds_train, cl_train, k, c): #従来手法
        self.ds_train = ds_train #訓練文書
        self.cl_train = cl_train #訓練文書に割り当てられている分野
        self.crd = CC.CategoryCentroid(self.ds_train, self.cl_train).calCentroid() #分野ごとの重心を求める
        self.k = k #k-NN法で対象文書の近くから取得する文書数
        self.a = a #対象文書のみに出現する単語の重要度にかける重み #本手法
        self.hwc = [LinearSVC(C=c, random_state=0, max_iter=500000) for i in range(len(self.crd.index))] #SVMの定義 #分野数分を定義する
        self.col = list(ds_train.columns) #SDRのcolumn
        self.col.append('cos_sim')
        self.col.extend(list(self.crd.index))
        self.knn_pos = faiss.IndexFlatIP(len(self.ds_train.columns)) #k-NN法の定義
        self.knn_neg = faiss.IndexFlatIP(len(self.ds_train.columns)) #k-NN法の定義
        """
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        self.knn_pos = faiss.GPUIndexFlatIP(res, len(self.ds_train.columns), flat_config)
        self.knn_neg = faiss.GPUIndexFlatIP(res, len(self.ds_train.columns), flat_config)
        """

    def sigmoid_func(self, x):    # シグモイド関数の定義
        return 1 / (1 + np.exp(-x))
        
    def fit(self):

        for l in tqdm(self.crd.index, desc='train'):
            index = 0 #Spos, Snegの参照に使う変数
            Dpos = self.ds_train[self.cl_train['Category'] == l].values #指定した分野に属する文書集合を取得
            Dneg = self.ds_train[self.cl_train['Category'] != l].values #指定した分野に属さない文書集合を取得
            Spos = np.empty((Dpos.shape[0]*self.k, len(self.col)), dtype=np.float32) #分野が同じ(positive)SDRを格納 #従来手法
            Sneg = np.empty((len(self.ds_train)*self.k, len(self.col)), dtype=np.float32) #分野が異なる(negative)SDRを格納
            tmpDpos = Dpos.copy(order='C').astype(np.float32) #配列をC連続？しないと正規化できない #float64だとエラーが出るからfloat32に変換 #Dposをコピー
            faiss.normalize_L2(tmpDpos) #DposデータをL2正規化
            self.knn_pos.add(tmpDpos) #Dposデータでk-NN学習
            tmpDneg = Dneg.copy(order='C').astype(np.float32) #Dnegをコピー
            faiss.normalize_L2(tmpDneg) #DnegデータをL2正規化
            self.knn_neg.add(tmpDneg) #Dnegデータでk-NN学習

            for i in tqdm(range(len(self.ds_train.index)), desc='SDRs'): #ds_train.indexが文書番号 #文書名に重複がある場合の対策としてインデックス番号を使用
                target = self.ds_train.iloc[i].values #訓練セット中の対象文書
                tmptarget = target.copy(order='C').astype(np.float32).reshape(1, len(self.ds_train.columns))
                faiss.normalize_L2(tmptarget)

                if self.cl_train.iat[i, 0] == l: #対象文書に割り当てられている分野と同じ分野の文書集合に対してif文内を処理
                    D, I = self.knn_pos.search(tmptarget, self.k + 1) #対象文書の近くの文書をインデックス番号でk個取得
                    I = np.delete(I, 0, 1) #対象文書自身のインデックス番号が入っているから削除
                    D = np.delete(D, 0, 1) #対象文書自身のインデックス番号が入っているから削除

                    neighbor = sy.Synthetic(target, Dpos, self.crd.values, I[0], self.a) #重み付け手法 #対象文書1つに対してk個のSDRが返ってくる
                    #neighbor = sy.Synthetic(target, Dpos, self.crd.values, I[0]) #従来手法
                    
                    for j, n in enumerate(neighbor):
                        Spos[index*len(neighbor) + j] = n #SDRを追加
                    
                    index += 1

                D, I = self.knn_neg.search(tmptarget, self.k + 1) #対象文書の近くの文書をインデックス番号でN個取得
                I = np.delete(I, 0, 1)
                D = np.delete(D, 0, 1) #手法1 + 改良手法2 + 係数a
                
                neighbor = sy.Synthetic(target, Dneg, self.crd.values, I[0], self.a) #重み付け手法 #対象文書1つに対してk個のSDRが返ってくる
                #neighbor = sy.Synthetic(target, Dneg, self.crd.values, I[0]) #従来手法

                for j, n in enumerate(neighbor):
                    Sneg[i*len(neighbor) + j] = n #SDRを追加

            #同じ分野には1、異なる分野には0のSDRのラベルがつく    
            Labelpos = np.ones(Spos.shape[0], dtype=np.int8) #SDRpositiveのラベルを生成
            Labelneg = np.zeros(Sneg.shape[0], dtype=np.int8) #SDRnegativeのラベルを生成
            SDRs = np.concatenate([Spos, Sneg], 0) #縦方向に結合
            SDRsLabel = np.concatenate([Labelpos, Labelneg], 0)

            del Dpos #メモリ不足による処理落ちを防ぐため
            del Dneg
            del Spos
            del Sneg
            del tmpDneg
            del tmpDpos
            del D
            del I
            del Labelpos
            del Labelneg
            del neighbor
            gc.collect()
            
            self.hwc[self.crd.index.get_loc(l)].fit(SDRs, SDRsLabel) #SVMによる学習 #超平面の取得
            self.knn_pos.reset() #Dposの学習を消す
            self.knn_neg.reset() #Dnegの学習を消す

    def predict(self, ds_test, num):
        m = pd.DataFrame(columns=self.crd.index, index=ds_test.index) #すべての分野の超平面に対して計算した距離を保持

        for i in tqdm(range(len(self.crd.index)), desc='test'):
            Dpos = self.ds_train[self.cl_train['Category'] == self.crd.index[i]].values #指定した分野に属する文書集合を取得
            h = np.empty((len(ds_test.index), self.k), dtype=np.float32)
            Spos = np.empty((len(ds_test.index)*self.k, len(self.col)), dtype=np.float32)
            tmpDpos = Dpos.copy(order='C').astype(np.float32)
            faiss.normalize_L2(tmpDpos)
            self.knn_pos.add(tmpDpos)

            for j in range(len(ds_test.index)):
                target = ds_test.iloc[j].values #テストセット中の対象文書
                tmptarget = target.copy(order='C').astype(np.float32).reshape(1, len(ds_test.columns))
                faiss.normalize_L2(tmptarget)
                S = np.empty((self.k, len(self.col)), dtype=np.float64) #対象のテスト文書のSDRを格納しておく配列 #従来手法
                D, I = self.knn_pos.search(tmptarget, self.k + 1) #対象のテスト文書の近くの文書をインデックス番号でN個取得
                I = np.delete(I, 0, 1)
                D = np.delete(D, 0, 1)

                neighbor = sy.Synthetic(target, Dpos, self.crd.values, I[0], self.a) #重み付け手法 #テスト文書1つに対してk個のSDRが返ってくる
                #neighbor = sy.Synthetic(target, Dpos, self.crd.values, I[0]) #従来手法

                for k, n in enumerate(neighbor):
                    S[k] = n
                    Spos[j * self.k + k] = n
                
                #print(Spos)
                H = [self.sigmoid_func(i) for i in self.hwc[i].decision_function(S)] #対象のテスト文書から生成したSDRと超平面との距離を計測 #計測した距離をシグモイド関数で正規化
                h[j] = H
                #H.sort() #求めた距離を昇順にソート
                #m.iat[j, i] = max(H) #求めた距離の中から最大値を格納 #従来手法
                m.iat[j, i] = np.average(H) #求めた距離の平均値を格納 #改良手法

            
            Spos = pd.DataFrame(Spos, columns=self.col)
            Spos.to_csv('./testSDR_CHI_{}_{}.csv'.format(self.crd.index[i], num))
            h = pd.DataFrame(h, index=ds_test.index)
            h.to_csv('./testSDR_dis_CHI_{}_{}.csv'.format(self.crd.index[i], num))
            

            self.knn_pos.reset()

        #m.to_csv('./distance_CHI_{}.csv'.format(num))
        #m = m.rank(axis=1, ascending=False)
        #m.to_csv('./rank_CHI_{}.csv'.format(num)) #分野ごとの超平面とテスト文書のSDR間の最大距離を格納してあるデータをファイルに出力
        test_pred = m.astype(np.float64).idxmax(axis=1) #テスト文書ごとに距離が最大である分野を取得 #idxmax関数は数値型ではないと使えない

        return test_pred

pre = Preprocessing.preprocesser()
pre.doc_load('exp') #クラスPreprocessing内でデータを読み込み分割、単語の重要度を算出する #'search'でパラメータサーチ、'exp'でテスト
"""
###パラメータサーチ始め###
params = { #パラメータサーチ用の引数
    'W': (2000, 8000),
    'C': (-5, 5), #正則化パラメータ
    'K': (3, 10), #近傍文書の取得数
    'a': (0, 1), #重み付け係数
}
"""
def syn_cv(W, C, K, a): #パラメータサーチで使う関数
    score = 0.0
    ds, cl = pre.preprocess(int(W)) #特徴選択による単語削除
    cl.drop(index=ds[ds.isnull().any(axis=1)].index, inplace=True) #欠損値を含む文書の分野情報を削除 
    ds.dropna(inplace=True) #欠損値を含む文書(行)を削除
    sp_ds_train, sp_ds_test, sp_cl_train, sp_cl_test = train_test_split(ds, cl, test_size=0.2, train_size=0.8, shuffle=True, stratify=cl)
    syn = SYN(sp_ds_train, sp_cl_train, int(K), 2**int(C), a) #重み付け手法
    #syn = SYN(sp_ds_train, sp_cl_train, int(K), 2**int(C)) #従来手法
    syn.fit()
    syn_pred = syn.predict(sp_ds_test)
    score = f1_score(sp_cl_test, syn_pred, average='micro') #MicroF1スコアを計算

    """
    skf = StratifiedKFold(n_splits=5, shuffle=True) #5分割交差検証

    for train_index, test_index in skf.split(ds, cl):
        syn = SYN(ds.iloc[train_index], cl.iloc[train_index], int(K), 2**int(C))
        syn.fit()
        syn_pred = syn.predict(ds.iloc[test_index])
        score += f1_score(cl.iloc[test_index], syn_pred, average='micro')
    """

    return score
"""
start = time.time()
syn_cv_bo = BayesianOptimization(syn_cv, params, random_state=0) #ベイズ最適化を使用したパラメータサーチ
syn_cv_bo.maximize(init_points=3, n_iter=15) #パラメータサーチ実行
end = time.time()
print('max score:', syn_cv_bo.max['target'])
print('W:', int(syn_cv_bo.max['params']['W']))
print('K:', int(syn_cv_bo.max['params']['K']))
print('C:', 2**int(syn_cv_bo.max['params']['C']))
print('a:', syn_cv_bo.max['params']['a'])
print("search time:", (end - start)) # 実行時間計測
###パラメータサーチ終わり###
"""

MacroF1 = 0
MicroF1 = 0
c = 0.0625 #正則化パラメータ
k = 3 #k-NNにより取得する近傍文書の数
word_num = 5431 #特徴選択で取得する上位単語の数
a = 0.2334 #重み付け係数
ds, cl = pre.preprocess(word_num) #特徴選択による単語削除
cl.drop(index=ds[ds.isnull().any(axis=1)].index, inplace=True) #欠損値を含む文書の分野情報を削除
ds.dropna(inplace=True) #欠損値を含む文書(行)を削除
num = 0

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0) #5分割交差検証

start = time.time()

for train_index, test_index in skf.split(ds, cl): #訓練データとテストデータに分割 #5回ループ
    num += 1
    syn = SYN(ds.iloc[train_index], cl.iloc[train_index], k, c, a) #重み付け手法
    #syn = SYN(ds.iloc[train_index], cl.iloc[train_index], k, c) #従来手法
    syn.fit()
    syn_pred = syn.predict(ds.iloc[test_index], num)
    #cl.iloc[test_index].to_csv('./true_{}.csv'.format(num)) #5分割交差検証により分割された正解ラベルをファイルに出力
    MacroF1 += f1_score(cl.iloc[test_index], syn_pred, average='macro') #MacroF1スコアを計算
    MicroF1 += f1_score(cl.iloc[test_index], syn_pred, average='micro') #MicroF1スコアを計算
    print('-------------------')
    print("MacroF1 score: {}".format(f1_score(cl.iloc[test_index], syn_pred, average='macro')))
    print("MicroF1 score: {}".format(f1_score(cl.iloc[test_index], syn_pred, average='micro')))
    print('-------------------')

end = time.time()

#MacroF1とMicroF1の平均を出力する
print('MacroF1 ave: {}'.format(MacroF1 / 5)) #MacroF1スコアの平均を出力
print('MicroF1 ave: {}'.format(MicroF1 / 5)) #MicroF1スコアの平均を出力
print('time: {}'.format(end - start))

