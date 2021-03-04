import cython #cython言語ライブラリ
import numpy as np
cimport numpy as np #cython用ライブラリ

DTYPE = np.float #グローバルに型を定義
ctypedef np.float_t DTYPE_t #グローバルに型を定義

cdef double cos_sim(np.ndarray v1, np.ndarray v2): #コサイン類似度を計算
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)) #dotは内積計算

def Synthetic(np.ndarray[np.long_t, ndim=1] target, np.ndarray[np.long_t, ndim=2] D, np.ndarray[object, ndim=2] crd, np.ndarray[np.int_t, ndim=1] N, DTYPE_t a): #重み付け手法
#def Synthetic(np.ndarray[np.long_t, ndim=1] target, np.ndarray[np.long_t, ndim=2] D, np.ndarray[object, ndim=2] crd, np.ndarray[np.int_t, ndim=1] N): #targetは対象文書、Dはデータセットの一部の集合、Nはtargetの近傍文書のインデックス番号を格納(インデックス番号はデータセット内の番号) #従来手法

    #c言語と同様の型宣言を行う
    cdef int i = 0, n = 0, j = 0 
    cdef double cos_target = 0.0, cos_neighbor = 0.0, t = 0.0
    cdef np.ndarray[DTYPE_t, ndim=2] SDRs = np.zeros((len(N), D.shape[1] + 1 + crd.shape[0]), dtype=DTYPE) #近傍文書分のSDRを生成 #従来手法
    cdef np.ndarray[object, ndim=1] c
    cdef np.ndarray[DTYPE_t, ndim=1] syn_vec = np.zeros(D.shape[1], dtype=DTYPE) #合成した文書ベクトルを一時的に保存しておく変数 #改良手法2
    
    for i, n in enumerate(N): #従来手法
        
        for j, t in enumerate(target):
            """
            SDRs[i, j] = t + D[n, j] #IG, MI, X2、DFのマージはこの式を使う #加算手法
            #syn_vec[j] = t + D[n, j] #改良手法2
            """
            
            if t == 0: #合成先に重み付け #重み付け手法
                SDRs[i, j] = t + (D[n, j] * a) 
                syn_vec[j] = t + (D[n, j] * a) 
            else:
                SDRs[i, j] = t + D[n, j] 
                syn_vec[j] = t + D[n, j] 
            
            """
            if D[n, j] == 0: #対象文書に重み付け #重み付け手法
                SDRs[i, j] = (t * a) + D[n, j]
                #syn_vec[j] = (t * a) + D[n, j]
            else:
                SDRs[i, j] = t + D[n, j]
                #syn_vec[j] = t + D[n, j]
            """
            """
            if t <= D[n, j]: #ベクトルの各要素同士を比較 #従来手法
                SDRs[i, j] = t
                #syn_vec[j] = t #改良手法2
            else:
                SDRs[i, j] = D[n, j]
                #syn_vec[j] = D[n, j] #改良手法2
            """
        
        SDRs[i, len(D[n])] = cos_sim(target, D[n]) #従来手法
        
        for j, c in enumerate(crd): 
            """
            cos_target = cos_sim(target, c) #従来手法
            cos_neighbor = cos_sim(D[n], c)
            
            if cos_target <= cos_neighbor: #ペアである各文書とカテゴリの重心とのコサイン類似度を比較 #従来手法
                SDRs[i, len(target) + (j+1)] = cos_target
            else:
                SDRs[i, len(target) + (j+1)] = cos_neighbor
            """

            SDRs[i, len(D[n]) + (j + 1)] = cos_sim(syn_vec, c) #合成した文書ベクトルと各カテゴリの重心のコサイン類似度を計算 #改良手法2

    return SDRs
