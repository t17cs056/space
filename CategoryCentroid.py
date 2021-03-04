import numpy as np
import pandas as pd

class CategoryCentroid():
    def __init__(self, ds, cl):
        self.ds = ds
        self.cl = cl
        self.CentroidList = pd.DataFrame(columns=ds.columns, index=cl['Category'].unique()) #indexにカテゴリの種類を格納

    def calCentroid(self):
        for l in self.cl['Category'].unique(): #カテゴリごとの重心を求める
            self.CentroidList.loc[l] = np.mean(self.ds[self.cl['Category'] == l]) #重心の公式から、単語ごとの特徴量の平均を計算する
            #代入するときはカラム数が合致してないといけない

        return self.CentroidList
        