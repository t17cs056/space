import cython
import numpy as np
from tqdm import tqdm
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

def cos_sim(np.ndarray doc1, np.ndarray doc2):

    cdef int i = 0, j = 0
    cdef np.ndarray[DTYPE_t, ndim=2] Cos_Sim = np.empty((doc1.shape[0], doc2.shape[0]), dtype=DTYPE)

    for i in tqdm(range(doc1.shape[0]), desc="cos_sim"):
        for j in range(doc2.shape[0]):
            Cos_Sim[i][j] = np.dot(doc1[i], doc2[j]) / (np.linalg.norm(doc1[i]) * np.linalg.norm(doc2[j])) #dotは内積計算

    return Cos_Sim