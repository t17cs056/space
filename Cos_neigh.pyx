import cython
import numpy as np
from tqdm import tqdm
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

def cos_neigh(np.ndarray[DTYPE_t, ndim=2] doc1, np.ndarray[DTYPE_t, ndim=2] doc2):
    
    cdef int i = 0, j = 0
    cdef np.ndarray[DTYPE_t, ndim=2] Cos_neigh = np.empty((doc1.shape[0], doc2.shape[0]), dtype=DTYPE)

    #Cos_neigh = np.empty((doc1.shape[0], doc2.shape[0]), dtype=np.float32)

    for i in tqdm(range(doc1.shape[0]), desc="Euclid"):
        for j in range(doc2.shape[0]):
            Cos_neigh[i][j] = np.linalg.norm(doc2[j] - doc1[i])

    return Cos_neigh
