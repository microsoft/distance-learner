import os
import time
import multiprocessing
from functools import partial


import faiss
import numpy as np
import torch

from tqdm import tqdm
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


class FaissKNeighbors:
    def __init__(self, k=5):
        self.index = None
        self.y = None
        self.k = k

    def fit(self, X, y=None):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))
        self.res = faiss.StandardGpuResources()

        self.gpu_index = faiss.index_cpu_to_gpu(
            self.res,
            0,
            self.index
        )
        self.y = y

    def predict(self, X):
        distances, indices = self.gpu_index.search(X.astype(np.float32), k=self.k)
        votes = None
        if self.y is not None:
            votes = self.y[indices]
            predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return indices, distances


# basic params

k = 50
n = 500
N = 510000
num_neg = 10000
num_pos = N - num_neg
r = 1.0
max_norm = 0.1
debug = True

k_neighbors = 50

test_on = 1000

print("k: {}".format(k))
print("n: {}".format(n))
print("N: {}".format(N))
print("num_neg: {}".format(num_neg))
print("r: {}".format(r))
print("max_norm: {}".format(max_norm))
print("k_nbhrs: {}".format(k_neighbors))

# making the sphere

points_k = np.random.normal(0, 1, size=(num_pos, k))
points_k = r * (points_k / np.linalg.norm(points_k, ord=2, axis=1).reshape(-1, 1))

points_n = np.zeros((N, n))
points_n[num_neg:, :k] = points_k

poca_idx = np.zeros(num_neg, dtype=np.int64) # point-of-closest-approach -- poca

poca_idx[:num_pos] = np.arange(min(num_pos, num_neg), dtype=np.int64)
poca_idx[num_pos:] = np.random.choice(np.arange(num_pos), size=max(0, num_neg - num_pos), replace=True).astype(np.int64)
if debug:
    if num_neg > num_pos:
        pass
    else:
        poca_idx = np.random.choice(np.arange(num_pos), size=num_neg, replace=False).astype(np.int64)
poca = points_n[num_neg:][poca_idx] # points of closest approach


knn = FaissKNeighbors(k=k_neighbors + 1)
knn.fit(points_n[num_neg:])

# nbhrs, dists = knn.predict()
    
var_thresh = 0.99 # threshold of explained variance for selecting normal directions

indices, dists = knn.predict(poca)

# assert (indices[:, 0] == poca_idx).all()
# dist_quantiles = np.zeros(num_neg)
# for i in range(num_neg):
#     dist_quantiles[i] = np.quantile(dists[i][indices[i] != poca_idx[i]], 0.25)
# debug_prturb_size = np.mean(dist_quantiles)
debug_prturb_size = 1e-3
print("debug on-mfld prturbation size = {}".format(debug_prturb_size))

start_computing_new_poca = time.perf_counter()

off_mfld_pts = np.zeros((num_neg, n))
off_mfld_dists = np.zeros(num_neg)

new_poca = np.zeros((num_neg, n))
new_poca_prturb_sizes = np.zeros(num_neg)

# def make_off_mfld_eg(idx, points_n, indices):

def make_perturbed_poca(idx):
    
    global points_n
    global poca
    global poca_idx
    global indices
    global dists
    global num_neg
    global num_pos
    global debug
    
    prturb_size = 0
    
    if idx < num_pos and not debug:
        # one copy of poca should be unperturbed
        return (idx, poca[idx], prturb_size)
    
    on_mfld_pt = poca[idx]
    nbhr_indices = indices[idx]
    if poca_idx[idx] in nbhr_indices:
        nbhr_indices = nbhr_indices[nbhr_indices != poca_idx[idx]]
    else:
        nbhr_indices = nbhr_indices[:-1]

    nbhrs = points_n[num_neg:][nbhr_indices]
    nbhr_local_coords = nbhrs - on_mfld_pt
    nbhr_dists = dists[idx]
    
    pca = PCA()
    pca.fit(nbhr_local_coords)
    expl_var = pca.explained_variance_ratio_
    cum_expl_var = np.cumsum(expl_var)
    tmp = np.where(cum_expl_var > var_thresh)[0][0] + 1
    normal_dirs = pca.components_[tmp:] # picking components that explain (1 - var_thresh)
    tangential_dirs = pca.components_[:tmp]
    
    rdm_coeffs = np.random.normal(0, 1, size=tangential_dirs.shape[0])
    delta = np.sum(rdm_coeffs.reshape(-1, 1) * tangential_dirs, axis=0)
    
    prturb_max = np.sqrt(max(nbhr_dists)) # faiss returns square of L2 norm of difference
    # prturb_size = np.random.uniform(0, 1) * prturb_max
    # for debugging
    prturb_size = debug_prturb_size
    delta = (prturb_size / np.linalg.norm(delta, ord=2)) * delta
    
    prturb_poca = on_mfld_pt + delta
    return (idx, prturb_poca, prturb_size)
    
with multiprocessing.Pool(processes=24) as pool:
    results = pool.map(make_perturbed_poca, range(num_neg))
    
for i in range(num_neg):
    new_poca[i] = results[i][1]
    new_poca_prturb_sizes[i] = results[i][2]

end_computing_new_poca = time.perf_counter()

print("time to compute new poca: {} seconds".format(end_computing_new_poca - start_computing_new_poca))

start_knn_for_new_poca = time.perf_counter()

new_indices, new_dists = knn.predict(new_poca)

end_knn_for_new_poca = time.perf_counter()

print("time to compute knn of new poca: {} seconds".format(end_knn_for_new_poca - start_knn_for_new_poca))

start_off_mfld = time.perf_counter()

def make_off_mfld_eg(idx):
    
    global new_poca
    global points_n
    global poca_idx
    global new_indices
    global num_neg
    global num_pos   
    
    on_mfld_pt = new_poca[idx]
    nbhr_indices = new_indices[idx]
    if idx < num_pos and poca_idx[idx] in nbhr_indices:
        nbhr_indices = nbhr_indices[nbhr_indices != poca_idx[idx]]
    else:
        nbhr_indices = nbhr_indices[:-1]
        
    nbhrs = points_n[num_neg:][nbhr_indices]
    nbhr_local_coords = nbhrs - on_mfld_pt
    
    pca = PCA()
    pca.fit(nbhr_local_coords)
    expl_var = pca.explained_variance_ratio_
    cum_expl_var = np.cumsum(expl_var)
    tmp = np.where(cum_expl_var > var_thresh)[0][0] + 1
    normal_dirs = pca.components_[tmp:] # picking components that explain (1 - var_thresh)
        
    rdm_coeffs = np.random.normal(0, 1, size=normal_dirs.shape[0])
    off_mfld_pt = np.sum(rdm_coeffs.reshape(-1, 1) * normal_dirs, axis=0)
    rdm_norm = np.random.uniform(0, max_norm)
    off_mfld_pt = off_mfld_pt * (rdm_norm / np.linalg.norm(off_mfld_pt))
    off_mfld_pt += on_mfld_pt
    
    return (idx, off_mfld_pt, rdm_norm)

with multiprocessing.Pool(processes=24) as pool:
    results = pool.map(make_off_mfld_eg, range(num_neg))


    
for i in range(len(results)):
    off_mfld_pts[i] = results[i][1]
    off_mfld_dists[i] = results[i][2]

end_off_mfld = time.perf_counter()

print("time to compute off mfld eg: {} seconds".format(end_off_mfld - start_off_mfld))


rdm_idx = np.random.choice(np.arange(num_neg), test_on, replace=False)

min_true_dists = np.zeros(test_on)
true_on_mfld_poca = np.zeros((test_on, n))
dev_frm_on_mfld_poca = np.zeros(test_on)

for idx in tqdm(range(rdm_idx.shape[0])):    
    true_off_mfld_dists = np.linalg.norm(off_mfld_pts[rdm_idx[idx]] - points_n[num_neg:,:], ord=2, axis=1)
    min_true_dist = np.min(true_off_mfld_dists)
    min_true_dists[idx] = min_true_dist
    min_true_dist_idx = np.argmin(true_off_mfld_dists)
    dev_frm_on_mfld_poca[idx] = np.linalg.norm(points_n[num_neg:][min_true_dist_idx] - new_poca[rdm_idx[idx]], ord=2)
    true_on_mfld_poca[idx] = points_n[num_neg:][min_true_dist_idx]

plt.figure()
plt.hist(np.abs(min_true_dists - off_mfld_dists[rdm_idx]))
plt.xlabel("abs. error between true and approx. distance")
plt.ylabel("count")
plt.title("error in distance for k={},n={} ({} random samples)".format(k, n, test_on))
plt.savefig("l2_err_trueVapprox_k{}n{}_knn{}_N{}_num_neg{}_dps{}.png".format(k, n, k_neighbors, N, num_neg, debug_prturb_size))


plt.figure()
plt.hist(new_poca_prturb_sizes)
plt.savefig("new_poca_prturb_sizes_k{}n{}_knn{}_N{}_num_neg{}_dps{}.png".format(k, n, k_neighbors, N, num_neg, debug_prturb_size))

plt.figure()
plt.hist(dev_frm_on_mfld_poca)
plt.xlabel("L2 distance")
plt.ylabel("count")
plt.title("true vs approx. point of approach for k={},n={} ({} random samples)".format(k, n, test_on))
plt.savefig("dev_trueVapprox_poca_k{}n{}_knn{}_N{}_num_neg{}_dps{}.png".format(k, n, k_neighbors, N, num_neg, debug_prturb_size))


