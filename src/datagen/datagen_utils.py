import os
import sys
from collections import OrderedDict

import faiss
import torch
from torch.utils.data import Dataset

import numpy as np

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

class TensorShardsDataset(Dataset):
    """
     each file created has:
     {
         "key": tells about order in overall array 
         "start_idx": smallest index in overall array in the file
         "end_offset": offset/index till before which this chunk has elements
         "chunk": actual dump of the chunk
     }
     
     to fetch populated elements do dump["chunk"][:dump["end_offset"]]
     for ease of debugging, chunks are initialized with np.nan
    """
    def __init__(self,\
                 root_dir,\
                 data_shape,\
                 data=None, \
                 chunk_size=100000,\
                ):
        
        self.root_dir = root_dir
        self.data_shape = data_shape
        self.chunk_size = chunk_size
        
        self.cur_len = 0 # current length that has been populated
        self.cur_idx = 0 # index where next element will go
        
        self.map = OrderedDict()
        self.verbose_map = OrderedDict()
        
        self.chunk = np.zeros((self.chunk_size, *self.data_shape))
        self.chunk[:] = np.nan
        
        if data is not None:
            flush_it = False
            for i in range(len(data)):
                if i == len(data) - 1:
                    flush_it = True
                self.append(data[i], flush_it)
    
    def append(self, pt, flush_it=False):
        
        self.chunk[self.cur_idx] = pt
        self.cur_idx += 1
        self.cur_len += 1
        
        if self.cur_len % self.chunk_size == 0 and self.cur_idx == self.chunk_size:
            self.flush_chunk()
        
        elif flush_it:
            self.flush_chunk()
            
    def extend(self, pts, flush_it=False):
        """
        TODO: re-write this and do better
        """
        for i in range(pts.shape[0]):
            self.append(pts[i])
    
    def _save_flush(self, fn, to_dump):
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir, exist_ok=True)
            
        torch.save(to_dump, fn)
    
    def flush_chunk(self):
        key_to_dump_in = self.cur_len // self.chunk_size # which multiple of self.chunk_size is currently running
        if self.cur_len % self.chunk_size == 0:
            if self.cur_len == 0:
                raise RunTimeWarning("no data to flush!")
            else:
                key_to_dump_in -= 1
        
        key = key_to_dump_in
        start_idx = None
        end_offset = None
        
        if key_to_dump_in in self.map:
            fn = self.map[key_to_dump_in]
            load_dump = torch.load(fn)
            load_chunk = load_dump["chunk"]
            load_start = self.verbose_map[key_to_dump_in]["start_idx"]
            load_end_offset = self.verbose_map[key_to_dump_in]["end_offset"]

            assert load_start == key_to_dump_in * self.chunk_size
            assert load_end_offset <= self.cur_idx
            assert (load_chunk[:load_end] == self.chunk[:load_end]).all() == True, "loaded chunk not same as data to be dumped!"
            
            key = key_to_dump_in
            start_idx = self.verbose_map[key_to_dump_in]["start_idx"]
            end_offset = self.cur_idx
            
            self.verbose_map[key_to_dump_in]["end_offset"] = end_offset
        
        else:
            fn = os.path.join(self.root_dir, str(key_to_dump_in) + ".pth")
            start_idx = key_to_dump_in * self.chunk_size
            end_offset = self.cur_idx
            
            self.verbose_map[key_to_dump_in] = {
                "start_idx": start_idx,
                "end_offset": self.cur_idx,
                "name": fn,
            }
            
            self.map[key_to_dump_in] = fn
        
        to_dump = {
            "start_idx": start_idx,
            "end_offset": end_offset,
            "key": key_to_dump_in,
            "chunk": self.chunk
        }
        # print(to_dump)
        self._save_flush(fn, to_dump)
        if self.cur_idx == self.chunk_size: # chunk was full before dumping
            self.chunk[:] = np.nan # reset chunk
            self.cur_idx = 0 # reset current index so that new element is inserted at start        
                
    
    def __getitem__(self, idx):
        # print(idx)
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        key_to_load = idx // self.chunk_size
        idx_to_load = idx % self.chunk_size
        
        fn = self.map[key_to_load]
        
        loaded_chunk = torch.load(fn)
        return loaded_chunk["chunk"][idx_to_load]
    
        
    def __len__(self):
        return self.cur_len