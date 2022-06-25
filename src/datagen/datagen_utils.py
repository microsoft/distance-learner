import os
import sys
import json
from collections import OrderedDict

import faiss
import torch
from torch.utils.data import Dataset

import numpy as np

def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False

class FaissKNeighbors:
    def __init__(self, k=5):
        self.index = None
        self.y = None
        self.k = k

    def fit(self, X, y=None, gpu_index=2):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(np.array(X).astype(np.float32))
        self.res = faiss.StandardGpuResources()

        self.gpu_index = faiss.index_cpu_to_gpu(
            self.res,
            gpu_index,
            self.index
        )
        self.y = y

    def predict(self, X):
        distances, indices = self.gpu_index.search(np.array(X).astype(np.float32), k=self.k)
        votes = None
        if self.y is not None:
            votes = self.y[indices]
            predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return distances, indices

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

     CAUTION: don't add points to it in a parallelized fashion; compute
     points parallely but append sequentially
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

        self.cache_dump = None
        
        if data is not None:
            flush_it = False
            for i in range(len(data)):
                if i == len(data) - 1:
                    flush_it = True
                self.append(data[i], flush_it)

        self.is_flushed = False
    
    def append(self, pt, flush_it=False):
        
        self.chunk[self.cur_idx] = pt
        self.cur_idx += 1
        self.cur_len += 1
        
        self.is_flushed = False

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
                raise RuntimeWarning("no data to flush!")
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
            assert (load_chunk[:load_end_offset] == self.chunk[:load_end_offset]).all() == True, "loaded chunk not same as data to be dumped!"
            
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
        
        self.is_flushed = True
    
    def __getitem__(self, idx):

        if idx > self.cur_len:
            raise IndexError("idx ({}) > .cur_len attribute".format(idx, self.cur_len))
            
        key_to_load = idx // self.chunk_size
        idx_to_load = idx % self.chunk_size
        
        if key_to_load not in self.map:
            # this means the required id is in self.chunk
            if idx_to_load >= self.cur_idx:
                raise IndexError("idx_to_load exceeds .cur_idx attribute")
        fn = self.map[key_to_load]

        if not (key_to_load == self.cache_dump["key"] and idx_to_load >= self.cache_dump["start_idx"]):
            self.cache_dump = torch.load(fn)
        if idx_to_load > self.cache_dump["start_idx"] + self.cache_dump["end_offset"]:
            raise IndexError("idx out of bounds")

        return self.cache_dump["chunk"][idx_to_load]
    
        
    def __len__(self):
        return self.cur_len

class TensorFileDataset(Dataset):
    """
    Inspired by ImageFolder (https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html)
    and how ImageNet is stored by Pytorch to store large tensors in a directory like structure
    """
    
    def __init__(self,
                 root_dir,
                 total_len=100000,
                 per_dir_size=10000):
        self.root_dir = root_dir
        self.total_len = total_len
        self.per_dir_size = per_dir_size
        
        self.flist = list()
        
        self.make_dataset()
        
    
    def __len__(self):
        return self.total_len
        
    
    def __getitem__(self, idx):
        
        req_fn = self.flist[idx]
        if not os.path.exists(req_fn):
            raise RuntimeError("requested file {} does not exist!".format(req_fn))
            
        arr = torch.load(req_fn)
        return arr
    
    def make_dataset(self):
        num_dirs = (self.total_len // self.per_dir_size) + 1
        for i in range(num_dirs):
            cur_dir_idx = 0
            for j in range(self.per_dir_size):
                fn = str((i * self.per_dir_size) + j) + ".pth"
                full_path = os.path.join(self.root_dir, str(i), fn)
                self.flist.append(full_path)
                
            