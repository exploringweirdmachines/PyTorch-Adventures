import cupy as cp
import numpy as np
import threading
import queue

class Dataset:
    """
    Dataset base class needs __len__ and __getitem__
    """
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
    
class DataLoader:
    def __init__(self, 
                 dataset,
                 batch_size=32, 
                 shuffle=True, 
                 num_workers=0,
                 prefetch=2, 
                 collate_fn=None):
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.prefetch = prefetch
        self.collate_fn = collate_fn if collate_fn is not None else self.default_collate

        self.indices = np.arange(len(dataset))
        self.queue = queue.Queue(maxsize=prefetch)
        self.workers = []
        self.stop_signal = False

