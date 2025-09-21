import numpy as np
import threading
import queue

class DataLoader:
    """
    Minimal multithreaded DataLoader with optional collate function.

    Features:
    - Supports single-threaded and multi-threaded batch loading
    - Optional shuffling at the start of each epoch
    - Prefetch queue for asynchronous loading
    - Custom collate function or a default stack-based collator
    - Clean worker shutdown
    - __len__ gives number of batches per epoch
    """

    def __init__(self, dataset, batch_size=32, shuffle=True, 
                 num_workers=0, prefetch_factor=2, collate_fn=None,
                 timeout=1):

        """
        Args:
            dataset: Any object implementing __len__ and __getitem__.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle dataset indices at start of each epoch.
            num_workers (int): Number of background worker threads. 0 = no multithreading.
            prefetch (int): Number of batches to prefetch in the queue.
            collate_fn (callable): Function to merge list of samples into a batch.
                                   Defaults to stacking along axis 0.
            timeout (float): Amount of seconds you will wait for batches to be grabbed before exiting
        """

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.prefetch = prefetch_factor  # number of batches to prefetch
        self.collate_fn = collate_fn if collate_fn is not None else self.default_collate
        self.timeout = timeout

        self.indices = np.arange(len(dataset))
        
        if self.num_workers > 0:
            self.queue = queue.Queue(maxsize=prefetch_factor * num_workers)
        else:
            self.queue = None

        self.workers = []
        self.lock = threading.Lock() # Can use to makes sure before a thread can access a shared resource it must aquire a lock
        self.stop_signal = threading.Event() # enables inter-thread comm. Initialized as true, can be toggled
        self.current_idx = 0

    def default_collate(self, batch):
        """
        Default collate function: stacks each element along axis 0.
        Handles datasets returning tuples/lists per sample.
        """

        if isinstance(batch[0], (tuple, list)):
            return tuple(self.default_collate([b[i] for b in batch]) for i in range(len(batch[0])))
        else:
            return np.stack(batch)
    
    def _get_next_batch_indices(self):

        """
        Grab next slice of indexes for sampling
        """

        with self.lock:
            if self.current_idx >= len(self.dataset):
                raise StopIteration
            
            batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
            self.current_idx += self.batch_size

        return batch_indices

    def _worker_loop(self):
        """
        This is the code that each thread will run. It repeatedly grabs some indices
        from the dataset, grab the samples, collates them and saves them in the queue. 

        The queue only holds a max of `prefetch` batches in memory. So once the queue is full
        queue.put() will block the queue  until it is emptied again. 
        
        if _get_next_batch_indices() raises a StopIteration, the _worker_loop will also break
        and exit!
        """
        while not self.stop_signal.is_set(): # Returns true if internal flag is true, init as false
            try:
                batch_indices = self._get_next_batch_indices()
                batch = [self.dataset[int(i)] for i in batch_indices]
                batch_array = self.collate_fn(batch)

                ### Blocking statement,that waits until space in the queue opens up ###
                self.queue.put(batch_array)
            except StopIteration:
                break    
        
        ## when we hit stop iteration, just add a None ###
        self.queue.put(None)

    def __iter__(self):
        
        """
        This returns an iterable that we can actually iterate through. If we shuffle, we first
        randomize the order of the indexes. It also resets our current index to start from the 
        beginning and then the stop_signal is set to false, as we are just starting!

        We then spawn num_workers threads each running the worker loop (independently)
        """
        
        if self.shuffle:
            self.indices = np.random.permutation(len(self.dataset))

        ### Sets current index back to start ###
        self.current_idx = 0

        ### Sets stop signal to False ###
        self.stop_signal.clear()

        ### Start worker threads if needed. Even if you have num_workers=1, it will ###
        ### prefetch batches to store in memory ###
        self.workers = []
        for _ in range(self.num_workers):
            t = threading.Thread(target=self._worker_loop, daemon=True)
            t.start()
            self.workers.append(t)

        return self

    def __next__(self):

        """
        When single threaded we just grab a batch and collate

        When multithreaded we get a batch from our queue. If no batch arrives
        within 5 seconds then the code sets a stop_signal to be True and we 
        stop iteration. 
        """
        if self.num_workers == 0:
            
            ### Load data synchronously ###
            try:
                batch_indices = self._get_next_batch_indices()
                batch = [self.dataset[int(i)] for i in batch_indices]
                return self.collate_fn(batch)
            except StopIteration:
                raise StopIteration

        while True:
            try:
                batch = self.queue.get(timeout=self.timeout)
                if batch is None: # Info from worker that we are out 
                    raise StopIteration
                return batch
            except queue.Empty:
                self.stop_signal.set()
                raise StopIteration
    
    def shutdown(self):
        
        ### Sets stop_signal to True telling all workers to stop collecting ###
        self.stop_signal.set()
        for t in self.workers:
            t.join(timeout=1)
        self.workers.clear()
    
    def __del__(self):
        """
        What to do when called by Garbage Collector
        """
        self.shutdown()


    def __len__(self):
        ### Ensures always rounds up for num batches ###
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


if __name__ == "__main__":

    from torchvision.datasets import MNIST
    from tqdm import tqdm

    def default_collate(batch):
        # stack features and labels
        x = np.stack([np.array(b[0]).reshape(-1) for b in batch])
        y = np.array([b[1] for b in batch])
        return x, y
    
    ### Load Dataset ###
    train_dataset = MNIST("../../data", train=True, download=True)

    ### Create dataloader ###
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=6, 
        collate_fn=default_collate
    )

    for images, labels in tqdm(train_loader):
        print(labels)
        pass