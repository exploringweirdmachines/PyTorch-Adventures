import os
import cupy as cp
import numpy as np
from cupyx.distributed import NCCLBackend
import mytorch
import wandb

class Accelerator:
    def __init__(self, 
                 num_gpus=None, 
                 rank=None, 
                 master_addr="127.0.0.1", 
                 master_port="13333",
                 gradient_accumulation_steps=1):
        
        ### Set Number of GPUs if not provided from environment ###
        self.rank = rank if rank is not None else int(os.environ.get("RANK", 0))
        self.world_size = num_gpus if num_gpus is not None else int(os.environ.get("WORLD_SIZE", 1))

        ### Set Address and Port ####
        self.master_addr = master_addr
        self.master_port = master_port

        ### Set Device for this rank ###
        cp.cuda.Device(self.rank).use()
        
        ### Accumulation ###
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.step_counter = 0

        ### Initialize NCCL ###
        self.comm = None
        if self.world_size > 1:
            self.comm = NCCLBackend(
                n_devices=self.world_size, 
                rank=self.rank, 
                host=self.master_addr, 
                port=int(self.master_port)
            )

        ### Random Seed per Rank ###
        cp.random.seed(seed=42 + self.rank)

    def is_main_process(self):
        return self.rank == 0
    
    def prepare(self, *args, **kwargs):

        prepared = []

        for obj in args:
            if isinstance(obj, mytorch.nn.Module):
                prepared.append(self.prepare_model(obj))
            elif isinstance(obj, mytorch.optim.Optimizer):
                prepared.append(self.prepare_optimizer(obj))
            elif isinstance(obj, mytorch.data.DataLoader):
                prepared.append(self.prepare_dataloaders(obj))
        
        return prepared
    
    def prepare_model(self, model):

        ### Store Access to Model ###
        self.model = model

        ### Broadcast Weights from Model into Other GPUs ###
        if self.comm is not None:
            print("WE ARE SYNCING!")
            for param in self.model.parameters():
                self.comm.broadcast(param.data, root=0)
        
        return self.model
    
    def prepare_optimizer(self, optimizer):

        accelerator = self 

        class OptimizerWrapper:
            def __init__(self, base_optimizer):
                self.base_optimizer = base_optimizer
            
            def step(self, *args, **kwargs):
                if accelerator.step_counter % accelerator.gradient_accumulation_steps == 0:
                    return self.base_optimizer.step(*args, **kwargs)
            
            def zero_grad(self, *args, **kwargs):
                if accelerator.step_counter % accelerator.gradient_accumulation_steps == 0:
                    return self.base_optimizer.zero_grad(*args, **kwargs)
            
            def __getattr__(self, name):
                return getattr(self.base_optimizer, name)
            
        return OptimizerWrapper(optimizer)
    
    def prepare_dataloaders(self, dataloader):

        if self.world_size <= 1:
            return dataloader

        class ShardDataset:
            def __init__(self, base_dataset, rank, world_size, shuffle=True):
                self.base = base_dataset
                self.rank = rank
                self.world_size = world_size
                self.shuffle = shuffle
                self.epoch = 0

                ### Number of Samples per Rank ###
                self.num_samples_per_rank = (len(self.base) + self.world_size - 1) // self.world_size
                self.total_size = self.num_samples_per_rank * self.world_size

                ### Initialize Indices ###
                self.indices = np.arange(len(self.base))

            def set_epoch(self, epoch):

                ### Per Epoch Reshuffle of Data Before Resharding ###
                self.epoch = epoch
                rand_gen = np.random.default_rng()
                indices = np.arange(len(self.base))

                ### Random Shuffle Indices ###
                if self.shuffle:
                    indices = rand_gen.permutation(indices)

                ### Pad to make Divisible by World Size * Samples Per Rank ###
                ### This makes sure we have even number of batches every time ###
                if len(indices) < self.total_size:
                    padding = rand_gen.choice(indices, self.total_size - len(indices), replace=True)
                    indices = np.concatenate([indices, padding])

                self.indices = indices

            def __len__(self):
                return (len(self.base) + self.world_size - 1) // self.world_size
        
            def __getitem__(self, idx):

                """
                Interleaved sampling:
                Dataset indices: 0 1 2 3 4 5 6 7 8 9 10 11, ...

                Rank 0 gets: 0, 4, 8, ...
                Rank 1 gets: 1, 5, 9, ...
                Rank 2 gets: 2, 6, 10, ...
                Rank 3 gets: 3, 7, 11, ...
                """
                real_idx = self.indices[idx*self.world_size + self.rank]
                return self.base[real_idx]
        
        ### Grab Old Dataset ###
        shuffle_flag = getattr(dataloader, "shuffle", True)
        base_dataset = dataloader.dataset
        sharded_dataset = ShardDataset(base_dataset, world_size=self.world_size, rank=self.rank, shuffle=shuffle_flag)
        dataloader.dataset = sharded_dataset

        ### Wrap Dataloader for Epoch Based Shuffling ###
        class EpochShuffledDataLoader:
            def __init__(self, dataloader, sharded_dataset):
                self.dataloader = dataloader
                self.sharded_dataset = sharded_dataset
                self.epoch = 0

            def __iter__(self):
                self.sharded_dataset.set_epoch(self.epoch)
                self.epoch += 1
                return iter(self.dataloader)

            def __len__(self):
                return len(self.dataloader)
            
        return EpochShuffledDataLoader(dataloader, sharded_dataset)
    
    def backward(self, loss):

        ### Scale Loss By Gradient Accumulation Steps ###
        loss = loss/self.gradient_accumulation_steps

        ### Normal backward ###
        loss.backward()
        self.step_counter += 1

        if self.step_counter % self.gradient_accumulation_steps == 0:
            ### Allreduce Gradients ###
            if self.comm is not None:
                for param in self.model.parameters():
                    if hasattr(param, "grad") and param.grad is not None:
                        out = cp.empty_like(param.grad)
                        self.comm.all_reduce(param.grad, out, op="sum")
                        param.grad[:] = out / self.world_size

    def clip_grad_norm_(self, max_norm=1.0):
        
        ### Only Clip Norm when Accumulation Complete ###
        if self.step_counter % self.gradient_accumulation_steps != 0:
            return None
        
        ### Compute Total Norm across
        total_norm = 0.0
        for param in self.model.parameters():
            if hasattr(param, "grad") and param.grad is not None:
                total_norm += float(cp.linalg.norm(param.grad.reshape(-1), ord=2.0)) ** 2
        total_norm = total_norm ** 0.5

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            for param in self.model.parameters():
                if hasattr(param, "grad") and param.grad is not None:
                    param.grad *= clip_coef
        
    def gather_for_metrics(self, value):
 
        assert isinstance(value, mytorch.Tensor), "Value must be a Tensor"
        assert value.shape == (), "Value must be a Scalar"

        if self.world_size <= 1 or self.comm is None:
            return float(value.data[0])

        out = cp.zeros_like(value.data)
        self.comm.all_reduce(value.data, out, op="sum")
        return out.item() / self.world_size
    
    def wait_for_everyone(self):
        self.comm.barrier()
    
    def end_training(self):
        self.comm.barrier()
        self.comm.stop()

    def print(self, *args, **kwargs):
        if self.is_main_process():
            print(*args, **kwargs)

    def init_tracker(self,
                     project_name, 
                     run_name, 
                     config=None):

        if config is not None:
            assert isinstance(config, dict), "Config must be a dictionary!"
            
        if self.rank == 0:
            wandb.init(
                project=project_name,
                name=run_name,
                config=config
            )

    def log(self, log_dict, step):

        assert isinstance(log_dict, dict), "log_dict must be dictionary!"
        assert isinstance(step, int), "step must be integer!"

        if self.rank == 0:
            wandb.log(log_dict, step=step)

    def __del__(self):
        if self.comm is not None:
            self.comm.stop()