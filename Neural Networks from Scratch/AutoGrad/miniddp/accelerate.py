import os
import cupy as cp
import numpy as np
from cupyx.distributed import NCCLBackend
import mytorch
import wandb

class GradScaler:
    def __init__(self, 
                 init_scale=2.0**16,
                 growth_factor=2.0, 
                 backoff_factor=0.5, 
                 growth_interval=2000):
        
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.unskipped = 0
        self._grads_unscaled = False

    def scale_loss(self, loss):
        """
        The key part of mixes precision training is scaling our loss. 
        The problem is fp16 has lower numerical precision, so small 
        loss values will lead to grad underflows. 

        So what we do is scale our gradients up by a factor, and the 
        scale is initialized as approximately the largest representable
        value of fp16 (65536.0) 

        This means if you had a loss like 0.01, it gets scaled up to 
        65536.0 * 0.01 = 655.36

        But what if our loss is 2? Then our scaled value is 65536 * 2
        which exceeds the maximum representable numbers in fp16. Thus
        we need some rules about how we dynamically update the scaling
        factor!
        """
        return loss.astype("float32") * self.scale
    
    def unscale_grads(self, params):

        """
        The scaling we do in scale_loss() is just for keeping our
        precision when computing gradients. Of course, our gradients
        need to be scaled down again to we have the correct magnitudes!
        """

        ### If grads are alread unscaled, nothing to do! ###
        if self._grads_unscaled:
            return True
        
        inv_scale = 1.0 / self.scale
        for param in params:
            if hasattr(param, "grad") and param.grad is not None:
                param.grad *= inv_scale

        ### Flag that gradients have been unscaled ###
        self._grads_unscaled = True

    def update(self, found_inf):

        """
        To update our scaling factor:

        If we found INF/NAN, that means we are scaling
        our values too high (self.scale is too large) 
        so lets reduce our scaling factor. This is important
        at the start of training as our loss will be high

        If we dont find any INF/NAN, that means we are good 
        to go, and if we consistently find this to be true
        for atleast self.growth_interval steps, we can go 
        ahead and increase our scale. This is important as our
        loss gets smaller and smaller

        A small scale reduces the risk of overflows, but may not 
        sufficiently amplify tiny grads leading to a loss of precision

        Large scale better preserves the small gradients, but 
        increases risk of overflow.

        """
        if found_inf:
            self.scale *= self.backoff_factor
            self.unskipped = 0
        else:
            self.unskipped += 1
            if self.unskipped == self.growth_interval:
                self.scale *= self.growth_factor
                self.unskipped = 0
    
    def reset_unscaled_flag(self):
        self._grads_unscaled = False

class Accelerator:
    def __init__(self, 
                 num_gpus=None, 
                 rank=None, 
                 master_addr="127.0.0.1", 
                 master_port="13333",
                 gradient_accumulation_steps=1,
                 mixed_precision=False):
        
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

        ### Mixed Precision ###
        self.mixed_precision = mixed_precision
        self.skip_optimizer_step = False
        if self.mixed_precision:
            self.scaler = GradScaler()

    def is_main_process(self):
        return self.rank == 0
    
    @property
    def device(self):
        return f"cuda:{self.rank}"
    
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
        self.model = model.to(f"cuda:{self.rank}")
        
        ### Broadcast Weights from Model into Other GPUs ###
        if self.comm is not None:
            for param in self.model.parameters():
                self.comm.broadcast(param.data._array, root=0)

        ### Set Up Mixed-Precision Training ###
        if self.mixed_precision:
            self.fp32_params = []
            for param in self.model.parameters():

                ### Store a Copy of Full Precision Weights ###
                fp32_param = mytorch.Tensor(param.data._array.copy(), dtype=mytorch.float32)
                fp32_param.requires_grad = param.requires_grad
                self.fp32_params.append(fp32_param)

                ### Cast our Parameters in Model to Float16 ###
                param.data._array = param.data._array.astype("float16")
        
        return self.model
    
    def prepare_optimizer(self, optimizer):

        accelerator = self 

        ### We will update Full Precision Params but train w/ Half Precision ###
        if self.mixed_precision:
            optimizer.params = self.fp32_params
            
        ### Adam has momentum params that have already been initialized ###
        ### we need to reinit them on the correct device ###
        if hasattr(optimizer, "m"):
            optimizer.m = [mytorch.zeros_like(p).data for p in optimizer.params]
        if hasattr(optimizer, "v"):
            optimizer.v = [mytorch.zeros_like(p).data for p in optimizer.params]

        class OptimizerWrapper:
            def __init__(self, base_optimizer):
                self.base_optimizer = base_optimizer
            
            def step(self, *args, **kwargs):

                ### Only Step After Grad Accumulations are Done ###
                if accelerator.step_counter % accelerator.gradient_accumulation_steps == 0:
                    ### In mixed precision we may have to skip a step ###
                    if accelerator.mixed_precision and accelerator.skip_optimizer_step:
                        ### Reset the flag and do nothing ###
                        accelerator.skip_optimizer_step = False
                        return 
                    
                    ### update our parameters ###
                    self.base_optimizer.step(*args, **kwargs)

                    ### If we just updated and were in mixed precision mode, we only updated ###
                    ### our fp32 copy of the weights. We need to copy those back into our model ###
                    ### now for the next iteration! ###
                    if accelerator.mixed_precision:
                        for fp32_param, param in zip(accelerator.fp32_params, accelerator.model.parameters()):
                            param.data = fp32_param.data.astype(cp.float16)

            def zero_grad(self, *args, **kwargs):   

                if accelerator.step_counter % accelerator.gradient_accumulation_steps == 0:
                    self.base_optimizer.zero_grad(*args, **kwargs)

                    ### If in Mixed Precision ###
                    ### Remember, our optimizer looks at the copy of full precision weights ###
                    ### our model still looks at our half precision weights, so we need to manually ###
                    ### zero them out here ###
                    if accelerator.mixed_precision:
                        for param in accelerator.model.parameters():
                            if hasattr(param, "grad") and param.grad is not None:
                                param.grad[:] = 0.0

                        ### We have updated our model, reset flag for the next scaling ###
                        accelerator.scaler.reset_unscaled_flag()
            
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

        ### If Mixed Precision We Scale our Loss ###
        if self.mixed_precision:
            loss = self.scaler.scale_loss(loss)
            self.scaler.reset_unscaled_flag()

        ### Normal backward ###
        loss.backward()
        self.step_counter += 1

        if self.step_counter % self.gradient_accumulation_steps == 0:

            ### Check for any NAN Gradients ###
            skip_step = False
            if self.mixed_precision:
                found_inf = 0.0
                for param in self.model.parameters():
                    if hasattr(param, "grad") and param.grad is not None:
                        ### Detect INF and NAN (If we do set as 1.0) ###
                        if cp.any(cp.isinf(param.grad)) or cp.any(cp.isnan(param.grad)):
                            found_inf = 1.0
                            break
                
                ### Gather from all devices to see if ANY device had an NAN/INF ###
                found_inf_arr = cp.array([found_inf])
                if self.comm is not None:
                    out = cp.empty_like(found_inf_arr)
                    self.comm.all_reduce(found_inf_arr, out, op="max")
                found_inf = found_inf_arr > 0.0

                if found_inf:
                    ### Zero Scaled Grads as its is not a goo dupdate ###
                    for param in self.model.parameters():
                        if hasattr(param, "grad") and param.grad is not None:
                            param.grad[:] = 0.0
                    self.scaler.update(True)
                    skip_step = True
                
                else:
                    self.scaler.unscale_grads(self.model.parameters())
                    self.scaler.update(False)

            ### Allreduce Gradients ###
            if self.comm is not None:
                for param in self.model.parameters():
                    if hasattr(param, "grad") and param.grad is not None:
                        out = cp.empty_like(param.grad)

                        ### Quick check. In our backward pass there are two options:
                        ### - Auto backward which will use our Array type
                        ### - Manual backward which will use either cp.ndarray or np.ndarray. But 
                        ###   we are in distributed training here so we only care about cp.ndarray

                        ### This means if we have an Array type we need to get the "_array" that hold the 
                        ### actual underlying data in Cupy for NCCL all_reduce. But if its already a 
                        ### cp.ndarray theres nothing to get, so we just have a quick sanity check here
                        self.comm.all_reduce(param.grad._array if hasattr(param.grad, "_array") else param.grad, out, op="sum")
                        param.grad[:] = out / self.world_size

            ### Cast Grads back to FP32 to Update our FP32 Copy of Weights ###
            if self.mixed_precision and not skip_step:
                for fp32_param, param in zip(self.fp32_params, self.model.parameters()):
                    if hasattr(param, "grad") and param.grad is not None:
                        fp32_param.grad = param.grad.astype(cp.float32)
                    else:
                        fp32_param.grad = None

            self.skip_optimizer_step = skip_step

    def clip_grad_norm_(self, max_norm=1.0):
        
        ### Only Clip Norm when Accumulation Complete ###
        if self.step_counter % self.gradient_accumulation_steps != 0:
            return None
        
        ### If we are in Mixed Precision Mode ###
        if self.mixed_precision:
            self.scaler.unscale_grads(self.model.parameters())
        
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

        ### If in Mixed Precision, we want to copy our clipped grads to the fp32 params for optimizer ###
        if self.mixed_precision:
            for fp32_param, param in zip(self.fp32_params, self.model.parameters()):
                if hasattr(param, "grad") and param.grad is not None:
                    fp32_param.grad = param.grad.astype(cp.float32)
                else:
                    fp32_param.grad = None

    def gather_for_metrics(self, value):
 
        assert isinstance(value, mytorch.Tensor), "Value must be a Tensor"
        assert value.shape == (), "Value must be a Scalar"

        if self.world_size <= 1 or self.comm is None:
            return float(value.data._array[0])
        
        data = value.data._array if hasattr(value.data, "_array") else value.data
        out = cp.zeros_like(data)
        self.comm.all_reduce(data, out, op="sum")
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

# class Accelerator:
#     def __init__(self, 
#                  num_gpus=None, 
#                  rank=None, 
#                  master_addr="127.0.0.1", 
#                  master_port="13333",
#                  gradient_accumulation_steps=1):
        
#         ### Set Number of GPUs if not provided from environment ###
#         self.rank = rank if rank is not None else int(os.environ.get("RANK", 0))
#         self.world_size = num_gpus if num_gpus is not None else int(os.environ.get("WORLD_SIZE", 1))

#         ### Set Address and Port ####
#         self.master_addr = master_addr
#         self.master_port = master_port

#         ### Set Device for this rank ###
#         cp.cuda.Device(self.rank).use()
        
#         ### Accumulation ###
#         self.gradient_accumulation_steps = gradient_accumulation_steps
#         self.step_counter = 0

#         ### Initialize NCCL ###
#         self.comm = None
#         if self.world_size > 1:
#             self.comm = NCCLBackend(
#                 n_devices=self.world_size, 
#                 rank=self.rank, 
#                 host=self.master_addr, 
#                 port=int(self.master_port)
#             )

#         ### Random Seed per Rank ###
#         cp.random.seed(seed=42 + self.rank)

#     def is_main_process(self):
#         return self.rank == 0
    
#     def prepare(self, *args, **kwargs):

#         prepared = []

#         for obj in args:
#             if isinstance(obj, mytorch.nn.Module):
#                 prepared.append(self.prepare_model(obj))
#             elif isinstance(obj, mytorch.optim.Optimizer):
#                 prepared.append(self.prepare_optimizer(obj))
#             elif isinstance(obj, mytorch.data.DataLoader):
#                 prepared.append(self.prepare_dataloaders(obj))
        
#         return prepared
    
#     def prepare_model(self, model):

#         ### Store Access to Model ###
#         self.model = model

#         ### Broadcast Weights from Model into Other GPUs ###
#         if self.comm is not None:
#             for param in self.model.parameters():
#                 self.comm.broadcast(param.data, root=0)
        
#         return self.model
    
#     def prepare_optimizer(self, optimizer):

#         accelerator = self 

#         class OptimizerWrapper:
#             def __init__(self, base_optimizer):
#                 self.base_optimizer = base_optimizer
            
#             def step(self, *args, **kwargs):
#                 if accelerator.step_counter % accelerator.gradient_accumulation_steps == 0:
#                     return self.base_optimizer.step(*args, **kwargs)
            
#             def zero_grad(self, *args, **kwargs):
#                 if accelerator.step_counter % accelerator.gradient_accumulation_steps == 0:
#                     return self.base_optimizer.zero_grad(*args, **kwargs)
            
#             def __getattr__(self, name):
#                 return getattr(self.base_optimizer, name)
            
#         return OptimizerWrapper(optimizer)
    
#     def prepare_dataloaders(self, dataloader):

#         if self.world_size <= 1:
#             return dataloader

#         class ShardDataset:
#             def __init__(self, base_dataset, rank, world_size, shuffle=True):
#                 self.base = base_dataset
#                 self.rank = rank
#                 self.world_size = world_size
#                 self.shuffle = shuffle
#                 self.epoch = 0

#                 ### Number of Samples per Rank ###
#                 self.num_samples_per_rank = (len(self.base) + self.world_size - 1) // self.world_size
#                 self.total_size = self.num_samples_per_rank * self.world_size

#                 ### Initialize Indices ###
#                 self.indices = np.arange(len(self.base))

#             def set_epoch(self, epoch):

#                 ### Per Epoch Reshuffle of Data Before Resharding ###
#                 self.epoch = epoch
#                 rand_gen = np.random.default_rng()
#                 indices = np.arange(len(self.base))

#                 ### Random Shuffle Indices ###
#                 if self.shuffle:
#                     indices = rand_gen.permutation(indices)

#                 ### Pad to make Divisible by World Size * Samples Per Rank ###
#                 ### This makes sure we have even number of batches every time ###
#                 if len(indices) < self.total_size:
#                     padding = rand_gen.choice(indices, self.total_size - len(indices), replace=True)
#                     indices = np.concatenate([indices, padding])

#                 self.indices = indices

#             def __len__(self):
#                 return (len(self.base) + self.world_size - 1) // self.world_size
        
#             def __getitem__(self, idx):

#                 """
#                 Interleaved sampling:
#                 Dataset indices: 0 1 2 3 4 5 6 7 8 9 10 11, ...

#                 Rank 0 gets: 0, 4, 8, ...
#                 Rank 1 gets: 1, 5, 9, ...
#                 Rank 2 gets: 2, 6, 10, ...
#                 Rank 3 gets: 3, 7, 11, ...
#                 """
#                 real_idx = self.indices[idx*self.world_size + self.rank]
#                 return self.base[real_idx]
        
#         ### Grab Old Dataset ###
#         shuffle_flag = getattr(dataloader, "shuffle", True)
#         base_dataset = dataloader.dataset
#         sharded_dataset = ShardDataset(base_dataset, world_size=self.world_size, rank=self.rank, shuffle=shuffle_flag)
#         dataloader.dataset = sharded_dataset

#         ### Wrap Dataloader for Epoch Based Shuffling ###
#         class EpochShuffledDataLoader:
#             def __init__(self, dataloader, sharded_dataset):
#                 self.dataloader = dataloader
#                 self.sharded_dataset = sharded_dataset
#                 self.epoch = 0

#             def __iter__(self):
#                 self.sharded_dataset.set_epoch(self.epoch)
#                 self.epoch += 1
#                 return iter(self.dataloader)

#             def __len__(self):
#                 return len(self.dataloader)
            
#         return EpochShuffledDataLoader(dataloader, sharded_dataset)
    
#     def backward(self, loss):

#         ### Scale Loss By Gradient Accumulation Steps ###
#         loss = loss/self.gradient_accumulation_steps

#         ### Normal backward ###
#         loss.backward()
#         self.step_counter += 1

#         if self.step_counter % self.gradient_accumulation_steps == 0:
#             ### Allreduce Gradients ###
#             if self.comm is not None:
#                 for param in self.model.parameters():
#                     if hasattr(param, "grad") and param.grad is not None:
#                         out = cp.empty_like(param.grad)
#                         self.comm.all_reduce(param.grad, out, op="sum")
#                         param.grad[:] = out / self.world_size

#     def clip_grad_norm_(self, max_norm=1.0):
        
#         ### Only Clip Norm when Accumulation Complete ###
#         if self.step_counter % self.gradient_accumulation_steps != 0:
#             return None
        
#         ### Compute Total Norm across
#         total_norm = 0.0
#         for param in self.model.parameters():
#             if hasattr(param, "grad") and param.grad is not None:
#                 total_norm += float(cp.linalg.norm(param.grad.reshape(-1), ord=2.0)) ** 2
#         total_norm = total_norm ** 0.5

#         clip_coef = max_norm / (total_norm + 1e-6)
#         if clip_coef < 1.0:
#             for param in self.model.parameters():
#                 if hasattr(param, "grad") and param.grad is not None:
#                     param.grad *= clip_coef
        
#     def gather_for_metrics(self, value):
 
#         assert isinstance(value, mytorch.Tensor), "Value must be a Tensor"
#         assert value.shape == (), "Value must be a Scalar"

#         if self.world_size <= 1 or self.comm is None:
#             return float(value.data[0])

#         out = cp.zeros_like(value.data)
#         self.comm.all_reduce(value.data, out, op="sum")
#         return out.item() / self.world_size
    
#     def wait_for_everyone(self):
#         self.comm.barrier()
    
#     def end_training(self):
#         self.comm.barrier()
#         self.comm.stop()

#     def print(self, *args, **kwargs):
#         if self.is_main_process():
#             print(*args, **kwargs)

#     def init_tracker(self,
#                      project_name, 
#                      run_name, 
#                      config=None):

#         if config is not None:
#             assert isinstance(config, dict), "Config must be a dictionary!"
            
#         if self.rank == 0:
#             wandb.init(
#                 project=project_name,
#                 name=run_name,
#                 config=config
#             )

#     def log(self, log_dict, step):

#         assert isinstance(log_dict, dict), "log_dict must be dictionary!"
#         assert isinstance(step, int), "step must be integer!"

#         if self.rank == 0:
#             wandb.log(log_dict, step=step)

#     def __del__(self):
#         if self.comm is not None:
#             self.comm.stop()