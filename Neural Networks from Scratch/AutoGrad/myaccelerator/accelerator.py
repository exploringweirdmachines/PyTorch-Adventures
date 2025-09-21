import os
import cupy as cp
from cupyx.distributed import NCCLBackend

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

    def is_main_process(self):
        return self.rank == 0
    
    def prepare_model(self, model):
        self.model = model
        return self.model
    
    def prepare_optimizer(self, optimizer):

        class OptimizerWrapper:
            def __init__(self, base_optimizer):
                self.base_optimizer = base_optimizer
            
            def step(self, *args, **kwargs):
                if self.step_counter % self.gradient_accumulation_steps == 0:
                    return self.base_optimizer.step(*args, **kwargs)
            
            def zero_grad(self, *args, **kwargs):
                if self.step_counter % self.gradient_accumulation_steps == 0:
                    return self.base_optimizer.zero_grad(*args, **kwargs)
            
            def __getattr__(self, name):
                return getattr(self.base_optimizer, name)
            
        return OptimizerWrapper(optimizer)
    
    def prepare_dataloaders(self, dataloader):
        return dataloader
    
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
                        out = cp.zeros_like(param.grad)
                        self.comm.all_reduce(param.grad, out, op="sum")
                        param.grad[:] = out / self.world_size

    def wait_for_everyone(self):
        self.comm.barrier()
    
    def end_training(self):
        self.comm.barrier()
        self.comm.stop()

    def print(self, *args, **kwargs):
        if self.is_main_process():
            print(*args, **kwargs)

    def __del__(self):
        if self.comm is not None:
            self.comm.stop()