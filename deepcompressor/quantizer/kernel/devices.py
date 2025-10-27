
import functools
import gc

import torch

from deepcompressor.data.cache import TensorCache
from deepcompressor.utils import tools


def _backup_deivces():
    return ["cuda:5", "cuda:6", "cuda:7"]
    

def try_all_devices(forced_dtype=None):
    logger = tools.logging.getLogger(__name__)
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # args_tensors = []
            # for arg in args:
            #     if isinstance(arg, torch.Tensor):
            #         args_tensors.append(arg)
            #     elif isinstance(arg, TensorCache):
            #         for _d in arg.data:
            #             args_tensors.append(_d)
            # if len(args_tensors) > 0:
            #     _arg_device = args_tensors[0].device
            #     _arg_dtype = args_tensors[0].dtype
            #     assert all(t.device == _arg_device for t in args_tensors)
            #     assert all(t.dtype == _arg_dtype for t in args_tensors)
            
            # kwargs_tensors = []
            # for k, v in kwargs.items():
            #     if isinstance(v, torch.Tensor):
            #         kwargs_tensors.append(v)
            #     elif isinstance(v, TensorCache):
            #         for _d in v.data:
            #             kwargs_tensors.append(_d)
            # if len(kwargs_tensors) > 0:
            #     _kwarg_device = kwargs_tensors[0].device
            #     _kwarg_dtype = kwargs_tensors[0].dtype
            #     assert all(t.device == _kwarg_device for t in kwargs_tensors), f"{list(kwargs.keys())}, {[t.device for t in kwargs_tensors]}"
            #     assert all(t.dtype == _kwarg_dtype for t in kwargs_tensors), f"{list(kwargs.keys())}, {[t.dtype for t in kwargs_tensors]}"
                
            # if len(args_tensors) == 0 and len(kwargs_tensors) == 0:
            #     # no tensor in args and kwargs
            #     return func(*args, **kwargs)
            # if len(args_tensors) > 0 and len(kwargs_tensors) > 0:
            #     assert _arg_device == _kwarg_device
            #     assert _arg_dtype == _kwarg_dtype
            
            # orig_device = args_tensors[0].device if len(args_tensors) > 0 else kwargs_tensors[0].device
            # orig_dtype = args_tensors[0].dtype if len(args_tensors) > 0 else kwargs_tensors[0].dtype
            
            assert len(args) >= 1 and isinstance(args[0], torch.Tensor)
            orig_device = args[0].device
            orig_dtype = args[0].dtype
            
            devices = [orig_device, *_backup_deivces()]
            to_dtype = orig_dtype if forced_dtype is None else forced_dtype
            
            tried_devices = []
            class _PlaceHolder:
                pass
            ret = _PlaceHolder()
            
            for idx, dev in enumerate(devices):
                try:
                    # if idx == 0:
                    #     prev_args = args
                    #     prev_kwargs = kwargs
                    moved_args = []
                    for arg in args:
                        if isinstance(arg, torch.Tensor):
                            moved_args.append(arg.to(device=dev, dtype=to_dtype))
                            del arg
                        # elif isinstance(arg, TensorCache):
                        #     _arg = TensorCache(
                        #         data=[_d.to(device=dev, dtype=to_dtype) for _d in arg.data],
                        #         channels_dim=arg.channels_dim,
                        #         reshape=arg.reshape,
                        #         num_cached=arg.num_cached,
                        #         num_total=arg.num_total,
                        #         num_samples=arg.num_samples,
                        #         orig_device=arg.orig_device
                        #     )
                        #     moved_args.append(_arg)
                        else:
                            moved_args.append(arg)
                    moved_kwargs = {}
                    for k, v in kwargs.items():
                        if isinstance(v, torch.Tensor):
                            moved_kwargs[k] = v.to(device=dev, dtype=to_dtype)
                            del v
                        # elif isinstance(v, TensorCache):
                        #     moved_kwargs[k] = TensorCache(
                        #         data=[_d.to(device=dev, dtype=to_dtype) for _d in v.data],
                        #         channels_dim=v.channels_dim,
                        #         reshape=v.reshape,
                        #         num_cached=v.num_cached,
                        #         num_total=v.num_total,
                        #         num_samples=v.num_samples,
                        #         orig_device=v.orig_device
                        #     )
                        else:
                            moved_kwargs[k] = v
                    # gc.collect()
                    # torch.cuda.empty_cache()
                    # torch.cuda.synchronize()
                    ret = func(*moved_args, **moved_kwargs)
                    break
                except (torch.OutOfMemoryError, RuntimeError) as e:
                    if isinstance(e, torch.OutOfMemoryError) or "CUDA out of memory" in str(e):
                        logger.info(f"** OOM OCCURRED when calling {func.__name__} on device {dev}")
                        tried_devices.append(dev)
                        last_error = e
                        # prev_args = moved_args
                        # prev_kwargs = moved_kwargs
                    else:
                        raise e
            if isinstance(ret, _PlaceHolder):
                logger.info(f"** OOM OCCURRED on these devices {tried_devices}")
                raise last_error
            else:
                if isinstance(ret, tuple):
                    moved_ret = []
                    for t in tuple:
                        if isinstance(t, torch.Tensor):
                            moved_ret.append(t.to(device=orig_device, dtype=orig_dtype))
                        else:
                            moved_ret.append(t)
                    return tuple(moved_ret)
                elif isinstance(ret, torch.Tensor):
                    return ret.to(device=orig_device, dtype=orig_dtype)
                else:
                    return ret

        return wrapper

    return decorator


#### for demo ####
def repeat(num_times=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(num_times):
                func(*args, **kwargs)
        return wrapper
    return decorator
####

if __name__ == "__main__":
    
    @repeat(num_times=3)
    def hello(name):
        print(f"hello {name}")
        
    hello("alice")
        
    
    @try_all_devices()
    def func1(x,*, y=4):
        print(f"in func1, x={x}, y={y}, x.device={x.device}, x.dtype={x.dtype}")
        if x.device == torch.device("cuda:4"):
            raise RuntimeError("CUDA out of memory fake")
        return x + y

    
    t1 = torch.Tensor([1,2]).to(device="cuda:4", dtype=torch.bfloat16)
    # t2 = torch.Tensor([3,4]).to("cuda:4")
    
    t3 = func1(t1, y=5)
    print(t3)
    print(t3.dtype)
    
    
    