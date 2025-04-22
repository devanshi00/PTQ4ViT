import sys
sys.path.insert(0,'..')
sys.path.insert(0,'.')
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
from importlib import reload, import_module
import multiprocessing
import os
import time
from itertools import product
import traceback
# from get_int import *
import utils.integer as integer
import utils.datasets as datasets
import utils.net_wrap as net_wrap
from utils.quant_calib import QuantCalibrator, HessianQuantCalibrator
from utils.models import get_net

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=6)
    parser.add_argument("--multiprocess", action='store_true')
    args = parser.parse_args()
    return args

# def test_classification(net, test_loader, max_iteration=None, description=None):
#     print("[INFO] Starting Classification Testing...")
#     pos = 0
#     tot = 0
#     i = 0
#     max_iteration = len(test_loader) if max_iteration is None else max_iteration
#     with torch.no_grad():
#         q = tqdm(test_loader, desc=description)
#         for inp, target in q:
#             i += 1
#             inp = inp.cuda()
#             target = target.cuda()
#             print(f"[DEBUG] Batch {i}: Input Shape: {inp.shape}, Target Shape: {target.shape}")
            
#             out = net(inp)
#             print(f"[DEBUG] Batch {i}: Output Shape: {out.shape}, Output Sample: {out[0][:5]}")  # Print first few logits
            
#             pos_num = torch.sum(out.argmax(1) == target).item()
#             pos += pos_num
#             tot += inp.size(0)
#             q.set_postfix({"acc": pos / tot})
            
#             if i >= max_iteration:
#                 break

#     accuracy = pos / tot
#     print(f"[INFO] Classification Test Completed. Accuracy: {accuracy:.4f}")
    # return accuracy
# import torch
# import torch.profiler
# from tqdm import tqdm

# import time  # For wall-clock timing
def test_classification(net, test_loader, max_iteration=None, description=None):
    print("[INFO] Starting Classification Testing...")
    pos = 0
    tot = 0
    i = 0
    cuda_total_time = 0.0  # Total inference time on GPU (ms)
    wall_total_time = 0.0  # Total wall-clock time (ms)
    max_iteration = len(test_loader) if max_iteration is None else max_iteration

    # Track memory
    max_memory_allocated = 0  # in bytes

    with torch.no_grad():
        q = tqdm(test_loader, desc=description)
        for inp, target in q:
            i += 1
            inp = inp.cuda()
            target = target.cuda()

            torch.cuda.reset_peak_memory_stats()

            # Wall-clock start
            wall_start = time.perf_counter()

            # CUDA timing start
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            out = net(inp)
            end_event.record()
            torch.cuda.synchronize()

            # Wall-clock end
            wall_end = time.perf_counter()

            # Time measurements
            batch_cuda_time = start_event.elapsed_time(end_event)  # in ms
            batch_wall_time = (wall_end - wall_start) * 1000  # in ms

            cuda_total_time += batch_cuda_time
            wall_total_time += batch_wall_time

            # Accuracy tracking
            pos_num = torch.sum(out.argmax(1) == target).item()
            pos += pos_num
            tot += inp.size(0)
            q.set_postfix({"acc": pos / tot})

            # Memory tracking
            current_max_memory = torch.cuda.max_memory_allocated()
            max_memory_allocated = max(max_memory_allocated, current_max_memory)

            if i >= max_iteration:
                break

    # Convert memory to MB
    max_memory_allocated_MB = max_memory_allocated / (1024 ** 2)

    # Timing metrics
    avg_batch_cuda_time = cuda_total_time / i
    avg_sample_cuda_time = cuda_total_time / tot
    avg_batch_wall_time = wall_total_time / i
    avg_sample_wall_time = wall_total_time / tot
    accuracy = pos / tot

    # Final Output
    print(f"[INFO] Classification Test Completed")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"[CUDA Timing]")
    print(f"  Avg batch time: {avg_batch_cuda_time:.2f}ms")
    print(f"  Avg sample time: {avg_sample_cuda_time:.2f}ms")
    print(f"[Wall-Clock Timing]")
    print(f"  Avg batch time: {avg_batch_wall_time:.2f}ms")
    print(f"  Avg sample time: {avg_sample_wall_time:.2f}ms")
    print(f"  Total inference wall time: {wall_total_time:.2f}ms")
    print(f"[GPU Memory Usage]")
    print(f"  Peak memory allocated: {max_memory_allocated_MB:.2f} MB")

    return accuracy


# def test_classification(net, test_loader, max_iteration=None, description=None):
#     print("[INFO] Starting Classification Testing...")
#     pos = 0
#     tot = 0
#     i = 0
#     cuda_total_time = 0.0  # Total inference time on GPU (ms)
#     wall_total_time = 0.0  # Total wall-clock time (s)
#     max_iteration = len(test_loader) if max_iteration is None else max_iteration

#     with torch.no_grad():
#         q = tqdm(test_loader, desc=description)
#         for inp, target in q:
#             i += 1
#             inp = inp.cuda()
#             target = target.cuda()

#             # Wall-clock start
#             wall_start = time.perf_counter()

#             # CUDA timing start
#             start_event = torch.cuda.Event(enable_timing=True)
#             end_event = torch.cuda.Event(enable_timing=True)

#             start_event.record()
#             out = net(inp)
#             end_event.record()
#             torch.cuda.synchronize()

#             # Wall-clock end
#             wall_end = time.perf_counter()

#             # Time measurements
#             batch_cuda_time = start_event.elapsed_time(end_event)  # in ms
#             batch_wall_time = (wall_end - wall_start) * 1000  # convert to ms

#             cuda_total_time += batch_cuda_time
#             wall_total_time += batch_wall_time

#             pos_num = torch.sum(out.argmax(1) == target).item()
#             pos += pos_num
#             tot += inp.size(0)
#             q.set_postfix({"acc": pos / tot})

#             if i >= max_iteration:
#                 break

#     # Calculate timing metrics
#     avg_batch_cuda_time = cuda_total_time / i
#     avg_sample_cuda_time = cuda_total_time / tot

#     avg_batch_wall_time = wall_total_time / i
#     avg_sample_wall_time = wall_total_time / tot

#     accuracy = pos / tot

#     print(f"[INFO] Classification Test Completed")
#     print(f"Accuracy: {accuracy:.4f}")
#     print(f"[CUDA Timing]")
#     print(f"  Avg batch time: {avg_batch_cuda_time:.2f}ms")
#     print(f"  Avg sample time: {avg_sample_cuda_time:.2f}ms")
#     print(f"[Wall-Clock Timing]")
#     print(f"  Avg batch time: {avg_batch_wall_time:.2f}ms")
#     print(f"  Avg sample time: {avg_sample_wall_time:.2f}ms")
#     print(f"  Total inference wall time: {wall_total_time:.2f}ms")

#     return accuracy




def process(pid, experiment_process, args_queue, n_gpu):
    """
    Worker process. 
    """
    gpu_id = pid % n_gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu_id}'
    tot_run = 0

    while args_queue.qsize():
        test_args = args_queue.get()
        print(f"[INFO] Run {test_args} on pid={pid} gpu_id={gpu_id}")
        try:
            experiment_process(**test_args)
        except Exception as e:
            print(f"[ERROR] Process {pid} encountered an error: {str(e)}")
            traceback.print_exc()

        time.sleep(0.5)
        tot_run += 1

    print(f"[INFO] Process {pid} completed {tot_run} runs.")

def multiprocess(experiment_process, cfg_list=None, n_gpu=6):
    """
    Run experiment processes on "n_gpu" cards via "n_gpu" worker processes.
    """
    args_queue = multiprocessing.Queue()
    for cfg in cfg_list:
        args_queue.put(cfg)

    ps = []
    for pid in range(n_gpu):
        p = multiprocessing.Process(target=process, args=(pid, experiment_process, args_queue, n_gpu))
        p.start()
        ps.append(p)
    for p in ps:
        p.join()

def init_config(config_name):
    """Initialize the config. Use reload to make sure it's a fresh one!"""
    print(f"[INFO] Loading Configuration: {config_name}")
    
    _, _, files = next(os.walk("./configs"))
    if config_name + ".py" in files:
        quant_cfg = import_module(f"configs.{config_name}")
    else:
        raise NotImplementedError(f"[ERROR] Invalid config name {config_name}")

    reload(quant_cfg)
    print(f"[INFO] Configuration {config_name} Loaded Successfully.")
    return quant_cfg

def experiment_basic(net='vit_base_patch16_224', config="PTQ4ViT"):
    """
    A basic testbench.
    """
    print(f"[INFO] Running Experiment: Net={net}, Config={config}")
    name=net
    quant_cfg = init_config(config)
    net = get_net(net).cuda()
    # print(f"[INFO] Model {net} Loaded Successfully.")

    wrapped_modules = net_wrap.wrap_modules_in_net(net, quant_cfg)
    # print(f"[INFO] Modules Wrapped: {wrapped_modules.keys()}")

    g = datasets.ViTImageNetLoaderGenerator('/content/datasets/imagenet', 'imagenet', 32, 32, 2, kwargs={"model": net})
    
    test_loader = g.test_loader()
    calib_loader = g.calib_loader(num=32)
    print(f"[INFO] Dataloaders Initialized: Test Loader Size={len(test_loader)}, Calibration Loader Size={len(calib_loader)}")

    quant_calibrator = HessianQuantCalibrator(net, wrapped_modules, calib_loader, sequential=False, batch_size=32)
    print("[INFO] Starting Quantization Calibration...")
    
    quant_calibrator.batching_quant_calib()
    print("[INFO] Quantization Calibration Completed.")

    test_classification(net, test_loader)
    # int_weights = integer.get_model_int_weight(wrapped_modules)
    # torch.save(int_weights, f"./{name}.pth")
# def get_int_weights(name, config_name):
#     quant_cfg = init_config(config_name)

#     net = get_net(name)

#     wrapped_modules=net_wrap.wrap_modules_in_net(net,quant_cfg)
    
#     g=datasets.ViTImageNetLoaderGenerator('./datasets/imagenet','imagenet',32,32,16, kwargs={"model":net})
#     test_loader=g.test_loader()
#     calib_loader=g.calib_loader(num=32)
    
#     quant_calibrator = HessianQuantCalibrator(net,wrapped_modules,calib_loader,sequential=False,batch_size=4) # 16 is too big for ViT-L-16
#     quant_calibrator.batching_quant_calib()

#     int_weights = integer.get_model_int_weight(wrapped_modules)
#     torch.save(int_weights, f"./int_weights/{name}.pth")

if __name__ == '__main__':
    args = parse_args()
    # cfg_list = []

    nets = [ 
      # "vit_tiny_patch16_224",
        # "vit_small_patch32_224",
        # "vit_small_patch16_224",
        # "vit_base_patch16_224",
        # "vit_base_patch16_384",

        # "deit_tiny_patch16_224",
        # "deit_small_patch16_224",
        # "deit_base_patch16_224",
        # "deit_base_patch16_384",
        "cait_s24_224",

        # "swin_tiny_patch4_window7_224",
        "swin_small_patch4_window7_224",
        # "swin_base_patch4_window7_224",
        # "swin_base_patch4_window12_384",
        ]
    configs = ['PTQ4ViT']

    cfg_list = [{
        "net": net,
        "config": config,
    } for net, config in product(nets, configs)]

    if args.multiprocess:
        print("[INFO] Running in Multiprocessing Mode.")
        multiprocess(experiment_basic, cfg_list, n_gpu=args.n_gpu)
    else:
        print("[INFO] Running in Single Process Mode.")
        for cfg in cfg_list:
            experiment_basic(**cfg)
