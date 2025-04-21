import sys
import torch
from tqdm import tqdm
import argparse
import multiprocessing
import os
import time
import timm  # Added timm import
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet
from torchvision.datasets import ImageFolder
# from  example.test_vit import test_classification

# Remove get_net-related imports
sys.path.insert(0,'..')
sys.path.insert(0,'.')
import utils.datasets as datasets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=6)
    parser.add_argument("--multiprocess", action='store_true')
    return parser.parse_args()

# def test_classification(net, test_loader, max_iteration=None, description=None):
#     print("[INFO] Starting Classification Testing with Profiler...")
#     print(f"[INFO] Using device: {torch.cuda.get_device_name(0)}")

#     net.eval()
#     pos = 0
#     tot = 0
#     batch_count = 0

#     with profile(
#         activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#         on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
#         record_shapes=True,
#         with_stack=True,
#         profile_memory=True,
#         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
#     ) as prof, torch.no_grad():

#         q = tqdm(test_loader, desc=description)
#         for i, (inp, target) in enumerate(q):
#             inp = inp.cuda(non_blocking=True)
#             target = target.cuda(non_blocking=True)

#             with record_function("model_inference"):
#                 out = net(inp)

#             pos_num = torch.sum(out.argmax(1) == target).item()
#             pos += pos_num
#             tot += inp.size(0)

#             q.set_postfix({"acc": pos / tot})
#             batch_count += 1

#             prof.step()
#             if max_iteration and i >= max_iteration:
#                 break

#     accuracy = pos / tot
#     print(f"\n[RESULTS] Accuracy: {accuracy:.4f}")
#     print(f"[INFO] Profiler trace written to ./log/profiler/")
#     print(f"➡ Run `tensorboard --logdir=./log/profiler` to visualize.")

#     return accuracy


# (Keep process/multiprocess functions unchanged)
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


def experiment_basic(net='vit_base_patch16_224'):
    """Simplified testbench without quantization or custom get_net"""
    print(f"[INFO] Running Baseline Experiment: Net={net}")
    
    # Directly load model using timm
    model = timm.create_model(net, pretrained=True).cuda()
    # Evaluate original model
    g = datasets.ViTImageNetLoaderGenerator('/content/datasets/imagenet', 'imagenet', 32, 32, 2, kwargs={"model": model})
    
    test_loader = g.test_loader()
    test_classification(model, test_loader)

if __name__ == '__main__':
    names = [
        # "vit_tiny_patch16_224",
        "vit_small_patch32_224",
        # "vit_small_patch16_224",
        # "vit_base_patch16_224",
        # "vit_base_patch16_384",

        # "deit_tiny_patch16_224",
        "deit_small_patch16_224",
        # "deit_base_patch16_224",
        # "deit_base_patch16_384",

        # "swin_tiny_patch4_window7_224",
        "swin_small_patch4_window7_224",
        # "swin_base_patch4_window7_224",
        # "swin_base_patch4_window12_384",
        ]
    for model_name in names:
        print(f"\n\n=========================")
        print(f"Running test on: {model_name}")
        print(f"=========================")
        try:
            experiment_basic(net=model_name)
        except Exception as e:
            print(f"[ERROR] Failed to run {model_name}: {e}")

