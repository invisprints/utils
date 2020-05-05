# -*- coding: utf-8 -*-
"Utility functions to help deal with user environment"
import os
import subprocess
import re
from importlib import import_module
import sys
import platform
import torch
import psutil

# __all__ = ['show_env']

def get_env(name):
    "Return env var value if it's defined and not an empty string, or return Unknown"
    res = os.environ.get(name,'')
    return res if len(res) else "Unknown"

def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor

def show_env(show_nvidia_smi:bool=False):
    "Print user's setup information"

    rep = []
    opt_mods = []

    rep.append(["=== Software ===", None])
    rep.append(["python", platform.python_version()])
    rep.append(["torch",  torch.__version__])

    try:
        import pip
        info = pip.__version__
        # print('Version      :', pip.__version__)
        # print('Directory    :', os.path.dirname(pip.__file__))
    except ImportError:
        info = "No pip install"
    rep.append(["Pip", info])

    # nvidia-smi
    cmd = "nvidia-smi"
    have_nvidia_smi = False
    try: result = subprocess.run(cmd.split(), shell=False, check=False, stdout=subprocess.PIPE)
    except: pass
    else:
        if result.returncode == 0 and result.stdout: have_nvidia_smi = True

    # XXX: if nvidia-smi is not available, another check could be:
    # /proc/driver/nvidia/version on most systems, since it's the
    # currently active version

    if have_nvidia_smi:
        smi = result.stdout.decode('utf-8')
        # matching: "Driver Version: 396.44"
        match = re.findall(r'Driver Version: +(\d+\.\d+)', smi)
        if match: rep.append(["nvidia driver", match[0]])

    available = "available" if torch.cuda.is_available() else "**Not available** "
    rep.append(["torch cuda", f"{torch.version.cuda} / is {available}"])

    # no point reporting on cudnn if cuda is not available, as it
    # seems to be enabled at times even on cpu-only setups
    if torch.cuda.is_available():
        enabled = "enabled" if torch.backends.cudnn.enabled else "**Not enabled** "
        rep.append(["torch cudnn", f"{torch.backends.cudnn.version()} / is {enabled}"])

    rep.append(["\n=== Hardware ===", None])

    # it's possible that torch might not see what nvidia-smi sees?
    gpu_total_mem = []
    nvidia_gpu_cnt = 0
    if have_nvidia_smi:
        try:
            cmd = "nvidia-smi --query-gpu=memory.total --format=csv,nounits,noheader"
            result = subprocess.run(cmd.split(), shell=False, check=False, stdout=subprocess.PIPE)
        except:
            print("have nvidia-smi, but failed to query it")
        else:
            if result.returncode == 0 and result.stdout:
                output = result.stdout.decode('utf-8')
                gpu_total_mem = [int(x) for x in output.strip().split('\n')]
                nvidia_gpu_cnt = len(gpu_total_mem)


    if nvidia_gpu_cnt: rep.append(["nvidia gpus", nvidia_gpu_cnt])

    torch_gpu_cnt = torch.cuda.device_count()
    if torch_gpu_cnt:
        rep.append(["torch devices", torch_gpu_cnt])
        # information for each gpu
        for i in range(torch_gpu_cnt):
            rep.append([f"  - gpu{i}", (f"{gpu_total_mem[i]}MB | " if gpu_total_mem else "") + torch.cuda.get_device_name(i)])
    else:
        if nvidia_gpu_cnt:
            rep.append([f"Have {nvidia_gpu_cnt} GPU(s), but torch can't use them (check nvidia driver)", None])
        else:
            rep.append([f"No GPUs available", None])

    rep.append(['machine', platform.machine()])
    rep.append(['processor', platform.processor()])
    rep.append(["==CPU Info==", None])
    if sys.platform.startswith('darwin'):
        rep.append(["CPU Name", subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode('utf-8')])
    elif sys.platform.startswith('linux'):
        rep.append(["CPU Name", subprocess.call(['lscpu'])])
    elif sys.platform.startswith('win32'):
        rep.append(["CPU Name", subprocess.call(['wmic', 'cpu', 'get', 'name'])])

    # number of cores
    rep.append(["Physical cores:", psutil.cpu_count(logical=False)])
    rep.append(["Total cores:", psutil.cpu_count(logical=True)])
    # CPU usage
    rep.append(["CPU Usage Per Core:", None])
    for i, percentage in enumerate(psutil.cpu_percent(percpu=True, interval=1)):
        rep.append([f"Core {i}: {percentage}%", None])
    rep.append([f"Total CPU Usage: {psutil.cpu_percent()}%", None])
    # get the memory details
    rep.append(["==Memory Info==", None])
    svmem = psutil.virtual_memory()
    rep.append(["Total:", get_size(svmem.total)])
    rep.append(["Available:", get_size(svmem.available)])
    rep.append(["Used:", get_size(svmem.used)])
    rep.append(["Percentage: %", svmem.percent])
    rep.append(["=SWAP=", None])
    # get the swap memory details (if exists)
    swap = psutil.swap_memory()
    rep.append(["Total:", get_size(swap.total)])
    rep.append(["Free:", get_size(swap.free)])
    rep.append(["Used:", get_size(swap.used)])
    rep.append([f"Percentage: {swap.percent}%", None])

    rep.append(["\n=== Environment ===", None])

    rep.append(["platform", platform.platform()])
    rep.append(["Node Name", platform.node()])

    if platform.system() == 'Linux':
        distro = import_module('distro')
        if distro:
            # full distro info
            rep.append(["distro", ' '.join(distro.linux_distribution())])
        else:
            opt_mods.append('distro')
            # partial distro info
            rep.append(["distro", platform.uname().version])

    rep.append(["conda env", get_env('CONDA_DEFAULT_ENV')])
    rep.append(["python", sys.executable])
    rep.append(["sys.path", "\n".join(sys.path)])

    print("\n\n```text")

    keylen = max([len(e[0]) for e in rep if e[1] is not None])
    for e in rep:
        print(f"{e[0]:{keylen}}", (f": {e[1]}" if e[1] is not None else ""))

    if have_nvidia_smi:
        if show_nvidia_smi: print(f"\n{smi}")
    else:
        if torch_gpu_cnt: print("no nvidia-smi is found")
        else: print("no supported gpus found on this system")

    print("```\n")

    print("Please make sure to include opening/closing ``` when you paste into forums/github to make the reports appear formatted as code sections.\n")

    if opt_mods:
        print("Optional package(s) to enhance the diagnostics can be installed with:")
        print(f"pip install {' '.join(opt_mods)}")
        print("Once installed, re-run this utility to get the additional information")




if __name__ == "__main__":
    show_env()