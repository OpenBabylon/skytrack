"""Optional system‑metric helpers (GPU utilisation, memory, CPU)."""
import wandb, psutil, time, threading, torch

def _gpu_stats():
    if not torch.cuda.is_available():
        return {}
    util = torch.cuda.utilization()
    mem  = torch.cuda.memory_allocated() / 2**20
    return {"gpu/util": util, "gpu/mem_MB": mem}

def _sys_stats():
    return {
        "cpu/%": psutil.cpu_percent(),
        "ram/GB": psutil.virtual_memory().used / 2**30,
    }

def start_background(interval: float = 30.0):
    """Start a daemon thread that logs system stats every *interval* sec."""
    def loop():
        while True:
            stats = {}
            stats.update(_gpu_stats())
            stats.update(_sys_stats())
            if stats:
                wandb.log(stats)
            time.sleep(interval)
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t
