import sys
import time
import psutil
from pathlib import Path

# Add tools/server to path for DXGI query
tools_server = Path(__file__).resolve().parent / "server"
sys.path.insert(0, str(tools_server))
from service_benchmark import query_dxgi_video_memory

def monitor(process_name="Aila.exe", interval=1.0):
    print(f"[*] Monitoring process '{process_name}' (Ctrl+C to stop)...")
    print("Process WS excludes native large pages; system used/available includes their physical cost.")
    print(f"{'Time':8s} | {'PID':7s} | {'Process WS GiB':14s} | {'Private GiB':11s} | {'System used/free GiB':20s} | {'Adapter VRAM MiB':16s} | {'Budget MiB':10s}")
    print("-" * 75)

    try:
        while True:
            target_proc = None
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    if proc.info['name'] and proc.info['name'].lower() == process_name.lower():
                        target_proc = proc
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            cur_time = time.strftime("%H:%M:%S")
            physical = psutil.virtual_memory()
            used_gib = (physical.total - physical.available) / 2**30
            available_gib = physical.available / 2**30
            vram_stats = query_dxgi_video_memory()
            vram_usage_mb = (vram_stats['local_vram_usage_bytes'] / (1024 * 1024)) if vram_stats else 0
            vram_budget_mb = (vram_stats['local_vram_budget_bytes'] / (1024 * 1024)) if vram_stats else 0

            if target_proc:
                try:
                    mem = target_proc.memory_info()
                    ws_gb = mem.rss / (1024**3)
                    commit_gb = getattr(mem, "private", mem.vms) / (1024**3)
                    print(f"{cur_time:8s} | {target_proc.pid:<7d} | {ws_gb:14.2f} | {commit_gb:11.2f} | {used_gib:8.2f}/{available_gib:8.2f} | {vram_usage_mb:16.1f} | {vram_budget_mb:10.1f}", flush=True)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            else:
                if vram_usage_mb > 100:
                    print(f"{cur_time:8s} | (none)  |            --- |         --- | {used_gib:8.2f}/{available_gib:8.2f} | {vram_usage_mb:16.1f} | {vram_budget_mb:10.1f}", flush=True)

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[*] Monitor stopped.")

if __name__ == "__main__":
    proc_name = sys.argv[1] if len(sys.argv) > 1 else "Aila.exe"
    monitor(proc_name)
