import platform
import os
import sys
import shutil

def print_system_info():
    print(f"Hostname:       {platform.node()}")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"CPU Count:      {os.cpu_count()}")
    mem_kb = int(open("/proc/meminfo").readline().split()[1])
    print(f"RAM Total:      {mem_kb / 1024 / 1024:.1f} GB")
    total, used, free = shutil.disk_usage("/home/equansa00/beast/workspace")
    print(f"Workspace disk: {total // 2**30} GB total, {free // 2**30} GB free")

print_system_info()
