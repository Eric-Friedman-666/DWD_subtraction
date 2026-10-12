"""Read-only GPU telemetry; total device use is distinguished from this process."""
import csv
import os
import subprocess
import threading
import time


class GPUUsageRecorder:
    def __init__(self,path):
        self.path,self.stop = path,threading.Event()
        self.thread = threading.Thread(target=self._record,daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self,*args):
        self.stop.set()
        self.thread.join(timeout=8)

    def _record(self):
        gpu = os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0]
        with open(self.path,'x',newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['unix_time','pid','device_utilization_percent','device_memory_MiB',
                             'process_memory_MiB','process_SM_percent'])
            while not self.stop.is_set():
                try:
                    text = subprocess.check_output(['nvidia-smi','-i',gpu,
                        '--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True,timeout=5)
                    utilization,memory = text.strip().split(',')
                    text = subprocess.check_output(['nvidia-smi','-i',gpu,
                        '--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=5)
                    process_memory = next((line.split(',')[1].strip() for line in text.splitlines()
                                           if line.split(',')[0].strip()==str(os.getpid())),'')
                    pmon = subprocess.check_output(['nvidia-smi','pmon','-i',gpu,'-s','u','-c','1'],text=True,timeout=5)
                    sm = next((line.split()[3] for line in pmon.splitlines()
                               if not line.startswith('#') and len(line.split())>3
                               and line.split()[1]==str(os.getpid())),'')
                    writer.writerow([time.time(),os.getpid(),utilization.strip(),memory.strip(),process_memory,sm])
                    stream.flush()
                except (OSError,subprocess.SubprocessError,ValueError):
                    pass
                self.stop.wait(2)
