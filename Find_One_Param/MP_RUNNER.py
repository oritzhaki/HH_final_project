import multiprocessing
import os
import psutil
import subprocess
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def worker(script_path, args):
    # Get the process ID and CPU core number of the current worker
    pid = os.getpid()
    core = psutil.Process(pid).cpu_affinity()[0]

    # Execute the Python script as a separate process with command-line arguments
    cmd = ['python', script_path] + args
    subprocess.run(cmd)

    # Return the process ID and CPU core number
    return (pid, core)

if __name__ == '__main__':
    # Determine the number of CPU cores to use
    num_cores = multiprocessing.cpu_count()
    print(f"Num Of Cores {num_cores}")

    # Create a list of tasks, one for each CPU core
    tasks = []
    for i in range(100):
        task = {'script_path': 'app_mp.py', 'args': ['--task', str(i)]}
        tasks.append(task)

    # Create a pool of workers
    pool = multiprocessing.Pool(processes=num_cores)

    # Set the CPU affinity for each worker process
    for i, worker_process in enumerate(pool._pool):
        core = i % num_cores
        psutil.Process(worker_process.pid).cpu_affinity([core])

    # Assign a task and script path with arguments to each worker
    results = []
    for task in tasks:
        script_path = task['script_path']
        args = task['args']
        result = pool.apply_async(worker, (script_path, args))
        results.append(result)

    # Wait for all tasks to complete and get the results
    for result in results:
        pid, core = result.get()
        print(f"Worker with PID {pid} on CPU core {core} executed a Python script Succesfully!")
