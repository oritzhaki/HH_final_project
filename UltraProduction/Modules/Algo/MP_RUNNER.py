import multiprocessing
import os
import subprocess
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def worker(script_path, args):
    # Get the process ID
    pid = os.getpid()

    # Execute the Python script as a separate process with command-line arguments
    cmd = ['python3', script_path] + args
    subprocess.run(cmd)

    # Return the process ID
    return pid


def run(cell_path):
    # Determine the number of CPU cores to use
    num_cores = multiprocessing.cpu_count() - 2
    print(f"Num Of Cores: {num_cores}")

    # Create a list of tasks, one for each CPU core
    tasks = []
    for i in range(30):
        run_type = 'train'
        task = {'script_path': 'Modules/Algo/app_mp.py', 'args': ['--task', str(i), '--run_type', str(run_type), '--cell_path', str(cell_path)]}
        tasks.append(task)
        
    # Create a pool of workers
    pool = multiprocessing.Pool(processes=num_cores)

    # Assign a task and script path with arguments to each worker
    results = []
    for task in tasks:
        script_path = task['script_path']
        args = task['args']
        result = pool.apply_async(worker, (script_path, args))
        results.append(result)

    # Wait for all tasks to complete and get the results
    for result in results:
        pid = result.get()
        print(f"Worker with PID {pid} executed a Python script successfully!")
