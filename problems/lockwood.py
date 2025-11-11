import numpy as np
import subprocess

RUNLOCK_PATH = '/lockwood/runlock/'


def runlock(rates=None, weights=[1, 1, 1], thread_id=0):
    if rates is None:
        rates = np.repeat(10000, 6)  # default value

    if rates.ndim > 1 and rates.shape[0] != 1:
        raise ValueError("rates should be a vector")

    # Write input file
    with open(RUNLOCK_PATH + "input"+str(thread_id)+".tst", "w") as f:
        f.write("6\n")
        for r in rates.flatten():
            f.write(f"{r}\n")

    # Run external process
    subprocess.run(["./RunLock", "input"+str(thread_id)+".tst", "output"+str(thread_id)+".tst", str(thread_id)], cwd=RUNLOCK_PATH, check=True)

    # Read output
    with open(RUNLOCK_PATH + "output"+str(thread_id)+".tst", "r") as f:
        output = [float(line.strip()) for line in f if line.strip()]

    return sum([o * w for o, w in zip(output, weights)]) # we consider the weighted sum: the original objective + l1 c_1(x) + l2 c_2(x)



def runlock_splitted(rates=None, thread_id=0):
    if rates is None:
        rates = np.repeat(10000, 6)  # default value

    if rates.ndim > 1 and rates.shape[0] != 1:
        raise ValueError("rates should be a vector")

    # Write input file
    with open(RUNLOCK_PATH + "input"+str(thread_id)+".tst", "w") as f:
        f.write("6\n")
        for r in rates.flatten():
            f.write(f"{r}\n")

    # Run external process
    subprocess.run(["./RunLock", "input"+str(thread_id)+".tst", "output"+str(thread_id)+".tst", str(thread_id)], cwd=RUNLOCK_PATH, check=True)

    # Read output
    with open(RUNLOCK_PATH + "output"+str(thread_id)+".tst", "r") as f:
        output = [float(line.strip()) for line in f if line.strip()]

    return output[0], output[1], output[2] 
