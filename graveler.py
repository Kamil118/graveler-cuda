from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import numpy as np
from datetime import datetime, timedelta

start_time = datetime.now()

ROLLCOUNT = 231
SUCCESS = 177 
REPEAT = 10000

@cuda.jit
def generate_numbers(rng_state, rollcount, out):
    thread_id = cuda.grid(1)
    fails = 0
    for i in range(rollcount):
        num = xoroshiro128p_uniform_float32(rng_state, thread_id)
        if num < 0.25:
            out[thread_id]+=1
        else:
            fails += 1
            if(fails > 54):
                return

#Seems to be optimal for T4 TPU
blocks = 4096
threads_per_block = 1024

total_threads = blocks * threads_per_block

print("runs per iteration: ", total_threads)
print("total runs: ", total_threads * REPEAT)

rng_states = create_xoroshiro128p_states(total_threads, seed=1)
out = np.zeros(total_threads, dtype=np.float32)

max_found = 0
successes = 0

for i in range(REPEAT):
  if i%1000 == 0:
    print(f"progress: {i} out of {REPEAT}")
    print("elapsed:", datetime.now() - start_time)
  generate_numbers[blocks, threads_per_block](rng_states, ROLLCOUNT, out)
  successes += (out >= SUCCESS).sum()
  if out.max() > max_found:
    max_found = out.max()
  out.fill(0)
  
print("Total Elapsed: ", datetime.now() - start_time)
print("Runs per second: ", total_threads * REPEAT/(datetime.now() - start_time).total_seconds())
print(f"Maximum paralize count: {max_found}\n\
Number of successes: {successes}")

