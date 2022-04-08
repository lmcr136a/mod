import multiprocessing as m
import time

def function(x, y):
    for i in range(10**8):
        y += 1
        x += 1
    return [1,2,3]

def merge_pool(args):
    x, y = args[0], args[1]
    return function(x, y)
    
p = m.Pool(16)   
start = time.time() 
print(p.map(merge_pool, [[1,1], [2,1]]))
print(time.time()- start)



start = time.time() 
print(function(1, 1))
print(function(2, 1))
print(time.time()- start)
