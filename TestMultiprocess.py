
from multiprocessing import Process
import multiprocessing
import time
import numpy as np


def processingLoop(q, inp):
    s_time = time.time()
    for i in range(3):
        print("Sub ", i)
        time.sleep(1)
    print("Sub Time", time.time() - s_time)
    q.put(np.array([inp, inp]))

def measureSingle(iteration):
    s_time = time.time()
    for i in range(5):
        print("Measure ", i)
        time.sleep(1)
    print("Measure Time", time.time() - s_time)
    return(iteration**2)

def multiprocessTest():
    q = multiprocessing.Queue()
    resultsArray = np.zeros((5,2))
    maxIteration = 5
    for i in range(maxIteration):
        inp = measureSingle(i)
        if (i != 0):
            resultsArray[i-1] = q.get()    # prints "[42, None, 'hello']"
            p.join()
        tempInp = inp
        p = Process(target=processingLoop, args=(q, tempInp))
        p.start()
    resultsArray[maxIteration-1] = q.get()    # prints "[42, None, 'hello']"
    p.join()
    return resultsArray


if __name__ == '__main__':
    start_time = time.time()
    results = multiprocessTest()
    print("Total Time", time.time() - start_time)
    print(results)

