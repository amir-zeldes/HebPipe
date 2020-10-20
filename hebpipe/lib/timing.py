import atexit
import sys
from time import time
from functools import reduce

def secondsToStr(t):
    return "%d:%02d:%02d.%03d" % \
        reduce(lambda ll,b : divmod(ll[0],b) + ll[1:],
            [(t*1000,),1000,60,60])

def log(s, elapsed=None):
    line = "="*40
    if elapsed:
        sys.stderr.write("Elapsed time: " + str(elapsed) + "\n")
    sys.stderr.write(line +"\n")

def endlog():
    end = time()
    print(start,end)
    elapsed = end-start
    log("End Program", secondsToStr(elapsed))

def now():
    return secondsToStr(time())

start = time()

atexit.register(endlog)
#log("Start Program")
