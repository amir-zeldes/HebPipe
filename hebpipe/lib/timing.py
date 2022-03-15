import atexit
import sys
from functools import reduce
try:
    from time import clock
except ImportError:
    try:
        from time import process_time as clock
    except ImportError:
        from time import perf_counter as clock


def secondsToStr(t):
    return "%d:%02d:%02d.%03d" % \
        reduce(lambda ll,b : divmod(ll[0],b) + ll[1:],
            [(t*1000,),1000,60,60])

line = "="*40
def log(s, elapsed=None):
    #print line
    #print secondsToStr(clock()), '-', s
    if elapsed:
        sys.stderr.write("Elapsed time: " + str(elapsed) + "\n")
    sys.stderr.write(line +"\n")

def endlog():
    end = clock()
    elapsed = end-start
    log("End Program", secondsToStr(elapsed))

def now():
    return secondsToStr(clock())

start = clock()

atexit.register(endlog)
#log("Start Program")