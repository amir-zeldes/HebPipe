import atexit
import sys
from time import process_time
from functools import reduce

def secondsToStr(t):
    return "%d:%02d:%02d.%03d" % \
        reduce(lambda ll,b : divmod(ll[0],b) + ll[1:],
            [(t*1000,),1000,60,60])

line = "="*40
def log(s, elapsed=None):
    #print line
    #print secondsToStr(process_time()), '-', s
    if elapsed:
        sys.stderr.write("Elapsed time: " + str(elapsed) + "\n")
    sys.stderr.write(line +"\n")

def endlog():
    end = process_time()
    elapsed = end-start
    log("End Program", secondsToStr(elapsed))

def now():
    return secondsToStr(process_time())

start = time.time()

atexit.register(endlog)
#log("Start Program")
