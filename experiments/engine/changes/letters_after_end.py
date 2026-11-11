"""teach.py's letters with each round's call started five seconds after the round before's call ended, rather than ten
seconds after its reply began: a trial of whether Z.AI reads the shared prefix of calls made one after another."""
import sys
import threading

sys.path.insert(0, sys.argv.pop(1))   # continual-arc-training's tree, for teach.py
import teach

call = teach.stream_call


def after_end(body, key, record, started, sse):
    return call(body, key, record, threading.Event(), sse)   # the gate is set only as the call ends (write's finally)


teach.stream_call, teach.LAG = after_end, 5
teach.main()
