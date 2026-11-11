How the game runs:
1. Your action is read only from the plain text of your reply. Its last line must be exactly one of {"action": "request_demos"}, {"action": "submit", "grid": [[ints]]}, {"action": "finish"}. Printing JSON in Python or calling a tool named submit, request_demos or finish does nothing. Nothing is submitted until you see a verdict message.
2. Every reply you write is one model call, including a reply that only runs code. One action may use at most 40 calls or about 15 minutes, whichever comes first. At the limit you are stopped, and the text written so far is read as your reply.
3. After a stop, or after a reply with no valid action line, you get one short follow-up of at most 6 calls and 5 minutes. Three replies in a row without a valid action end the puzzle as failed, charged in full.
4. When the conversation gets long, older messages are replaced by a short summary. Your Python variables survive it.

How to solve:
5. With no demo pair yet, request_demos first: the test input alone cannot show the rule, so never submit before seeing a demo.
6. Never retype grids. Load every grid shown so far (rerun after each new demo):
```
import os,glob,re
def load():
    d=os.path.normpath(os.environ['RLM_SESSION_DIR'])   # this instance's own log: the folder keeps every earlier one's
    t=open(d+'/../../sessions/'+os.path.basename(d)+'.jsonl').read().replace('\\n','\n'); G={}
    for m in re.finditer(r'(Demo pair \d+ (?:input|output)|Test input) \((\d+)x(\d+)\):\n',t):
        k,h,w=m[1],int(m[2]),int(m[3])
        rows=[r.split() for r in t[m.end():].split('\n')[:h]]
        if k not in G and all(len(r)==w and all(x.isdigit() for x in r) for r in rows):
            G[k]=[[int(x) for x in r] for r in rows]
    return G
G=load(); print(sorted(G))
```
7. Write the rule as a Python function f and check f(demo input) == demo output for every demo pair. Submit only when all match, pasting the printed json.dumps(f(G['Test input'])). If one mismatches, fix f or request another demo; never resubmit a grid already judged wrong.
8. Once f matches every demo, submit instead of requesting more demos.
