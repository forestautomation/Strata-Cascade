"""bench_http.py - one 32K-prefill / N-decode HTTP bench against a running Strata server.

Called by tune_cascade.py:  python bench_http.py <port> <prompt-file> <max-tokens> <repeats>
Prints one "REQ n: ... prefill=<t/s> ... decode=<t/s> ..." line per repeat.
"""
import json, sys, time, urllib.request
from pathlib import Path

port = sys.argv[1]
prompt_path = sys.argv[2]
nmax = int(sys.argv[3])
reps = int(sys.argv[4])

prompt = Path(prompt_path).read_text(encoding="utf-8")
url = "http://127.0.0.1:%s/v1/chat/completions" % port

for i in range(1, reps + 1):
    body = json.dumps({
        "model": "strata",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": nmax,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=7200) as r:
        obj = json.loads(r.read())
    wall = time.time() - t0
    tm = obj.get("timings", {})
    print("REQ %d: wall=%.1fs prompt_n=%s cached=%s prefill=%s predicted_n=%s decode=%s drafts=%s accepted=%s finish=%s"
          % (i, wall, tm.get("prompt_n"), tm.get("cache_n"), tm.get("prompt_per_second"),
             tm.get("predicted_n"), tm.get("predicted_per_second"), tm.get("draft_n"),
             tm.get("draft_n_accepted"), obj["choices"][0].get("finish_reason")), flush=True)
