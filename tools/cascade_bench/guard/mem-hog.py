import sys, time, threading

# A synthetic background workload for the --windows-memory-guard test.  Allocate `mb` MiB of anonymous
# memory, touch every page so it is charged as resident, then hold it while OPTIONALLY keeping it active.
# Run it beside the engine to drive free physical memory down.
#
#   python mem-hog.py 10240 120          # ramp: allocate 10 GiB in 256 MiB chunks, then idle
#   python mem-hog.py 10240 120 burst    # one 10 GiB allocation (a sudden app launch)
#   python mem-hog.py 16000 120 active   # 16 GiB AND keep touching it (a real background app: browser,
#                                        # IDE indexer, VM).  `active` is the realistic overflow - the
#                                        # pages stay hot, so the OS has to page the engine out to fit it.
#
# `burst` tests a sudden demand; `active` tests sustained pressure (the page-file-volume case).
mb = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
hold = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
mode = sys.argv[3] if len(sys.argv) > 3 else "chunk"
total = mb * 1024 * 1024
blocks = []
stop = threading.Event()


def churn():
    """Re-touch a rolling window of the allocation so its pages stay resident/active, like a live app."""
    step = 4096
    i = 0
    while not stop.is_set():
        for b in blocks:
            n = len(b)
            # touch ~64 MiB per pass, walking forward so all of it stays warm
            for off in range(i, min(i + (64 << 20), n), step):
                b[off] = (off & 0xFF) or 1
            i = (i + (64 << 20)) % max(n, 1)
        time.sleep(0.5)


try:
    if mode == "burst":
        b = bytearray(total)
        b[::4096] = b"\x01" * (total // 4096)   # touch one byte per 4 KiB page, all at once
        blocks.append(b)
        print(f"mem-hog: allocated {total // (1024 * 1024)} MiB in one burst", flush=True)
    else:
        chunk = 256 * 1024 * 1024
        for i in range(total // chunk):
            b = bytearray(chunk)
            b[::4096] = b"\x01" * (len(b) // 4096)
            blocks.append(b)
        print(f"mem-hog: allocated {len(blocks) * chunk // (1024 * 1024)} MiB", flush=True)

    if mode == "active":
        t = threading.Thread(target=churn, daemon=True)
        t.start()
        print("mem-hog: keeping it active (background-app case)", flush=True)

    time.sleep(hold)
finally:
    stop.set()
    blocks.clear()
    print("mem-hog: released", flush=True)
