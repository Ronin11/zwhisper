#!/usr/bin/env python3
"""Compare two raw f32 logit dumps (zwhisper vs whisper.cpp reference)."""
import struct, sys, math

def load(p):
    b = open(p, "rb").read()
    return struct.unpack(f"<{len(b)//4}f", b)

a, b = load(sys.argv[1]), load(sys.argv[2])
assert len(a) == len(b), (len(a), len(b))
n = len(a)
diffs = [abs(x - y) for x, y in zip(a, b)]
mx = max(diffs)
mean = sum(diffs) / n
ia = max(range(n), key=lambda i: a[i])
ib = max(range(n), key=lambda i: b[i])
top_a = sorted(range(n), key=lambda i: -a[i])[:5]
top_b = sorted(range(n), key=lambda i: -b[i])[:5]
# correlation
ma, mb = sum(a)/n, sum(b)/n
cov = sum((x-ma)*(y-mb) for x, y in zip(a, b))
va = math.sqrt(sum((x-ma)**2 for x in a)); vb = math.sqrt(sum((y-mb)**2 for y in b))
print(f"n={n} max|d|={mx:.4f} mean|d|={mean:.5f} corr={cov/(va*vb):.6f}")
print(f"argmax ours={ia} ({a[ia]:.4f})  ref={ib} ({b[ib]:.4f})  {'OK' if ia == ib else 'MISMATCH'}")
print("top5 ours:", [(i, round(a[i], 3)) for i in top_a])
print("top5 ref: ", [(i, round(b[i], 3)) for i in top_b])
