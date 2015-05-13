__author__ = 'mikhail'

import sys

lineIn = sys.stdin.readline().split(" ")
n = int("11527523930876953")
m = int("26673")

fibPrev = 0
fib = 1
cached = [fibPrev, fib]

for curr in range(1, n):
    fibOld = fib
    fib = (fib + fibPrev) % m
    fibPrev = fibOld

    if fibPrev == 0 and fib == 1:
        cached.pop()
        break
    else:
        cached.append(fib)

offset = n % len(cached)
sys.stdout.write(str(cached[offset]))




