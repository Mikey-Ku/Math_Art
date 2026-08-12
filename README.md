# Math_Art

An animated deep zoom into the Mandelbrot set. A Numba-compiled kernel computes
smooth escape-time values fast enough to render the zoom live rather than to a
video file.

**Status:** a single 90-line script, kept as a small standalone piece. It is not
a library and has no tests.

## Run it

```sh
pip install -r requirements.txt
python Mandelbrot.py
```

A window opens and zooms toward `(-0.743643887037151, 0.131825904205330)` over
250 frames, each one 0.95x the width of the last, ending near 3.7e5x
magnification. The title reports the current zoom.

## What is actually going on

Escape-time colouring, with the fractional part recovered so the bands come out
smooth instead of stepped. A point that escapes at iteration `n` is coloured
`n + 1 - log(log|z|)/log(2)` rather than just `n`, which is what stops the image
from looking like a contour map.

The whole thing is one 500x500 grid recomputed per frame. There is no
progressive refinement and no reuse between frames, because at this size it does
not need either.

### The parallelism is in the row loop

The kernel is `@jit(nopython=True, parallel=True, fastmath=True)` and the outer
loop over rows is `prange`. Both parts matter, and the second is easy to leave
out:

| kernel | ms/frame |
| --- | --- |
| `range`, `parallel=False` | 9.8 |
| `range`, `parallel=True` | 10.1 |
| `prange`, `parallel=True` | **3.6** |

Measured at 500x500, `max_iter=200`, five runs after warming the JIT. With a
plain `range`, `parallel=True` buys nothing: it auto-parallelises array
expressions, not explicit scalar loops, so the kernel compiles to the same
single-threaded code and pays a little setup cost on top. The first version of
this file had exactly that bug, with a comment claiming a speedup it was not
getting.

Rows are a safe axis to split on because each iteration writes only `result[i]`
and reads nothing another row writes.

## Known limits

- `matplotlib.use('MacOSX')` is hardcoded at the top, so it needs macOS as
  written. On another platform, drop that line and let matplotlib pick a
  backend.
- Zoom depth is bounded by float64. Past roughly 1e13 magnification the
  coordinates stop resolving and the image degrades into blocks. This run stops
  well short of that.
- `max_iter` is fixed at 200. Deeper zooms genuinely need more iterations to
  keep the detail, so the interesting structure thins out toward the end.
