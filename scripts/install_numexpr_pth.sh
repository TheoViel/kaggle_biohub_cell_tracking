#!/usr/bin/env bash
# Silence numexpr's startup line by capping its thread count before it imports.
#
# This machine reports 208 cores and numexpr is built with MAX_THREADS = 64, so
# at import it prints
#
#     Error.  nthreads cannot be larger than environment variable
#     "NUMEXPR_MAX_THREADS" (64)
#
# on **stdout** -- not stderr, not through `warnings`, so it cannot be filtered
# or caught. It lands on the first line of every run, concatenated with the
# program's own first line, which has swallowed real output more than once.
# pandas pulls numexpr in, so nearly everything here triggers it.
#
# A `.pth` and not `sitecustomize.py`: this image ships its own
# /usr/lib/python3.12/sitecustomize.py, and /usr/lib/python3.12 precedes
# site-packages on sys.path, so ours would never be imported. `site` executes
# `import ...` lines in .pth files, and does it before sitecustomize runs.
#
# NUMEXPR_NUM_THREADS=1 rather than raising the cap to 208: the post-processing
# runs 26 worker processes and already calls torch.set_num_threads(1) for the
# same reason. `setdefault`, so an explicit value still wins.
#
# Re-run after rebuilding the venv -- site-packages is not tracked.
set -euo pipefail
cd "$(dirname "$0")/.."
SP=$(.venv/bin/python -c "import site; print(site.getsitepackages()[0])")
cp scripts/numexpr_threads.pth "$SP/numexpr_threads.pth"
echo "installed -> $SP/numexpr_threads.pth"
out=$(.venv/bin/python -c "import pandas, tracksdata" 2>&1 || true)
if [ -z "$out" ]; then echo "verified: importing pandas + tracksdata is silent"
else echo "STILL NOISY:"; printf '%s\n' "$out"; exit 1; fi
