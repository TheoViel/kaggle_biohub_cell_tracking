"""Suppress tracks that shadow another track — the same cell detected twice.

The detector returns 38k nodes where the dataset estimates 24k, and non-maximum
suppression cannot tell the difference: the duplicates sit a full NMS radius
apart, because that is exactly what NMS leaves behind when a cell is bright
enough to peak twice. 47 of the 89 annotated cells have more than one detection
within the matching radius.

Over a single frame the two are indistinguishable. Over a track they are not:
a duplicate follows its original for the whole of its life, staying within a
cell's width the entire time. That is the signature this looks for, and it is
worth removing for two reasons — the node-count adjustment, and the fact that a
duplicate competes for the parent its original needed.
"""

from __future__ import annotations


import numpy as np

from params import SCALE

SCALE_ARR = np.asarray(SCALE)



