"""Put back the spacing the detector guaranteed and post-processing threw away.

Greedy NMS at inference leaves a hard floor between detections in a frame — on
this run 7.0 um requested, 7.08 um observed, with *not one* pair below it. Two
stages then move nodes: ``recover`` inserts interpolated ones, and
``pp_public``'s linefit smoothing shifts existing ones. Neither knows about the
floor, and between them 14–21% of nodes end up with a neighbour inside 7 um and
about 1% inside 5 um, down to nodes sitting on top of each other.

That is the failure mode of the worst stacks. When one cell carries two
detections, the metric ties the annotation to one of them and the track may run
through the other; the link is then charged as a false positive at both ends
even though it is physically right. Node recall does not notice — both copies
are inside the 7 um matching radius — and the score loses twice.

Which copy to keep barely matters, and cannot be decided anyway: over the six
worst stacks the annotation's own match sits on the longer track only 54.7% of
the time, a coin flip. What matters is that **one** of them survives, because
then the matcher and the tracker cannot disagree. So this suppresses rather than
chooses, and rewires the loser's links onto the winner instead of dropping them,
which keeps both tracks whole.

**MEASURED, AND IT DOES NOT PAY — this is not in the pipeline.** Over all 71
stacks: -0.0002 at a 5 um radius (3 stacks better, 6 worse) and -0.068 at 7 um.
The reasoning above is sound and the target is wrong. What costs the score is
two detections within 7 um *of the annotation*, and those sit 7-14 um apart from
**each other** — exactly what a 7 um NMS is built to allow. A radius that
reaches them is a radius that deletes cells: 9-11 um drives the rival rate to
zero and node recall from ~99% to 41-78%. With cells 8.0 um apart and a 7 um
matching radius, every detection set that covers them has rivals inside it.

Kept because the measurement is worth more than the stage: see
postprocessing/README.md, *Cutting the rival density*.
"""

from __future__ import annotations

import numpy as np

from params import SCALE

SCALE_ARR = np.asarray(SCALE)

DEFAULTS = dict(
    radius=7.0,          # um; the detector's own NMS radius
    position="keep",     # "keep" the winner's coordinates, or their "centroid"
    max_parents=1,
    max_children=1,
)



