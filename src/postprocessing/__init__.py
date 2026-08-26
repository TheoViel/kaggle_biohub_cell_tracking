"""Post-processing for the tracker's saved predictions.

The modules here import each other by bare name (``from relink import relink``)
so that every script in this directory can also be run directly. Importing the
package the ordinary way — ``from postprocessing.pipeline import postprocess``,
as the notebook would — therefore needs this directory on the path too, which is
what the line below does.

Typical use:

    from postprocessing.pipeline import postprocess
    coords, edges = postprocess(coords, edges)

or, for the whole tuned chain including tracking.pp_public, see
``src/main_postprocess.py`` and ``postprocess_npz.py``.
"""

import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
