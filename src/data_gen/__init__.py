"""Synthetic cell-tracking data generation.

    from data_gen import real_stats, generate_static, generate_sequence

The generator is calibrated against the real acquisition: cell counts and
tissue geometry are measured from real volumes (``real_stats``), and motion
from the real lineages. See ``notebooks/SyntheticData.ipynb`` for the pipeline
that builds a dataset with it.
"""

from data_gen.acquisition import Palette, Regime
from data_gen.calibrate import report_real, report_synth, print_table
from data_gen.constants import DOWNSAMPLE, EFF_VOXEL, POOL, VOXEL_NATIVE
from data_gen.geometry import fit_shell, sample_shell_positions
from data_gen.real_stats import find_train_dir, norm, pool_xy, read_frame, real_stats
from data_gen.render import gen_volume_native
from data_gen.sequences import generate_sequence, generate_static, to_uint16

__all__ = [
    "VOXEL_NATIVE", "POOL", "DOWNSAMPLE", "EFF_VOXEL",
    "Regime", "Palette",
    "fit_shell", "sample_shell_positions", "gen_volume_native",
    "real_stats", "find_train_dir", "read_frame", "norm", "pool_xy",
    "generate_static", "generate_sequence", "to_uint16",
    "report_real", "report_synth", "print_table",
]
