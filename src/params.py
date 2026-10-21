import os

# Relative to src/, where every entry point runs. Overridable to run the whole
# pipeline on another data copy or log root (see train.sh, SMOKE=1).
INPUT_FOLDER = os.environ.get("BIOHUB_INPUT", "../input/")
LOG_FOLDER = os.environ.get("BIOHUB_LOGS", "../logs/")
OUTPUT_FOLDER = "../output/"

IMG_FOLDER = INPUT_FOLDER + "processed/images/"
NODES_FOLDER = INPUT_FOLDER + "processed/nodes/"
EDGES_FOLDER = INPUT_FOLDER + "processed/edges/"

TRAIN_DIR = INPUT_FOLDER + "train/"
TEST_DIR  = INPUT_FOLDER + "test/"

RESOLUTION = {"z": 1.625, "y": 0.40625, "x": 0.40625}
SCALE = (1.625, 0.40625, 0.40625)


EMBRYOS = ['44b6', '6bba']
