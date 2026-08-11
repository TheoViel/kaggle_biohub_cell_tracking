import os
import re
import sys
import copy
import math
import json
import shutil
import datetime
import subprocess
import numpy as np

try:
    import neptune
except ImportError:
    pass


class Config:
    """
    Placeholder to load a config from a saved json.
    """

    def __init__(self, dic):
        """
        Initialize the Config object from a dictionary.

        Args:
            dic (dict): Dictionary containing configuration parameters.
        """
        for k, v in dic.items():
            setattr(self, k, v)


class Logger(object):
    """
    A logger class that writes output to multiple file streams simultaneously.

    This class implements a custom logger that can write the same output to multiple
    file streams, ensuring that all output is properly flushed to disk.

    Attributes:
        files (tuple): Collection of file objects to write to.

    Methods:
        write(obj): Writes the given object to all file streams.
        flush(): Flushes all file streams.
        close(): Closes all file streams.
    """

    def __init__(self, *files):
        """
        Initialize the logger with multiple file streams.

        Args:
            *files: Variable number of file objects to write to.
        """
        self.files = files

    def write(self, obj):
        """
        Write the given object to all file streams.

        Args:
            obj: The object to write to the file streams.
        """
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        """
        Flush all file streams to ensure data is written to disk.
        """
        for f in self.files:
            f.flush()

    def close(self):
        """
        Close all file streams.
        """
        pass


def create_logger(directory="", name="logs.txt"):
    """
    Creates a logger to log output in a chosen file

    Args:
        directory (str, optional): Path to save logs at. Defaults to "".
        name (str, optional): Name of the file to save the logs in. Defaults to "logs.txt".
    """
    log = open(directory + name, "a", encoding="utf-8")
    file_logger = Logger(sys.stdout, log)

    sys.stdout = file_logger
    sys.stderr = file_logger


def prepare_log_folder(log_path):
    """
    Creates the directory for logging.
    Logs will be saved at log_path/date_of_day/exp_id.

    Args:
        log_path (str): Directory
    Returns:
        str: Path to the created log folder
    """
    today = str(datetime.date.today())
    log_today = f"{log_path}{today}/"

    if not os.path.exists(log_today):
        os.mkdir(log_today)

    exps = []
    for f in os.listdir(log_today):
        try:
            exps.append(int(f))
        except Exception:
            continue
    exp_id = np.max(exps) + 1 if len(exps) else 0

    log_folder = log_today + f"{exp_id}/"

    assert not os.path.exists(log_folder), "Experiment already exists"
    os.mkdir(log_folder)

    return log_folder


def get_last_log_folder(log_path):
    """
    Get the path to the most recent log folder from today or the previous day.

    This function finds the most recent experiment log folder by checking today's
    date first, and if no folders exist, checking yesterday's date. It identifies
    the folder with the highest experiment ID number.

    Args:
        log_path (str): Base path to the log folders directory.

    Returns:
        str: The path to the most recent log folder.
    """
    today = str(datetime.date.today())
    log_today = f"{log_path}{today}/"

    if not os.path.exists(log_today):
        today = today - datetime.timedelta(days=1)
        log_today = f"{log_path}{today}/"

    exps = []
    for f in os.listdir(log_today):
        try:
            exps.append(int(f))
        except Exception:
            continue
    exp_id = np.max(exps)

    log_folder = log_today + f"{exp_id}/"
    return log_folder


def save_config(config, path):
    """
    Save configuration parameters to a JSON file.

    This function serializes a configuration object to JSON format, excluding
    internal Python attributes. It can handle both absolute and relative paths,
    automatically appending .json extension if not present.

    Args:
        config (Config): Configuration object to save.
        path (str): Path where the configuration should be saved.
    """
    dic = config.__dict__.copy()
    del (dic["__doc__"], dic["__module__"], dic["__dict__"], dic["__weakref__"])

    if not path.endswith(".json"):
        path += ".json"

    with open(path, "w") as f:
        json.dump(dic, f)


def get_size(folder):
    """
    Calculate the total size of a folder and its contents in gigabytes.

    Args:
        folder (str): Path to the folder to calculate size for.

    Returns:
        float: Total size of the folder and its contents in gigabytes.
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    total_size = total_size / math.pow(1024, 3)
    return total_size


def upload_to_kaggle(folders, directory, dataset_name, update_folders=True):
    """
    Uploads folders to Kaggle as a dataset.

    This function copies the specified folders to a target directory, creates or updates
    a Kaggle dataset, and uploads it to Kaggle. It handles both new dataset creation and
    version updates of existing datasets.

    Args:
        folders (list): List of folder paths to be included in the dataset.
        directory (str): Target directory where the dataset will be prepared.
        dataset_name (str): Name of the Kaggle dataset.
        update_folders (bool, optional): Whether to update existing folders. Defaults to True.

    Note:
        - Requires Kaggle API credentials to be configured.
        - Creates a dataset-metadata.json file if it doesn't exist.
        - Removes validation feature files (fts_val_*.npy) from the dataset.
        - Handles both new dataset creation and version updates.
    """
    os.makedirs(directory, exist_ok=True)
    for folder in folders:
        print(f"- Copying {folder} ...")
        name = "_".join(folder[:-1].split("/")[-2:])
        try:
            shutil.copytree(
                folder,
                directory + name,
                ignore=lambda src, names: [n for n in names if "feature" in n],
            )
        except FileExistsError:
            if update_folders:
                shutil.rmtree(directory + name)
                shutil.copytree(
                    folder,
                    directory + name,
                    ignore=lambda src, names: [n for n in names if "feature" in n],
                )
        for i in range(4):
            try:
                os.remove(f"{directory + name}/fts_val_{i}.npy")
            except Exception:
                pass

    print(f"\nDataset size : {get_size(directory):.3f} Go")

    if os.path.exists(directory + "dataset-metadata.json"):
        # Update version
        print("- Update existing dataset !")
        command = f'kaggle d version -m "update" -p {directory} --dir-mode zip'
    else:
        print("- Create new dataset ! ")
        # Create dataset-metadata.json
        with open(directory + "dataset-metadata.json", "w") as f:
            slug = re.sub(" ", "-", dataset_name.lower())
            dic = {
                "title": f"{dataset_name}",
                "id": f"theoviel/{slug}",
                "licenses": [{"name": "CC0-1.0"}],
            }
            json.dump(dic, f)

        command = f"kaggle d create -p {directory} --dir-mode zip"

    # Upload dataset
    print("- Uploading ...")
    try:
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print("\nOutput :\n", output)
        print("\nError :\n", error)
    except Exception:
        print("\nUpload failed, Run command manually :\n", command)
