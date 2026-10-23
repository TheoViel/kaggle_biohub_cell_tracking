import os
import re
import sys
import datetime
import numpy as np


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


DATE_DIR = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def get_last_log_folder(log_path, contains="args.json"):
    """
    Get the path to the most recent experiment folder, as prepare_log_folder lays
    them out: ``log_path/<date>/<exp_id>/``.

    The dates are ISO, so they sort lexicographically, and the experiment ids are
    numeric — the most recent run is the highest id under the latest date that
    holds one. Every date is walked, newest first, rather than only today's:
    prepare_log_folder creates the date folder before the run it is for, so the
    newest one can be empty, and there is no reason the last run has to have
    happened today.

    Args:
        log_path (str): Base path to the log folders directory.
        contains (str, optional): A file the folder must hold to count as a run.
            Defaults to args.json, which main_track writes before it trains
            anything, so a folder created by a run that died at startup is
            skipped rather than returned. Pass None to accept any folder.

    Returns:
        str: The path to the most recent log folder, with a trailing slash.
    """
    if not os.path.isdir(log_path):
        raise FileNotFoundError(f"No log directory at {log_path}")

    dates = sorted(
        (
            d for d in os.listdir(log_path)
            if DATE_DIR.match(d) and os.path.isdir(os.path.join(log_path, d))
        ),
        reverse=True,
    )
    for date in dates:
        day = os.path.join(log_path, date)
        exps = sorted(
            (
                int(f) for f in os.listdir(day)
                if f.isdigit() and os.path.isdir(os.path.join(day, f))
                and (not contains or os.path.exists(os.path.join(day, f, contains)))
            ),
            reverse=True,
        )
        if exps:
            return os.path.join(day, str(exps[0])) + "/"

    raise FileNotFoundError(
        f"No experiment under {log_path}<date>/<id>/"
        + (f" holding {contains}" if contains else "")
    )




