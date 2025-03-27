import json
import os
from datetime import datetime
import csv
import copy

def init_logs_folder(savedir_base):
    exp_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    savedir = os.path.join(savedir_base, exp_date)

    os.makedirs(savedir, exist_ok=True)

    return savedir


def init_csv(savedir, csv_name, col_names):
    csv_path = os.path.join(savedir, csv_name)

    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(col_names)


def append_row_csv(savedir, csv_name, row):
    csv_path = os.path.join(savedir, csv_name)

    with open(csv_path, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(row)


def save_json(fname, data, makedirs=True):
    """
    # From Haven utils
    """

    # turn fname to string in case it is a Path object
    fname = str(fname)
    dirname = os.path.dirname(fname)
    if makedirs and dirname != "":
        os.makedirs(dirname, exist_ok=True)
    with open(fname, "w") as json_file:
        json.dump(data, json_file, indent=4, sort_keys=True)