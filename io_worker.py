import fnmatch
import pickle as pk
import json
import os
import csv
import logging
import sys
from collections import defaultdict
import numpy as np
import string
import sklearn
from attribute_obj import NumericalAttribute
from time import time
import random
import math
import pandas as pd
from distances import get_transformation
import texttable as tt


def print_progress(current, total, message=""):
    progress = current * 1. / total
    bar_length = 50
    block = int(round(bar_length * progress))
    text = "\r%6.2f%% - [%s] - %d/%d: %s" % (progress * 100,
                                             "=" * block + " " * (bar_length - block),
                                             current, total, message)
    sys.stdout.write(text)
    sys.stdout.flush()


def print_count(message, count):
    text = "\r%s: %d" % (message, count)
    sys.stdout.write(text)
    sys.stdout.flush()


def print_status(message, save_log_file=True, show_on_screen=True):
    if save_log_file:
        logging.debug(message)
    if show_on_screen:
        print(message)


def print_labeling(method, results):
    print_status("Method: %s" % method)
    headings = ["Rank", "Distance", "Label"]
    width = [4, 15, 30]
    for i_res, res in enumerate(results):
        temp_query = res[0].values
        temp_label = res[0].label
        print_status("\nQuery %d: %s [%s, ...]" % ((i_res + 1), temp_label,
                                                 ", ".join(["%.2f" % temp for temp in temp_query[:5]])))
        tab = tt.Texttable()
        tab.header(headings)
        tab.set_cols_width(width)
        for i_rank, rank in enumerate(res[1]):
            tab.add_row([i_rank + 1, rank[1], rank[0]])
        print_status(tab.draw())


def print_labeling_compare_mode(result_lists):
    n_methods = len(result_lists)
    n_queries = len(result_lists[0][1])
    methods = [result_lists[i][0] for i in range(n_methods)]
    headings = ['Rank'] + methods
    width = [4] + [15] * n_methods
    for i_query in range(n_queries):
        temp_query = result_lists[0][1][i_query][0].values
        temp_label = result_lists[0][1][i_query][0].label
        print_status("Query %d: %s [%s, ...]" % ((i_query + 1), temp_label,
                                                 ", ".join(["%.2f" % temp for temp in temp_query[:5]])))
        tab = tt.Texttable()
        # tab.set_deco(tab.HEADER | tab.VLINES)
        tab.set_cols_width(width)
        tab.header(headings)

        # print_status("Rank\t%s" % "".join(["%20s\t" % method for method in methods]))
        for i_rank in range(10):
            row = [(i_rank + 1)]
            for i_method in range(n_methods):
                # if len(result_lists[i_method][1][i_query][1][i_rank][0]) > 15:
                #     row.append("%s...\t" % result_lists[i_method][1][i_query][1][i_rank][0][:10])
                # else:
                row.append("%s\t" % result_lists[i_method][1][i_query][1][i_rank][0])
            tab.add_row(row)
        print_status(tab.draw())


def get_files_from_dir_subdir(folder_dir, extension="*"):
    all_files = []
    for root, folder_dirs, file_dirs in os.walk(folder_dir):
        for file_dir in fnmatch.filter(file_dirs, "*.%s" % extension):
            if ".DS_Store" not in file_dir:
                all_files.append(os.path.join(root, file_dir))
    return all_files


def get_files_from_dir(folder_dir, is_sorted=False, extension="*", decrease=True, num_files=-1):
    all_file_dirs = get_files_from_dir_subdir(folder_dir, extension)

    if is_sorted:
        file_with_size = [(f, os.path.getsize(f)) for f in all_file_dirs]
        file_with_size.sort(key=lambda f: f[1], reverse=decrease)
        all_file_dirs = [f for f, _ in file_with_size]

    num_files = len(all_file_dirs) if num_files == -1 or num_files > len(all_file_dirs) else num_files

    return all_file_dirs[:num_files]


def create_dir(file_dir):
    folder_dir = os.path.dirname(file_dir)
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)


def load_object_csv(file_name):
    df = pd.read_csv(file_name, header=None)
    return df.values[:, 0]


def save_object_pickle(file_name, save_object):
    create_dir(file_name)
    with open(file_name, "wb") as f:
        pk.dump(save_object, f)


def load_object_pickle(file_name):
    with open(file_name, "rb") as f:
        return pk.load(f)


def save_object_csv(file_name, rows):
    create_dir(file_name)
    with open(file_name, "w") as f:
        try:
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
            for r in rows:
                if isinstance(r, list) or isinstance(r, tuple):
                    writer.writerow(r)
                else:
                    writer.writerow([r])
        except Exception as message:
            print(message)


def remove_outliers(input_values, lowest=25, highest=75):
    values = np.array(input_values)
    values = values[~np.isnan(values)]
    q1 = np.percentile(values, lowest)
    q3 = np.percentile(values, highest)
    iqr = q3 - q1
    return values[~((values < (q1 - 1.5 * iqr)) | (values > (q3 + 1.5 * iqr)))]


def get_numbers_from_strings(input_strings):
    results = np.array(input_strings, dtype=np.float32)
    results = results[~np.isnan(results)]
    results = results[~np.isinf(results)]
    return results.tolist()


def load_csv_dataset(file, repartition_size=10):
    def repartition(attributes, n):
        if n <= 0:
            return []
        # random.shuffle(attributes)
        # save_object_csv(file, attributes)
        step = math.ceil(len(attributes) / n)
        return [attributes[p * step:p * step + step] for p in range(n)]

    label, _ = os.path.splitext(os.path.basename(file))
    if "]" == label[-1]:
        label = label[:-1].split("[")[1]
    csv_objects = load_object_csv(file)
    label_values = get_numbers_from_strings(csv_objects)
    return label, [NumericalAttribute(label, attr_values) for attr_values in repartition(label_values, repartition_size)
                   if len(attr_values) > 0]
    # return label, [NumericalAttribute(label, label_values)]


def load_txt_dataset(file):
    with open(file, "r") as file_reader:
        num_types = int(file_reader.readline().strip())
        file_reader.readline()
        for num_type in range(num_types):
            y = file_reader.readline().strip().split(":")[-1]
            num_values = int(file_reader.readline())
            temp_values = [file_reader.readline().split(" ", 1)[1] for _ in range(num_values)]
            temp_values = get_numbers_from_strings(temp_values)
            yield (y, NumericalAttribute(y, temp_values))
            file_reader.readline()


def load_numerical_dataset(data_name):
    data = []
    labels = set()
    if data_name in ["dbpedia", "wikidata"]:
        # csv dataset
        filenames = get_files_from_dir("./data/%s" % data_name, extension='csv')
        for f in filenames:
            label, pars = load_csv_dataset(f)
            labels.add(label)
            data.extend(pars)
    else:
        # txt dataset
        filenames = get_files_from_dir("./data/%s" % data_name, extension='txt')
        # filenames.sort(key=lambda f: int(f.split("/")[-1].replace(".txt", "")[1:]))
        for f in filenames:
            for label, attr in load_txt_dataset(f):
                labels.add(label)
                data.append(attr)
    return labels, data


def load_queries(data_dir):
    filenames = get_files_from_dir("./data/%s" % data_dir, extension='csv')
    queries = []
    for f in filenames:
        label, _ = os.path.splitext(os.path.basename(f))
        csv_objects = load_object_csv(f)
        label_values = get_numbers_from_strings(csv_objects)
        queries.append(NumericalAttribute(label, label_values))
    return queries
