import csv
from os import path
import numpy as np

def read_benchmark_list(filename):
    return csv.DictReader(open(path.join("results",filename)))


def write_benchmark_list(filename, models_list, fieldnames):
    writer = csv.DictWriter(open(path.join("results",filename), 'w'), fieldnames=fieldnames)
    writer.writeheader()
    for item in models_list:
        item['float_acc'], item['quant_acc'] = np.round((item['float_acc'], item['quant_acc']),4)
        writer.writerow(item)
