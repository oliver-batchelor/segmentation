import tensorflow as tf
from tensorflow.tensorboard.backend.event_processing import event_accumulator

from statistics import mean, stdev

from matplotlib import rc
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

from tools import Struct
import os


def extract(scalar, file):
    acc = event_accumulator.EventAccumulator(path=file)
    acc.Reload()
    return acc.Scalars(scalar)

def find_logs(path, prefix):
    return [os.path.join(path, file) for file in os.listdir(path) if file.startswith(prefix)]



def get_scalars(scalar, path, prefix):
    return [extract(scalar, file) for file in find_logs(path, prefix)]


def last_n(scalars, n):
    return sum([s[len(s) - n:] for s in scalars], [])


def limit_series(cat, ns=[1, 10, 100, 1000, 10000]):
    classname = cat
    if(isinstance(cat, tuple)):
        cat, classname = cat

    r = {}
    for n in ns:
        scalars = [s.value for s in last_n(get_scalars('test/classes/' + classname + '/iou', 'log/limit/' + cat + '/', str(n) + '-limit'), 2)]

        if(len(scalars)):
            r[n] = Struct(mean=mean(scalars), stdev=stdev(scalars))

    return cat, r


figure_path = "figures"

if not os.path.isdir(figure_path):
    os.makedirs(figure_path)

series = [limit_series(cat) for cat in ['aeroplane', ('trees', 'trunk'), 'cat', ('potted_plant', 'potted plant')]]


with PdfPages('figures/limit.pdf') as pdf:
    fig = plt.figure()
    fig, ax = plt.subplots(1,1)

    x = list(range(0, 5))
    ax.set_xticks(x)

    ax.set_xticklabels(['1', '10', '100', '1000', 'all'])


    for name, values in series:
        means = [v.mean for v in values.values()]
        devs = [v.stdev for v in values.values()]

        plt.errorbar(x[:len(means)], means, yerr=devs, marker='o', capsize=2, label=name)


    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    fig.suptitle('Number of images vs IOU')
    pdf.savefig(fig)
