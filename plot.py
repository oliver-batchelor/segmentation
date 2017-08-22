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


def class_cat(cat):
    if(isinstance(cat, tuple)):
        return cat
    else:
        return cat, cat

def limit_series(cat, ns=[1, 10, 100, 1000, 10000]):
    cat, classname = class_cat(cat)

    r = {}
    for n in ns:
        scalars = [s.value for s in last_n(get_scalars('test/classes/' + classname + '/iou', 'log/limit/' + cat + '/', str(n) + '-limit'), 2)]
        if(len(scalars)):
            r[n] = Struct(mean=mean(scalars), stdev=stdev(scalars))
    return cat, r


def depth_series(cat, ns=[1, 2, 3, 4, 5, 6]):
    cat, classname = class_cat(cat)

    r = {}
    for n in ns:
        scalars = [s.value for s in last_n(get_scalars('test/classes/' + classname + '/iou', 'log/depth/' + cat + '/', str(n) + '-depth'), 2)]
        if(len(scalars)):
            r[n] = mean(scalars)
    return cat, r


# def increment_series(cat):
#     cat, classname = class_cat(cat)
#
#     scalars = [s.value for s in get_scalars('test/classes/' + classname + '/iou', 'log/inc/' + cat + '/', 'pretrained')]


categories = [('trees', 'trunk'), 'aeroplane', 'cat', ('potted_plant', 'potted plant')]

def make_depth_graph():
    series = [depth_series(cat) for cat in categories]

    with PdfPages('figures/depth.pdf') as pdf:
        fig = plt.figure()
        fig, ax = plt.subplots(1,1)

        x = list(range(1, 7))
        ax.set_xticks(x)
        ax.set_xlabel("depth of model")
        ax.set_ylabel("IOU")

        for name, values in series:
            plt.plot(x, list(values.values()), label=name)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)

        fig.suptitle('Depth of model')
        pdf.savefig(fig)




def make_limit_graph():
    series = [limit_series(cat) for cat in categories[1:]]

    _, trees = limit_series(('trees', 'trunk'))
    trees[10000] = trees[1000]

    series.append(('trees', trees))

    with PdfPages('figures/limit.pdf') as pdf:
        fig = plt.figure()
        fig, ax = plt.subplots(1,1)

        x = list(range(0, 5))
        ax.set_xticks(x)

        ax.set_xticklabels(['1', '10', '100', '1000', 'all'])
        ax.set_ylabel("IOU")
        ax.set_xlabel("number of examples")


        for name, values in series:
            means = [v.mean for v in values.values()]
            devs = [v.stdev for v in values.values()]

            plt.errorbar(x[:len(means)], means, yerr=devs, marker='o', capsize=2, label=name)


        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)

        fig.suptitle('Number of images vs IOU')
        pdf.savefig(fig)



figure_path = "figures"

if not os.path.isdir(figure_path):
    os.makedirs(figure_path)


make_limit_graph()
make_depth_graph()
