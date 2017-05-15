from pycrayon import CrayonClient

cc = CrayonClient(hostname='localhost')



def enumerate_name(base, names):
    i = 1
    name = base
    while name in names:
        name = base + i
        i = i + 1

    return name



def new(args):
    if args.experiment == '':
        return None

    if args.load:
        return cc.open_experiment(args.experiment)

    name = enumerate_name(args.experiment, cc.get_experiment_names())
    return cc.create_experiment(name)
