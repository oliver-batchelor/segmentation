from models import fcn

def merge(*dicts):
    m = {}
    for d in dicts:
        m.update(d)

    return m


models = fcn.models
