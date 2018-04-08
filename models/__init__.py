

from models import ssd

def merge(*dicts):
    m = {}
    for d in dicts:
        m.update(d)

    return m


models = ssd.models
