

from models import pretrained, unet, ladder


def merge(*dicts):
    m = {}
    for d in dicts:
        m.update(d)

    return m


models = merge(pretrained.models, unet.models, ladder.models)
