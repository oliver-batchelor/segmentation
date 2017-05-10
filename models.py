import segnet
import unet

import tools.model_io as model_io

models = {
    "segnet" : segnet,
    "unet"   : unet
    }

def without(d, key):
    new_d = d.copy()
    new_d.pop(key)
    return new_d

def create(params):
    model = params['model']

    assert model in models, "invalid model type"
    return models[model].segmenter(** without(params, 'model'))


def save(path, model, model_params, epoch):
    state = {
        'epoch': epoch,
        'params': model_params,
        'state': model.state_dict()
    }

    model_io.save(path, epoch, state)

def load(path):

    state = model_io.load(path)
    params = state['params']
    model = create(params)

    model.load_state_dict(state['state'])

    return model, params, state['epoch']
