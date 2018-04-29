
from os import path
import json

import torch
from dataset.detection import DetectionDataset


def load_file(filename):
    with open(filename, "r") as file:
        str = file.read()
        return load_json(str)
    raise Exception('load_file: file not readable ' + filename)


def load_json(str):
    data = json.loads(str)
    config = data['config']

    classes = [{'id':int(k), 'name':v} for k, v in config['classes'].items()]
    class_mapping = {c['id']:i  for i, c in enumerate(classes)}

    def to_box(obj):
        b = obj['bounds']
        return [*b['lower'], *b['upper']]

    def to_label(obj):
        return class_mapping[obj['classId']]

    def to_image(image):
        objs = [obj for obj in image['instances'] if obj['tag'] == 'ObjBox']

        boxes = [to_box(obj) for obj in objs]
        labels = [to_label(obj) for obj in objs]

        return {
            'file':path.join(config['root'], image['imageFile']),
            'boxes': torch.FloatTensor(boxes),
            'labels': torch.LongTensor(labels)
        }

    train = [to_image(i) for i in data['images'] if i['category'] == 'Train']
    test = [to_image(i) for i in data['images'] if i['category'] == 'Test']

    return classes, DetectionDataset(train), DetectionDataset(test)


    # train = training_on(find_files(os.path.join(args.input, args.train_folder), args.limit), args)
    # test = testing_on(find_files(os.path.join(args.input, args.test_folder)), args)

    # return class_names, train, test
