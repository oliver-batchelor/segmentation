from tools.image import transforms, cv
from tools.image.index_map import default_map

def overlay(image, boxes, labels, confidence=None, classes=None):
    image = image.clone()

    for i in range(0, boxes.size(0)):
        color = default_map[labels[i]]
        label = labels[i]

        if label >= 0:
            cv.rectangle(image, boxes[i, :2], boxes[i, 2:], color=color, thickness=2)
            if(classes and label < len(classes)):
                cv.putText(image, classes[label]['name'], (boxes[i, 0], boxes[i, 1] + 12), scale = 0.5, color=color, thickness=1)

            if not (confidence is None):
                str = "{:.2f}".format(confidence[i])
                cv.putText(image, str, (boxes[i, 0], boxes[i, 3] - 2), scale = 0.5, color=color, thickness=1)

    return image

def display_batch(batch, cols=6, classes=None):
    print(classes)

    images = []
    for i in zip(batch['images'], batch['boxes'], batch['labels']):
        images.append(overlay(*i, classes=classes))

    return tensor.tile_batch(torch.stack(images, 0), cols)
