
import torch
from tqdm import tqdm
import gc


def train(model, loader, eval, optimizer):
    print("training:")
    stats = 0
    model.train()

    with tqdm(total=len(loader) * loader.batch_size) as bar:
        for data in loader:

            optimizer.zero_grad()
            result = eval(model, data)
            result.error.backward()
            optimizer.step()
            stats += result.statistics

            bar.update(result.statistics.size)

            del result
            gc.collect()

    return stats


def test(model, loader, eval):
    print("testing:")
    stats = 0
    model.eval()
    for data in tqdm(loader):

        result = eval(model, data)
        stats += result.statistics

        del result
        gc.collect()

    return stats


def test_images(model, files, eval):
    results = []
    model.eval()

    for (image_file, mask_file) in tqdm(files):
        data = dataset.load_rgb(image_file)
        labels = dataset.load_labels(mask_file)

        results.append((image_file, eval(data)))

    return results
