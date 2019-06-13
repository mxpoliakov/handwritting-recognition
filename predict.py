import sys

import cv2

from model import Model
from preprocess import preprocess


def test(paths):
    imgs = [preprocess(cv2.imread(path, cv2.IMREAD_GRAYSCALE)) for path in paths]
    model = Model()

    predicted = model.predict(imgs)

    print("\nYour predictions are:")
    [print(paths[i], ":", predicted[i]) for i in range(len(paths))]


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Specify test files")
    else:
        test(sys.argv[1:])
