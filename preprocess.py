import os
import glob
import cv2
import numpy as np
import pandas as pd


def preprocess(img, output_shape=(128, 32)):
    (w_output, h_output) = output_shape
    (h, w) = img.shape
    fx = w / w_output
    fy = h / h_output
    f = max(fx, fy)

    size = (max(min(w_output, int(w / f)), 1), max(min(h_output, int(h / f)), 1))
    img = cv2.resize(img, size)

    temp = np.ones([h_output, w_output]) * 255
    temp[0 : size[1], 0 : size[0]] = img
    img = cv2.transpose(temp)

    (mean, std_dev) = cv2.meanStdDev(img)
    (mean, std_dev) = (mean[0][0], std_dev[0][0])
    img = img - mean
    img = img / std_dev if std_dev > 0 else img

    return img


def create_dataset():
    words_df = pd.read_csv(
        "data/words.txt", skiprows=18, sep=" ", error_bad_lines=False, warn_bad_lines=False, header=None
    )
    words_df.columns = ["code", "status", "greylevel", "x", "y", "w", "h", "tag", "text"]
    words_df = words_df[words_df.status == "ok"]
    words_df = words_df[~words_df.text.str.contains("\n")]

    X = []
    y = []

    for word in glob.glob("data/**/*.png", recursive=True):
        word_code = os.path.splitext(os.path.basename(word))[0]

        try:
            word_text = words_df[words_df.code == word_code].text.item()
        except ValueError:
            continue

        img = cv2.imread(word, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = preprocess(img)
            X.append(img)
            y.append(word_text)

    chars = sorted(set.union(*(set(word_text) for word_text in words_df["text"])))

    return X, y, chars
