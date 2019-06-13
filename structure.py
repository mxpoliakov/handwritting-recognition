import tarfile
import os
import requests


def structure():
    url = "http://www.fki.inf.unibe.ch/DBs/iamDB/data/"
    auth = ("ucuai", "12345678")

    if not os.path.exists("data/words"):
        r = requests.get(url + "words/words.tgz", auth=auth)
        open("words.tgz", "wb").write(r.content)
        tar = tarfile.open("words.tgz")
        tar.extractall("data/words")
        tar.close()
        os.remove("words.tgz")
    else:
        print("data/words already exists!")

    if not os.path.exists("data/words.txt"):
        r = requests.get(url + "ascii/ascii.tgz", auth=auth)
        open("ascii.tgz", "wb").write(r.content)
        tar = tarfile.open("ascii.tgz")
        tar.extractall("data")
        tar.close()
        os.remove("ascii.tgz")
    else:
        print("data/words.txt already exists!")