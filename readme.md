# Handwriting recognition

This project is created to recognize handwritten words from [IAM database](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database). 

## Project structure
```
handwritting-recognition
│   README.md
|   requirements.txt
│   model - folder with pretrained model   
│   test - test dataset
|   main.py
│   layers.py - implementation of CNN, RNN and CTC layers
│   model.py - model implementation
|   predict.py - prediction of own handwriting
│   preprocess.py - input image preprocessing
│   structure.py - collecting dataset
```

## Getting Started

You should have Python and Pip installed.

```
pip install -r requirements.txt
```

```
python main.py
```

This will download the dataset automatically, create directory structure and train the model.

There is pretrained model in `/model` folder. If you want to retrain model or train it with your dataset - clear `/model` folder.

## Pipeline
![alt text](scheme.png)

The architecture:
1) convolutional layers, which extract a feature sequence from the input image;
2) recurrent layers, which predict a label distribution for each frame;
3) transcription layer, which translates the per-frame predictions into the final label sequence.


## Predicting results
To predict custom handwriting run

    python predict.py <path1> [path2] [path_n]
You should specify at least one image path. Example of usage:

    >python predict.py test/test1.png test/test2.png test/test3.png

    Your predictions are:
    test/test1.png : Anton
    test/test2.png : Gurbych
    test/test3.png : ucu-mL

| Image | ![Anton](test/test1.png) | ![Gurbych](test/test2.png) | ![ucu-ml](test/test3.png) |
| :---: | :---: | :---: | :---: |
| Predicted label | Anton | Gurbych | ucu-mL


## Authors

* **Mykhailo Poliakov** 
* **Anton Borkivskyi**
* **Bohdan Borkivskyi**

## Acknowledgments

[Build a Handwritten Text Recognition System using TensorFlow by Harold Scheidl](https://towardsdatascience.com/build-a-handwritten-text-recognition-system-using-tensorflow-2326a3487cd5)
