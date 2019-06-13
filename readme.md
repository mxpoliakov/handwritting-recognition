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

## Pipeline

![alt text](scheme.png)

The architecture:
1) convolutional layers, which extract a feature sequence from the input image;
2) recurrent layers, which predict a label distribution for each frame;
3) transcription layer, which translates the per-frame predictions into the final label sequence.


## Authors

* **Mykhailo Poliakov** 
* **Anton Borkivskyi**
* **Bohdan Borkivskyi**

## Acknowledgments

[Build a Handwritten Text Recognition System using TensorFlow by Harold Scheidl](https://towardsdatascience.com/build-a-handwritten-text-recognition-system-using-tensorflow-2326a3487cd5)
