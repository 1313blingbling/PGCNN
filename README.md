# Polymer Graph Convolutional Neural Network(PGCNN)
This repository contains a PGCNN model, which can predict the properties of polymer materials according to the molecular diagram of polymer materials on the basis of evidence learning, and output the predicted uncertainties.

## Dependencies
To run the code, the following dependencies are required:
- python (>=3.7)
- pytorch (support coming soon)

##Dataset
The dataset used for training and testing the PGCNN model is stored in the data directory. The dataset consists of polymer materials, each represented by SMILE. The molecular graph contains information about the atoms, bonds, and other features of the polymer.

##Model Architecture
The PGCNN model is a graph convolutional neural network that takes a molecular graph as input and predicts a property of the polymer material. The model consists of several layers of graph convolutional networks (GCNs) followed by a fully connected layer. The output of the model is a prediction for the property of interest.
The PGCNN model also includes an evidential layer, which provides uncertainty estimates for the predictions. This allows the model to provide a measure of confidence for each prediction.

##Training and Testing
To train and test the model, run the following command:
```
train and test.py
```
 The script takes the path of the trained model and the test dataset as parameters, and evaluates the performance of the model on the test dataset. At the same time, the trained model and the path of the test data set are taken as parameters, and the performance of the model on the test data set is evaluated.

##Results
The results of the training and testing are stored in the results directory. The results include the predictions, targets, and uncertainties for each sample in the test dataset. These results can be used to analyze the performance of the model and make further improvements if necessary.
