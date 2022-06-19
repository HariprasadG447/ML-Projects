# ML-Projects

This repository contains two Jupyter notebook files which contain python codes of two ML projects.

# Note

All the additional files which are used in the notebooks are present in the reporsitory. Kindly, modify the code in the Jupyter notebooks order to use the correct file path to import the data in these files.

## Project 1 - Segregation of customers using K Means clustering algorithm

This model takes an input of customer data which contains there Age, Income, Gender and expenditure and then segregates them into classes based on the values of these different parameters. The method used for the unsupervised clustering is the K Means clustering algorithm with the help of the Scikit library in Python.
The data is processed initially with steps such as standardisation and transforming every feature so that all have numerical values. In order to improve the performance of the model, PCA is performed and the model is evaluated. Finally, the clusters are visualised by plotting a graph of all the different features and colour coding the clusters.

## Project 2 - Recognising Hand-Written Digits using a 3-Layered Neural Network

In this project, we train a model which helps in recognising hand-written digits by using a model to train the famous MNIST data set of Hand-written digits. The model is a 3-Layered Neural network such that each unit of a layer is connected to each unit of the next layer. The output layer has 10 units which identifies the 10 digits of numbers from 0 to 9. The neural network model is set up with the help of the tensorflow library. After setting up the model, a few images hand-drawn in MS Paint are tested and the results are shown in the notebook.

## Libraries used

* Numpy
* Matplotlib and mpl_toolkits.mplot3d
* Pandas
* cv2 (OpenCV-Python)
* Scikit learn (sklearn.datasets, sklearn.cluster, sklearn.metrics, sklearn.preprocessing, sklearn.decomposition)
* Tensorflow
