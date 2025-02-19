In this project, I trained three different models on the classic MNIST dataset (those handwritten digit images) using Jupyter Notebook. The models are:
1) Random Forest – trained with sklearn
2) Feed-Forward Neural Network (FNN) – built with TensorFlow/Keras
3) Convolutional Neural Network (CNN) – also with TensorFlow/Keras
The MNIST dataset is pulled straight from TensorFlow, and each model is trained using the right tools:
Random Forest - sklearn.ensemble.RandomForestClassifier
FNN & CNN - tensorflow.keras

How It's Set Up
models.py:
This file has an interface (MnistClassifierInterface) that all models follow. It was created by using abstract base classes (ABCs). So every model is required to have train() and predict() methods, keeping things consistent across different classifiers.
Each model (Random Forest, FNN, CNN) has train() and predict() methods.
There's also a MnistClassifier class where you can pick which model you want to run.
main.py:
Loads data properly for Random Forest (load_data_for_RF()) and neural networks (load_data_for_NN()).
Has a function represent_predictions() so you can choose which test samples to check out.
settings.py:
This file includes all the important libraries and settings.

Installation & Setup

Clone the repository:

git clone <repository_url>
cd mnist_classifier

Create a virtual environment (optional but recommended):

venv\Scripts\activate #on Windows

Install dependencies:

pip install -r requirements.txt

Usage

Running the Main Script

To train and test models, run:

python main.py
Running the desired method u can change in main.py:
rf = MnistClassifier('rf')  # Random Forest
cnn = MnistClassifier('cnn')  # Convolutional Neural Network
nn = MnistClassifier('nn')  # Feed-Forward Neural Network

cnn.train(X_train=x_train, y_train=y_train)
represent_predictions(predictions, y_test, first_index=30, second_index=40)

Running the Jupyter Notebook

To see model performance and edge cases, open and run demo.ipynb:


