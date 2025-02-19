import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
# Load and prepare the MNIST dataset
mnist = tf.keras.datasets.mnist
