from settings import *


class MnistClassifierInterface(ABC):
    #create abstract methods
    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

# Implementation of a Random Forest classifier
class RandomForest(MnistClassifierInterface):
    def __init__(self, n_estimators=100, random_state=42):
        # Initialize the RandomForest model with specified parameters
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    
    def train(self, X_train, y_train):
        # Train the RandomForest model
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        # Make predictions using the trained model
        return self.model.predict(X_test)
    

# Implementation of a Convolutional Neural Network (CNN) classifier
class ConvolutionalNeuralNetwork(MnistClassifierInterface):
    def __init__(self, epoch=10):
        # Define the CNN architecture
        self.model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')  # Output layer for 10 classes
        ])
        self.epoch = epoch
    
    def train(self, X_train, y_train):
        # Compile and train the CNN model
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                           metrics=['accuracy'])
        self.model.fit(X_train, y_train, epochs=self.epoch, verbose=1)
    
    def predict(self, X_test):
        # Make predictions and return the class with the highest probability
        return  self.model.predict(X_test, verbose=0)

# Implementation of a Feed Forward Neural Network (FNN) classifier
class FeedForwardNeuralNetwork(MnistClassifierInterface):
    def __init__(self, epoch=10):
        # Define a simple Feed Forward Neural Network (fully connected layers)
        self.model = models.Sequential([
            layers.Flatten(input_shape=(28, 28,1)),
            layers.Dense(128, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        self.epoch = epoch
    
    def train(self, X_train, y_train):
        # Compile and train the FNN model
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                           metrics=['accuracy'])
        self.model.fit(X_train, y_train, epochs=self.epoch, verbose=1)

    def predict(self, X_test):
        # Make predictions and return the class with the highest probability
        return  self.model.predict(X_test, verbose=0)

# Wrapper Class to Select Classifier
class MnistClassifier:
    def __init__(self, model='rf'):
        match model:
            case "rf":
                self.classifier = RandomForest()
            case "nn":
                self.classifier = FeedForwardNeuralNetwork()
            case "cnn":
                self.classifier = ConvolutionalNeuralNetwork()
            case _:
                print("Error. Choose model from 'rf', 'nn', or 'cnn'.")
            
    def train(self, X_train, y_train):
        self.classifier.train(X_train, y_train)
    
    def predict(self, X_test):
        return self.classifier.predict(X_test)
    
