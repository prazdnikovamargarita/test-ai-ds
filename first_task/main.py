
from models import MnistClassifier
from settings import *


def load_data_for_NN(data):
    (x_train, y_train), (x_test, y_test) = data.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Reshape data to fit CNN (add channel dimension for grayscale images)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    return x_train, y_train, x_test, y_test

def load_data_for_RF(data):
    (x_train, y_train), (x_test, y_test) = data.load_data()

    # Reshape data to 2D (Flatten each image from 28x28 to a 784-pixel vector)
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    # Normalize pixel values to [0, 1]
    x_train, x_test = x_train / 255.0, x_test / 255.0

    return x_train, y_train, x_test, y_test

def represent_predictions(predictions, y_test, first_index=30, second_index=40):
    # Display predictions along with actual labels
    print("\nPredictions:")
    i = first_index + i
    for i, pred in enumerate(predictions[first_index:second_index]):

        if isinstance(pred, np.ndarray) and pred.ndim == 1:
            predicted_label = np.argmax(pred)  # Extract the class with the highest probability
        else:
            predicted_label = pred  # If the prediction is already a class label, use it directly
        
        actual_label = y_test[i]
        print(f"Sample {i}: Expected Label: {actual_label}, Predicted Label: {predicted_label}")


x_train, y_train, x_test, y_test = load_data_for_RF(mnist)
rf = MnistClassifier('rf')
rf.train(X_train=x_train, y_train=y_train)

represent_predictions(rf.predict(X_test=x_test), y_test=y_test)

x_train, y_train, x_test, y_test = load_data_for_NN(mnist)
cnn = MnistClassifier('cnn')
cnn.train(X_train=x_train, y_train=y_train)
print("predict cn:", cnn.predict(X_test=x_test))
represent_predictions(cnn.predict(X_test=x_test), y_test=y_test)


nn = MnistClassifier('nn')
nn.train(X_train=x_train, y_train=y_train)
print("predict nn:", nn.predict(X_test=x_test))
represent_predictions(nn.predict(X_test=x_test), y_test=y_test)
