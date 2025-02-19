For this task, I found that the better approach is to use a Bidirectional LSTM-based Text Classification Model because the NER model did not perform very well.

I also used CNN for classification, but the detection was not very accurate. However, I don’t have much time to fix it.

In main.py, I use trained models, and you can call the ImageRecognizing class with a sentence and an image path.

For better performance, detection should be improved.

I also tried RF and SVM, but the dataset contains too many images, making training inefficient.

You can run everything from main.py or main_demo.py.

First, run train_and_inference_cnn.py and train_and_inference_nlp.py to create trained model files in Keras.

Follow the same installation and setup as in the first task.

Running the Main Script
To train and test models, run:
python main.py

# Example usage
recognizer = ImageRecognizing("There is a cat in the picture.", "Animal Image Dataset\\validation\horse\\OIP-_KQbEV3mFTHovOMDGBF6cgHaF7.jpeg")
result = recognizer.show_result()
print("Result:", result)