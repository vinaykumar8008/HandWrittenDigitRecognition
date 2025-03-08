# HandWrittenDigitRecognition
🔢 Handwritten Digit Recognition with MNIST

This Python project uses a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. The model is trained on the dataset and then tested on a custom image containing handwritten digits.

🚀 Features

🏗️ Builds a CNN for digit classification using Keras & TensorFlow

🖼️ Preprocesses the MNIST dataset for model training

📊 Visualizes accuracy and loss trends during training

📸 Reads and processes a custom image containing handwritten digits

🔍 Identifies and predicts digits from the image

🛠 Technologies Used

Python 🐍

TensorFlow/Keras 🤖

OpenCV 🎥

NumPy 🔢

Matplotlib 📊

📌 How to Run

Install the required libraries:

pip install numpy tensorflow keras opencv-python matplotlib

Run the script to train the CNN model:

python mnist_digit_recognition.py

The model will train on the MNIST dataset and display accuracy/loss graphs.

The program will then process an external image (testt.webp) and predict the digits present in the image.

📖 Example Output

Epoch 1/6
Train Accuracy: 97%
Validation Accuracy: 98%
...
Predicted number: 5283

📜 Data Preprocessing

The MNIST dataset is reshaped to (28,28,1) and normalized.

Labels are converted to categorical format for multi-class classification.

The custom image is converted to grayscale and thresholded before passing through the model.

📸 Custom Image Processing

Loads the image with OpenCV.

Converts it to grayscale.

Applies binary thresholding to detect digits.

Extracts and resizes digits for prediction.

🏆 Benefits of This Project

Strengthens knowledge of deep learning and CNNs.

Provides practical experience with image processing using OpenCV.

Demonstrates real-world digit recognition applications.

📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

🤝 Contributions

Feel free to fork this repository and submit pull requests with improvements or bug fixes.

Happy coding! 🚀🎯

