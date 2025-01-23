# Deep Space Objects Classification Tool

## Introduction

The project aims to offer the user an automated way to differentiate between two classes of **Deep Space Objects** (DSO): **galaxies** and **nebulae**. With the use of an AI model, trained for 100 epochs, the model predicts the classes with an average validation accuracy of **0.8108** and a validation loss of **0.3961**. Although the performance can be improved, the dataset provided for this project is relatively small. It includes **236 images of galaxies** and **228 images of nebulae** taken from [ESA's Hubble Archive](https://esahubble.org/images/).

The project leverages the power of deep learning to analyze and classify images of deep space objects. By using a convolutional neural network (CNN), the model is able to learn and identify patterns within the images that distinguish galaxies from nebulae. This approach not only automates the classification process but also provides a scalable solution for analyzing large datasets of astronomical images.

In addition to the AI model, the project includes a web application built with Flask. This web app allows users to upload images of deep space objects and receive real-time predictions from the trained model. The user-friendly interface ensures that even those with limited technical knowledge can easily interact with the system and obtain accurate classifications.

Overall, this project demonstrates the potential of AI in the field of astronomy, providing a valuable tool for researchers and enthusiasts alike. By automating the classification of deep space objects, the project aims to contribute to the ongoing exploration and understanding of our universe.

---

## Technologies Used

- **AI Model**: A custom Convolutional Neural Network (CNN) architecture from a model found on [Kaggle](https://www.kaggle.com/code/fareselmenshawii/cats-vs-dogs-classification).
- **Dataset**: The dataset consists of **236 images of galaxies** and **228 images of nebulae** taken from [ESA's Hubble Archive](https://esahubble.org/images/).
- **Web App**: The app's interface is made with **HTML** templates (`index.html`, `documentation.html`) using **CSS** style (`style.css`), and the backend is powered by **Python** with the **Flask** framework (`app.py`).
- **Flask**: [Flask](https://flask.palletsprojects.com/en/stable/) is a lightweight WSGI web application framework in Python. It is designed with simplicity and flexibility in mind, making it easy to get started with web development.
- **TensorFlow and Keras**: Used for building and training the AI model. TensorFlow is an open-source machine learning framework developed by Google, and Keras is a high-level neural networks API that runs on top of TensorFlow.

---

## AI Model

The `model.py` file contains the implementation of a Convolutional Neural Network (CNN) used to classify deep space objects into two categories: galaxies and nebulae. After training, the model is saved as `nebulae_v_galaxies.h5` and used in the web app to make predictions or load the model again to run the tests in the terminal.

### Key Components of the AI Model:
- **Data Preparation**: The dataset is split into training, validation, and test sets. Data augmentation techniques are applied to the training set to improve the model's generalization.
- **Model Architecture**: A custom CNN architecture is defined, inspired by a model from Kaggle. The model consists of multiple convolutional layers followed by dense layers.
- **Training**: The model is compiled with the RMSprop optimizer and trained using the training and validation sets for 100 epochs.
- **Evaluation**: The trained model is evaluated on the test set to measure its performance.
- **Activation Maps**: Activation maps are generated to visualize the areas of the images that the model focuses on when making predictions.
- **Results**: The training history is saved and used to plot the accuracy and loss over epochs.

The model achieves an average validation accuracy of **0.8108** and a validation loss of **0.3961**, indicating good performance given the small dataset size.

---

## Results

List of average results:
- **Average Training Accuracy**: 0.8297
- **Average Validation Accuracy**: 0.8108
- **Average Training Loss**: 0.4008
- **Average Validation Loss**: 0.3961

---

## Web App

The web application is built using the Flask framework in Python. It provides an interface for users to upload images of deep space objects and receive predictions from the trained AI model.

### Key Components of the Web App:
- **Templates**: The HTML templates (`index.html` and `documentation.html`) define the structure and layout of the web pages. They include navigation links, forms for image upload, and sections for displaying predictions and documentation.
- **Routes**: The `app.py` file defines the routes for the web application. Key routes include:
  - `/`: Renders the main page where users can upload images and see predictions.
  - `/predict`: Handles the image upload and prediction logic, returning the predicted class and probability.
  - `/documentation`: Renders the documentation page with detailed information about the project.

The web app's interface is styled using custom CSS, providing a user-friendly experience for interacting with the AI model.

---

## User Manual

This user manual provides step-by-step instructions on how to use the Deep Space Objects Classification Tool.

1. **Access the Web App**: Open your web browser and navigate to the URL where the web application is hosted.
2. **Upload an Image**: On the main page, you will see a drag-and-drop area or a button to select an image. Drag and drop your image of a deep space object into the designated area or click the button to select an image from your device.
3. **Submit the Image**: Once the image is uploaded, click the "Upload your image" button to submit the image to the server.
4. **Get Predictions**: After the image is uploaded, click the "Predict" button to receive the classification result. The web app will display the predicted class (galaxy or nebula) along with the probability.
5. **View Documentation**: For more information about the project, click on the "Documentation" link in the navigation bar to access detailed documentation.


---

This project demonstrates the potential of AI in the field of astronomy, providing a valuable tool for researchers and enthusiasts alike. By automating the classification of deep space objects, the project aims to contribute to the ongoing exploration and understanding of our universe.
