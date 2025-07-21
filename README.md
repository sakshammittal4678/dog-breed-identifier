# End-to-End Dog Breed Classification Project

## 📖 Description

This project is an end-to-end dog breed classification model. The model is built using TensorFlow and utilizes transfer learning with a pre-trained model(inception_v3) from TensorFlow Hub to classify images of dogs into one of 120 different breeds. The primary goal is to accurately predict the breed of a dog given its image.

---

## 📊 Dataset

The data for this project is from the [Dog Breed Identification competition on Kaggle](https://www.kaggle.com/c/dog-breed-identification/overview). The dataset consists of:

* **10,200+** images in the training set
* **10,400+** images in the testing set
* **120** different dog breeds

---

## 🧠 Model Architecture

The model uses a pre-trained **Inception_v3** model from TensorFlow Hub as a feature extractor. A `Dense` layer with a `softmax` activation function is added on top of the pre-trained model to classify the images into 120 different dog breeds.

* **Input Image Size:** 299x299
* **Output Layer:** Dense layer with 120 units and softmax activation
* **Loss Function:** Categorical Crossentropy
* **Optimizer:** Adam

---

## 📈 Results

The model was trained on a subset of the training data and evaluated on a validation set. The best validation accuracy achieved was **97.7** with a validation loss of **0.1**.

---

## 🚀 How to Use

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/dog-breed-classification.git](https://github.com/your-username/dog-breed-classification.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook dog_breed_classification.ipynb
    ```

---

## ⚙️ Dependencies

* TensorFlow 2.x
* TensorFlow Hub
* Pandas
* NumPy
* Matplotlib
* scikit-learn
