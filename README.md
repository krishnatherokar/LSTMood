# LSTMood

**LSTMood** is a deep learning-based sentiment analysis and mood detection project. It uses Long Short-Term Memory (LSTM) networks to capture the sentiment of the text.

## Tech Stack

![Tensorflow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)

![Numpy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-333333?style=for-the-badge&logo=pandas&logoColor=white)


![Streamlit](https://img.shields.io/badge/streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

## Dataset
https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset

## Model Architecture
```python
model = Sequential([
    Vectorizer,
    Embedding(max_tokens, 256),
    LSTM(128),
    Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

## Accuracy

![Accuracy](res/Accuracy.png)

## Author
[![KAGGLE](https://img.shields.io/badge/Kaggle-krishnatherokar-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://kaggle.com/krishnatherokar/)