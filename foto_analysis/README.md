# Проект по анализу фотографий (Computer Vision)

## Задача

На основе набора данных из фотографий людей с указанием возраста, построить модель, которая по фотографии определит приблизительный возраст человека.

## Используемые библиотеки

```python
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, Flatten, Dense, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
```

## Этапы проекта

 - Исследовательский анализ данных
 - Обучение модели
 - Анализ модели / выводы
 
## 2. Обучение модели

 - создание и обучение свёрточной нейронной сети с основой от ResNet50.

## 2. Анализ модели / выводы

 - полученная модель имеет хорошую точность, метрика MAE = 5,9401. Нейронные сети отлично себя показыают на анализе графической информации, огромная заслуга результатов - в отточенной архитектуре модели, за основу которой мы брали ResNet50.


## 1. Исследовательский анализ данных

 - анализ дата сета и целевого признака
 
## 2. Обучение модели

 - создание и обучение свёрточной нейронной сети с основой от ResNet50.

## 2. Анализ модели / выводы

 - полученная модель имеет хорошую точность, метрика MAE = 5,9401. Нейронные сети отлично себя показыают на анализе графической информации, огромная заслуга результатов - в отточенной архитектуре модели, за основу которой мы брали ResNet50.