# Проект по определению перспективного тарифа для телеком компании (SDA)

Коммерческий департамент федерального оператора сотовой связи хочет скорректировать рекламный бюджет, чтобы понять какой тариф приносит больше прибыли. Из данных имеется небольшая выборка клиентов с базовой информацией: тариф, количество звонков, сообщений и пр.
  
*<h3 style="color:green;">Статус: завершен</h3>*

## Цель

Проанализировать поведение клиентов и сделать вывод — какой тариф лучше.

## Используемые библиотеки

Стэк: Python
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

### Исследовательский анализ данных
 - анализ дата сета и целевого признака

### Обучение модели
 - создание и обучение свёрточной нейронной сети с основой от ResNet50
 
### Анализ модели
 - разбор модели
 
## Результат

Проведен анализ.