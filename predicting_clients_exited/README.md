# Проект по прогнозированию оттока клиентов

Оператор связи хочет научиться прогнозировать отток клиентов. Если выяснится, что пользователь планирует уйти, ему будут предложены промокоды и специальные условия.
Имеются данные от команды оператора такие как: персональные данные о некоторых клиентах, информация об их тарифах и договорах.

## Цель

Построить прототип модели машинного обучения для предсказания оттока клиентов.

## Используемые библиотеки

```python
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
```

## Этапы проекта

### 1. Знакомство с данными
#### 1.1 Смотрим на файл `contract.csv`
#### 1.2 Смотрим на файл `personal.csv`
#### 1.3 Смотрим на файл `internet.csv`
#### 1.4 Смотрим на файл `phone.csv`

### 2. Исследовательский анализ данных
### 3. Предобработка данных
####3.1 Подготовка данных