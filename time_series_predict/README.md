# Проект по прогнозированию количества заказов

Оператор связи хочет научиться прогнозировать отток клиентов. Если выяснится, что пользователь планирует уйти, ему будут предложены промокоды и специальные условия.
Имеются данные от команды оператора такие как: персональные данные о некоторых клиентах, информация об их тарифах и договорах.

## Цель

Построить прототип модели машинного обучения для предсказания оттока клиентов.

## Используемые библиотеки

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from statsmodels.graphics.tsaplots import plot_acf
```

## Этапы проекта

### Знакомство с данными
 - получаем общее представление о данных и их структуре
### Исследовательский анализ данных
 - анализ ежемесечной платы за услуги
### Предобработка данных
 - подготовка данных, приводим к корректному типу, убираем пустые поля, добавляем целевой признак, анализ количественных признаков, убираем лишние признаки, кодируем категориальные признаки
### Обучение моделей
 - создаем и обучаем на обучающей выборке пять регресионных моделей (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, LGBMClassifier, CatBoostClassifier), определяем лучшую по метрике AUC-ROC.
### Финальное тестирование
 - тестируем модель на тестовой выборке, выводы