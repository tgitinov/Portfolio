# Проект по прогнозированию количества заказов

Компания-перевозчик собрала исторические данные о заказах такси в аэропортах, количестве заказов за каждые 10 минут. Чтобы привлекать больше водителей в период пиковой нагрузки, нужно спрогнозировать количество заказов такси на следующий час.

## Цель

Построить модель машинного обучения на датасете временного ряда.

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

### Подготовка
 - получаем общее представление о данных и их структуре, делаем ресемплирование в час
### Анализ данных
 - изучаем тренд, сезонность, остатки
### Обучение моделей
 - создаем признаки, делим выборки, строим и обучаем модели
### Тестирование моделей
 - тестируем модели на тестовой выборке, выводы