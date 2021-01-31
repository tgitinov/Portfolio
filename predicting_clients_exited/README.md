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
 
 <div class="alert alert-block alert-success">
<b>Успех:</b> Если все сделано отлично
</div>

<div class="alert alert-block alert-info">
<b>Совет: </b> Если можно немного улучшить
</div>

<div class="alert alert-block alert-danger">
<b>Ошибка:</b> Если требуются исправления. Работа не может быть принята с красными комментариями.
</div>