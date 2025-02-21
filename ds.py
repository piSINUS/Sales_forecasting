import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Загрузка данных (примерный формат)
data = pd.read_csv("sales_data.csv")  # Файл с данными о продажах

data['date'] = pd.to_datetime(data['date'])  # Преобразуем дату в datetime

data = data.sort_values(by='date')  # Сортируем по времени

# 2. Создание новых признаков
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['dayofweek'] = data['date'].dt.dayofweek

# 3. Подготовка данных для обучения
features = ['year', 'month', 'day', 'dayofweek']
X = data[features]
y = data['sales']  # Целевая переменная - объем продаж

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Обучение модели Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Оценка качества модели
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"MAE: {mae}, RMSE: {rmse}")

# 6. Визуализация предсказаний с помощью Seaborn
plt.figure(figsize=(10,5))
sns.lineplot(x=range(len(y_test)), y=y_test.values, label="Actual")
sns.lineplot(x=range(len(y_pred)), y=y_pred, label="Predicted")
plt.legend()
plt.title("Прогноз продаж с использованием Random Forest")
plt.show()
