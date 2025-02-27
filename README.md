# 📊 Анализ пользовательских отзывов

### 🚀 Описание проекта
Этот проект предназначен для автоматического анализа пользовательских отзывов с целью определения их тональности (**позитивный / негативный**). Анализ проводится с помощью обработки текста, машинного обучения и визуализации данных.

## 📌 Функциональность
✔ Загрузка и обработка текстовых данных из CSV-файла  
✔ Автоматическая разметка отзывов с помощью **NLTK VADER**  
✔ Преобразование текста в числовые признаки с **TF-IDF**  
✔ Обучение модели **Logistic Regression (sklearn)**  
✔ Оценка точности предсказаний (`accuracy_score`)  
✔ Визуализация распределения отзывов через **Seaborn**  
✔ Тестирование модели на новых данных  

---

## 🛠 Используемые технологии
- **Python** – основной язык разработки
- **Pandas** – загрузка и обработка данных
- **NLTK (VADER)** – анализ тональности текста
- **Scikit-learn** – машинное обучение (логистическая регрессия)
- **Seaborn, Matplotlib** – визуализация результатов

---

## 📂 Установка и запуск проекта
### 🔹 1. Клонирование репозитория
```bash
git clone https://github.com/username/sentiment-analysis.git
cd sentiment-analysis
```
### 🔹 2. Установка зависимостей
Рекомендуется создать виртуальное окружение:
```bash
python -m venv venv
source venv/bin/activate  # для Linux/Mac
venv\Scripts\activate  # для Windows
```
Установить необходимые библиотеки:
```bash
pip install -r requirements.txt
```
### 🔹 3. Запуск анализа
```bash
python sentiment_analysis.py
```

---

## 📊 Визуализация данных
Проект включает график распределения отзывов (**позитивные/негативные**), который строится с помощью `seaborn`:
```python
sns.countplot(x=df["sentiment"])
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.title("Distribution of Sentiments in Reviews")
plt.show()
```
Пример графика:
![Sentiment Distribution](https://via.placeholder.com/600x300)

---

## 🔎 Проверка модели
Можно протестировать модель на новых отзывах:
```python
new_reviews = ["I love this product!", "This is the worst experience ever."]
X_new = vectorizer.transform(new_reviews)
predictions = model.predict(X_new)
for review, pred in zip(new_reviews, predictions):
    print(f"Review: {review} -> Sentiment: {'Positive' if pred == 1 else 'Negative'}")
```

---

