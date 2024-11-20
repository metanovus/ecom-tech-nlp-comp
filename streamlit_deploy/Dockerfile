# Базовый образ с Python
FROM python:3.10.12

# Установка зависимостей системы
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Копирование всех файлов проекта в контейнер
COPY ecom_streamlit.py .
COPY requirements.txt .
COPY files/trends_description.csv files/
COPY files/ruroberta-ecom-tech-best.pth files/

# Установка Python-зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Установка порта для Streamlit
EXPOSE 8501

# Команда для запуска приложения
CMD ["streamlit", "run", "ecom_streamlit.py", "--server.port=8501", "--server.enableCORS=false"]
