## 💫 ecom.tech x Deep Learning Scool Competition + Streamlit

Этот репозиторий содержит ноутбуки участия в [соревновании по NLP](https://ods.ai/competitions/dls_ecomtech/leaderboard/private). В репозитории два ноутбука и файл requirements.txt для отслеживания зависимостей.

## 🏆 Результаты (Вадим Самойлов)
- 9 место на публичном и приватном лидерборде (скор`0,547` и `0.546` соответственно);
- 4 место после проведения код-ревью:

![image](https://github.com/user-attachments/assets/7e2c09df-b5dc-4bd2-a25b-e9a3b6853cc4)

## 📖 Обзор репозитория

1. [**`Samoilov_Vadim_ecom_tech_NLP_Learning.ipybn`**](https://github.com/metanovus/ecom-tech-nlp-comp/blob/master/notebooks/final/Samoilov_Vadim_ecom_tech_NLP_Learning.ipynb): Инициализация и обработка данных, получения готовых весов модели.
2. [**`Samoilov_Vadim_ecom_tech_NLP_Prediction.ipybn`**](https://github.com/metanovus/ecom-tech-nlp-comp/blob/master/notebooks/final/Samoilov_Vadim_ecom_tech_NLP_Prediction.ipynb): Составление прогнозов через подгрузку весов модели.
3. [**`ecom_tech_comp_version_2.ipynb`**](https://github.com/metanovus/ecom-tech-nlp-comp/blob/master/notebooks/test/ecom_tech_comp_version_2.ipynb): Версия 2 с аутпутами первого ноутбука и плюс эксперимент с обучением на всех фичах (текст, теги и оценка).
4. [**Предобученная мной модель**](https://huggingface.co/metanovus/ruroberta-ecom-tech-best)

## 🔌 Как запустить локально Streamlit

Необходимо выполнить следующие команды:
  ```bash
  git clone https://github.com/metanovus/ecom-tech-nlp-comp.git
  cd ecom-tech-nlp-comp
  pip install requirements.txt
  streamlit streamlit_app.py
  ```

После выполнения кода Streamlit запустится в браузере по адресу: http://localhost:8501.

## 🧪 Мои эксперименты (читать не обязательно)

- Эксперименты с классическими ML-моделями (Наивный Байес, Логистическая регрессия, CatBoost и др.) показали низкий скор (`0.13`-`0.3`).
- Векторизация с `rubert-tiny2` слегка повысила результат (до `0.26`), а синтетические данные на базе GPT подняли его до `0.36`.
- CatBoost, даже с GPU и синтетикой, не оправдал ожиданий, и максимальный скор составил `0.26`.
- ОHE-теги (`tags`) сбивали модель, а стекинг не дал улучшения (`0.3`).
- Псевдолейбелинг тоже не сработал и привел к переобучению.
- В итоге я перешел к глубинным моделям Hugging Face. Лучшие результаты достигнуты с `ruRoberta-large` (`0.546`–`0.548` на лидерборде).
- Попытки с `ruT5-base` на Kaggle GPU провалились из-за ограничений по памяти, поэтому я остался на `ruRoberta-large`.

Итоги: я считаю, что вполне заслуженное 9 место в приватном и публичном лидерборде.

Что осталось нереализованным:
- Попытка обучить нейронную сеть вообще на всех данных (без деления на тренировочные и тестовые) - часто случалось, что скор вырастал;
- Улучшить генерируемость данных, тестировать больше моделей (кроме GPT, Claude), например, llama;
- Более серьёзно углубиться в Feature Engineering и фильтрацию признаков - вдруг какие-то эмбеддинги те самые gold фичи.

P.S.: если вдруг будет интересно, что за данные мне сгенерировала GPT-модель, то прикладываю в репозиторий файлы: 
- [`gpt4o_additional_data.csv`](https://github.com/metanovus/ecom-tech-nlp-comp/blob/master/synthetic_data/gpt4o_additional_data.csv) (один класс);
- [`gpt4o_additional_data_multiclass.csv`](https://github.com/metanovus/ecom-tech-nlp-comp/blob/master/synthetic_data/gpt4o_additional_data_multiclass.csv) (несколько классов - столбец `class`).

---

Спасибо за внимание!
