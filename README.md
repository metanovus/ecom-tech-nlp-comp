# ODS x ecom.tech NLP competition + Streamlit

Этот репозиторий содержит ноутбуки участия в [соревновании по NLP](https://ods.ai/competitions/dls_ecomtech/leaderboard/private). В репозитории два ноутбука и файл requirements.txt для отслеживания зависимостей.

## Результаты (Вадим Самойлов)
- 9 место на публичном и приватном лидерборде (скор`0,547` и `0.546` соответственно);
- 4 место после проведения код-ревью:

![image](https://github.com/user-attachments/assets/7e2c09df-b5dc-4bd2-a25b-e9a3b6853cc4)

## Обзор репозитория

1. [**`Samoilov_Vadim_ecom_tech_NLP_Learning.ipybn`**](https://github.com/metanovus/ecom-tech-nlp-comp/blob/master/Samoilov_Vadim_ecom_tech_NLP_Learning.ipynb): Инициализация и обработка данных, получения готовых весов модели.
2. [**`Samoilov_Vadim_ecom_tech_NLP_Prediction.ipybn`**](https://github.com/metanovus/ecom-tech-nlp-comp/blob/master/Samoilov_Vadim_ecom_tech_NLP_Prediction.ipynb): Составление прогнозов через подгрузку весов модели.

- 🔗 [Скачать веса предобученой модели](https://drive.google.com/file/d/1FBfKdnTpTEdcioNvNZ43ky2mkWhF-M3N/view?usp=drive_link)
- 🔗 [Скачать Docker-образ Streamlit](https://drive.google.com/file/d/1LUDDnpkZhBJv7KpZqaH3QioIYL2VXV3o/view?usp=sharing) (установлены уже все библиотеки)
- 🔗 [Скачать файлы Streamlit без Docker](https://drive.google.com/drive/folders/1_xf5iDOH1ZgKiDmM5pu7E9h6rEKugNVW?usp=sharing) (+ `requirements.txt`)

## Порядок работы во время код-ревью

1. **Настройте окружение**:
   - Убедитесь, что у вас установлен Python (желательно версии 3.10 и выше).
   - При необходимости установите библиотеки (файл приложен в репозитории к ноутбукам):
     ```bash
     pip install -r requirements.txt
     ```

3. **Начните с `Samoilov_Vadim_ecom_tech_NLP_Learning.ipybn`**:
   - Откройте ноутбук в необходимом окружении и выполните **поэтапно**(!!!)*  все ячейки.
   - Убедитесь, что все выполняется без ошибок и выполнено сохранение весов `best`-модели перед переходом к следующему шагу.

4. **Продолжите с `Samoilov_Vadim_ecom_tech_NLP_Prediction.ipybn`**:
   - После успешного выполнения первого ноутбука откройте второй ноутбук.
   - Этот ноутбук отвечает за обучение модели и оценку её производительности**. Выполните **поэтапно**(!!!)*  все ячейки.
     
_*_ _-_ поэтапно, потому что следует изменить пути ко всем данным, иначе будет ошибка. При желании можно сразу изменить все пути и нажать `Выполнить всё`.

_*_* _-_ известно, что при обучении и составлении прогнозов на GPU результаты могут различаться в независимости от того, зафиксированы ли seed или нет. Пофиксить это не всегда возможно. но результат всегда будет возвращать плюс-минус очень похожие результаты.

## Устранение неполадок или по желанию пропуск первого ноутбука

Если что-то пошло не так либо же не хочется тратить 2-3 часа на дообучение модели в ноутбуке `Samoilov_Vadim_ecom_tech_NLP_Learning.ipybn`, вы можете скачать предобученную модель по ссылке ниже (она была получена путём выполнения обучения первого ноутбука):

🔗 [Скачать веса предобученой модели](https://drive.google.com/file/d/1FBfKdnTpTEdcioNvNZ43ky2mkWhF-M3N/view?usp=drive_link)

Просто скачайте модель и поместите её у себя. **Обязательно** в ноутбуке `Samoilov_Vadim_ecom_tech_NLP_Prediction.ipybn` укажите путь до весов на моменте их загрузки в объявленную модель.

## Как запускать Streamlit

**Версия 1 (Docker-образ):**
- Скачайте [образ Docker](https://drive.google.com/file/d/1LUDDnpkZhBJv7KpZqaH3QioIYL2VXV3o/view?usp=sharing) (файл .tar должен называться `ecom_streamlit_app.tar`:

Для системы Linux:
- выполните следующую команду в терминале (где лежит файл .tar):
  ```bash
  docker load -i ecom_streamlit_app.tar
  docker run -p 8501:8501 ecom_streamlit_app
  ```
Для системы Windows:
- Скачайте, установите и запустите Docker Desktop для Windows
- Откройте PowerShell или Командную строку на Windows
- Перейдите в директорию, где был сохранен файл .tar
- Выполните команду для загрузки образа в Docker:
```powershell
docker load -i C:\path\to\ecom_streamlit_app.tar
```
- После успешной загрузки образа в Docker выполните команду для запуска контейнера:
```powershell
docker run -p 8501:8501 ecom_streamlit_app
```
- Docker создаст и запустит контейнер из образа.

**Версия 2 (без образа Docker):**
- По желанию можно скачать просто [папку с файлами](https://drive.google.com/drive/folders/1_xf5iDOH1ZgKiDmM5pu7E9h6rEKugNVW?usp=sharing), тогда необходимо выполнить следующие команды:
  ```bash
  pip install requirements.txt
  streamlit ecom_streamlit.py
  ```

После выполнения версии 1 или версии 2 сервис Streamlit запустится в браузере по адресу: http://localhost:8501.

## Обратная связь

Если у вас возникли вопросы или вы столкнулись с проблемами, вы можете связаться со мной в Telegram в любое время. Мой никнейм: @samoilov_vadim

## Мои эксперименты (читать не обязательно)

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
- [`gpt4o_additional_data.csv`](https://github.com/metanovus/ecom-tech-nlp-comp/blob/master/gpt4o_additional_data.csv) (один класс);
- [`gpt4o_additional_data_multiclass.csv`](https://github.com/metanovus/ecom-tech-nlp-comp/blob/master/gpt4o_additional_data_multiclass.csv) (несколько классов - столбец `class`).

---

Спасибо за внимание!
