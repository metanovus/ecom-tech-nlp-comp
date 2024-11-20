import streamlit as st
import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification


# Настройка устройства и моделей
model_name = 'ai-forever/ruRoberta-large'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Загрузка токенизатора и описаний классов
tokenizer = RobertaTokenizer.from_pretrained(model_name)
class_descriptions = pd.read_csv('files/trends_description.csv')

# Определение класса модели
class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.bert_model = RobertaForSequenceClassification.from_pretrained(
            'metanovus/ruroberta-ecom-tech-best',
            return_dict=True,
            problem_type='multi_label_classification',
            num_labels=50
        )

    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.bert_model(
            input_ids,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids
        )
        return output.logits

# Загрузка модели и весов
model = BERTClass().to(device)
model.eval()

# Функции обработки и предсказания
def get_embeddings(text):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    return {
        'input_ids': inputs['input_ids'].squeeze(0),
        'attention_mask': inputs['attention_mask'].squeeze(0),
        'text': text
    }

def get_new_predictions(model, text):
    data = get_embeddings(text)
    
    ids = data['input_ids'].to(device, dtype=torch.long)
    mask = data['attention_mask'].to(device, dtype=torch.long)
    
    with torch.no_grad():
        outputs = model(input_ids=ids.unsqueeze(0), attention_mask=mask.unsqueeze(0))
        logits = outputs.logits
        
        prediction_probs = torch.sigmoid(logits).cpu().numpy().flatten()
        
        predictions = prediction_probs >= 0.45
    
    # Возвращаем предсказания и вероятности
    return predictions, prediction_probs

# Streamlit интерфейс
st.title('Классификация отзывов о доставке')
input_text = st.text_area('Введите отзыв о доставке:')

if st.button('Анализировать'):
    predictions, prediction_probs = get_new_predictions(model, input_text)

    # Отображение вероятностей как гистограммы
    st.subheader('Распределение вероятностей по классам')
    st.bar_chart(pd.DataFrame({
        'Классы': class_descriptions['trend'],
        'Вероятности': prediction_probs
    }).set_index('Классы'))

    # Показ текста с предсказанными классами
    st.subheader('Предсказанные классы:')
    for idx, (pred, prob) in enumerate(zip(predictions, prediction_probs)):
        if pred:
            class_name = class_descriptions.iloc[idx]['trend']
            st.write(f"{class_name} (Вероятность: {prob:.2f})")
