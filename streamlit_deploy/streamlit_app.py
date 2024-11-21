import streamlit as st
import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification


# Device and model setup
model_name = 'ai-forever/ruRoberta-large'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load tokenizer and class descriptions
tokenizer = RobertaTokenizer.from_pretrained(model_name)
class_descriptions = pd.read_csv('files/trends_description.csv')


class BERTClass(torch.nn.Module):
    """
    Custom classification model based on Roberta.

    Attributes:
        bert_model (RobertaForSequenceClassification): Pretrained Roberta model for classification tasks.
    """
    def __init__(self):
        """
        Initializes the classification model by loading a pretrained Roberta model
        with custom configuration for multi-label classification.
        """
        super(BERTClass, self).__init__()
        self.bert_model = RobertaForSequenceClassification.from_pretrained(
            'metanovus/ruroberta-ecom-tech-best',
            return_dict=True,
            problem_type='multi_label_classification',
            num_labels=50
        )

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        """
        Overrides the forward method to process input data through the model.

        Args:
            input_ids (torch.Tensor): Tensor of token IDs.
            attn_mask (torch.Tensor): Tensor of attention masks.
            token_type_ids (torch.Tensor): Tensor of token type IDs.

        Returns:
            torch.Tensor: Logits produced by the model.
        """
        output = self.bert_model(
            input_ids,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids
        )
        return output.logits


# Load the model and weights
model = BERTClass().to(device)
model.eval()


def get_embeddings(text: str) -> dict:
    """
    Creates embeddings for the input text using the Roberta tokenizer.

    Args:
        text (str): Input text for processing.

    Returns:
        dict: Dictionary containing tensors for input_ids, attention_mask, token_type_ids, and the original text.
    """
    inputs = tokenizer.encode_plus(
        text,
        None,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        return_token_type_ids=True,
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    item = {
        'input_ids': inputs['input_ids'].flatten(),
        'attention_mask': inputs['attention_mask'].flatten(),
        'token_type_ids': inputs['token_type_ids'].flatten(),
        'text': text
    }
    return item


def get_new_predictions(model: BERTClass, text: str) -> tuple:
    """
    Generates predictions for the input text.

    Args:
        model (BERTClass): Instance of the classification model.
        text (str): Input text to analyze.

    Returns:
        tuple: A tuple containing boolean predictions (active classes) and probabilities for all classes.
    """
    data = get_embeddings(text)
    ids = data['input_ids'].unsqueeze(0).to(device, dtype=torch.long)
    mask = data['attention_mask'].unsqueeze(0).to(device, dtype=torch.long)
    token_type_ids = data['token_type_ids'].unsqueeze(0).to(device, dtype=torch.long)

    with torch.no_grad():
        outputs = model(ids, mask, token_type_ids).logits
        prediction_probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        predictions = prediction_probs >= 0.45

    return predictions, prediction_probs


# Streamlit interface
st.title('Delivery Feedback Classification')
input_text = st.text_area('Enter your delivery feedback:')

if st.button('Analyze'):
    predictions, prediction_probs = get_new_predictions(model, input_text)

    # Display probabilities as a bar chart
    st.subheader('Probability Distribution by Classes')
    st.bar_chart(pd.DataFrame({
        'Classes': class_descriptions['trend'],
        'Probabilities': prediction_probs
    }).set_index('Classes'))

    # Show text with predicted classes
    st.subheader('Predicted Classes:')
    for idx, (pred, prob) in enumerate(zip(predictions, prediction_probs)):
        if pred:
            class_name = class_descriptions.iloc[idx]['trend']
            st.write(f"{class_name} (Probability: {prob:.2f})")
