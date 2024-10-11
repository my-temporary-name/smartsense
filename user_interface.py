import streamlit as st
import torch
import pickle
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn as nn

# Define LSTM Model
class EmailClassificationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(EmailClassificationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                            bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        packed_output, (hidden, cell) = self.lstm(embedded)
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        output = self.fc(hidden)
        return self.softmax(output)

# Load LSTM model and tokenizer
# @st.cache_data(allow_output_mutation=True)
def load_lstm_model():
    lstm_model = EmailClassificationModel(vocab_size=30522, embedding_dim=128, hidden_dim=256, output_dim=3, n_layers=4, bidirectional=True, dropout=0.3)
    lstm_model.load_state_dict(torch.load('LSTM_model_save/lstm_model.pth', map_location=torch.device('cpu')))
    lstm_model.eval()
    with open('LSTM_model_save/tokenizer.pkl', 'rb') as f:
        lstm_tokenizer = pickle.load(f)
    return lstm_model, lstm_tokenizer

# Load BERT model and tokenizer
# @st.cache_data(allow_output_mutation=True)
def load_bert_model():
    bert_model = BertForSequenceClassification.from_pretrained('fine-tuned-bert-email', num_labels=3)
    bert_tokenizer = BertTokenizer.from_pretrained('fine-tuned-bert-email')
    return bert_model, bert_tokenizer

# Predict with LSTM
def lstm_predict(email, lstm_model, lstm_tokenizer):
    encoding = lstm_tokenizer.encode_plus(email, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    with torch.no_grad():
        outputs = lstm_model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs, dim=1).item()
    return preds

# Predict with BERT
def bert_predict(email, bert_model, bert_tokenizer):
    encoding = bert_tokenizer.encode_plus(email, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    with torch.no_grad():
        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1).item()
    return preds

# Main function
# Main function
def main():
    st.title("Email Classification System")

    # Load models
    lstm_model, lstm_tokenizer = load_lstm_model()
    bert_model, bert_tokenizer = load_bert_model()

    # Choose model
    model_choice = st.radio(
        "Choose a model:",
        ('BERT', 'LSTM', 'Both')
    )

    # Input: predefined examples or user input
    email_choice = st.radio(
        "Choose an email example or input your own:",
        ('Predefined Examples', 'Custom Input')
    )

    # Provide 5 examples for each category
    predefined_examples = {
        "Research Email": [
            "I would like to submit my findings on quantum computing to your department for review.",
            "Our team is working on a new AI model and we would like to collaborate with your lab.",
            "I need access to the recent research papers in machine learning. Can you help?",
            "We are conducting a study on renewable energy, and your expertise would be invaluable.",
            "Could you share the data from your latest experiment for cross-validation?"
        ],
        "Student Email": [
            "Can you please provide the schedule for the upcoming semester?",
            "I am having trouble registering for my classes, can you assist?",
            "I need help with my final project submission deadline.",
            "Could you explain the prerequisites for the Data Science course?",
            "Is there any guidance available for the upcoming exam?"
        ],
        "Corporate Email": [
            "We have positions for 6 chemistry students interested in industrial applications.",
            "We have full-time positions and internships available in HR consulting.",
            "Our company is interested in sponsoring a hackathon at your university.",
            "We are looking to organize a workshop on advanced machine learning techniques.",
            "Can we discuss a potential startup for future projects in BioTechnology?"
        ]
    }

    # Predefined Examples
    if email_choice == 'Predefined Examples':
        # Select category first
        email_category = st.selectbox("Choose an email category:", list(predefined_examples.keys()))
        
        # Select one of the five examples from the chosen category
        example_choice = st.selectbox(f"Choose a {email_category} example:", predefined_examples[email_category])
        email_text = example_choice  # Selected example text
        
        st.write(f"Selected email: {email_text}")  # Display selected email

    else:
        email_text = st.text_area("Enter the email text:")  # Custom input from user

    # When "Classify Email" button is pressed
    if st.button("Classify Email"):
        categories = {0: 'Student', 1: 'Corporate', 2: 'Research'}

        # BERT Prediction
        if model_choice == 'BERT' or model_choice == 'Both':
            bert_pred = bert_predict(email_text, bert_model, bert_tokenizer)
            st.write(f"BERT Prediction: {categories[bert_pred]}")

        # LSTM Prediction
        if model_choice == 'LSTM' or model_choice == 'Both':
            lstm_pred = lstm_predict(email_text, lstm_model, lstm_tokenizer)
            st.write(f"LSTM Prediction: {categories[lstm_pred]}")


if __name__ == "__main__":
    main()
