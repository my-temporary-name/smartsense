import streamlit as st
import torch
import pickle
import faiss
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn as nn
import json
import random
import torch
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity


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





# Load FLAN-T5 model and tokenizer
flan_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
flan_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

# Function to load responses from a text file
def load_responses_from_txt(file_path):
    with open(file_path, 'r') as f:
        responses = f.readlines()
    # Strip newline characters and return as a list
    return [response.strip() for response in responses]

# Load research and student responses from text files
research_responses = load_responses_from_txt('data/research.txt')
student_responses = load_responses_from_txt('data/student.txt')

# Combine responses for FAISS index
all_responses = {
    "research": research_responses,
    "student": student_responses
}

# Create FAISS index
def create_faiss_index(responses):
    embeddings = []
    for category in responses:
        for response in responses[category]:
            input_ids = flan_tokenizer(response, return_tensors='pt').input_ids
            with torch.no_grad():
                embedding = flan_model.encoder(input_ids).last_hidden_state.mean(dim=1).numpy()
                embeddings.append((embedding, response))
    
    # Convert to numpy array for FAISS
    embedding_matrix = np.array([embedding[0] for embedding in embeddings]).squeeze()
    index = faiss.IndexFlatL2(embedding_matrix.shape[1])  # Use L2 distance
    index.add(embedding_matrix.astype('float32'))
    
    return index, embeddings

# Create the FAISS index for all responses
faiss_index, embeddings_with_responses = create_faiss_index(all_responses)

# Define a function to get the nearest response from FAISS
def get_nearest_response(query):
    input_ids = flan_tokenizer(query, return_tensors='pt').input_ids
    with torch.no_grad():
        query_embedding = flan_model.encoder(input_ids).last_hidden_state.mean(dim=1).numpy()
    
    distances, indices = faiss_index.search(query_embedding.astype('float32'), k=1)
    return [embeddings_with_responses[idx][1] for idx in indices[0]]

# Define a function to add header and footer for student and research emails
def format_response_with_header_footer(response, department="Department of [Something]"):
    header = "\nDear Sir/Ma'am,"
    footer = f"Regards,\n{department}"
    return f"{header}\n\n{response}\n\n{footer}"

# Define a professional response for corporate emails
def corporate_response():
    return "This is a sensitive email and will take time and personal attention to respond. Please bear with us."

# Define a function to get the nearest response and format it
def get_formatted_response(query, predicted_category):
    # If it's a corporate email, return a professional response
    if predicted_category.lower() == 'corporate':
        return corporate_response()
    
    # For student or research emails, get the nearest response and add header/footer
    nearest_response = get_nearest_response(query)
    department = "Department of [Something]"  # You can customize this or make it dynamic
    return format_response_with_header_footer(nearest_response, department)

# Main function for Streamlit app
def main():
    st.title("Email Classification and Response System")

    # Load models
    lstm_model, lstm_tokenizer = load_lstm_model()
    bert_model, bert_tokenizer = load_bert_model()

    # Choose model
    model_choice = st.radio(
        "Choose a model:",
        ('BERT', 'LSTM')
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
            "We are interested in collaborating with your university on an AI research project.",
            "We would like to offer internships to your top students in data science.",
            "Our company is interested in sponsoring a hackathon at your university.",
            "We are looking to organize a workshop on advanced machine learning techniques.",
            "Can we discuss a potential partnership for future projects in AI research?"
        ]
    }

    # Predefined Examples
    if email_choice == 'Predefined Examples':
        email_category = st.selectbox("Choose an email category:", list(predefined_examples.keys()))
        example_choice = st.selectbox(f"Choose a {email_category} example:", predefined_examples[email_category])
        email_text = example_choice
        st.write(f"Selected email: {email_text}")
    else:
        email_text = st.text_area("Enter the email text:")

    # When "Classify Email" button is pressed
    if st.button("Classify Email"):
        categories = {0: 'Student', 1: 'Corporate', 2: 'Research'}

        # BERT Prediction
        if model_choice == 'BERT':
            bert_pred = bert_predict(email_text, bert_model, bert_tokenizer)
            predicted_category = categories[bert_pred]
            st.write(f"BERT Prediction: {predicted_category}")

        # LSTM Prediction
        if model_choice == 'LSTM':
            lstm_pred = lstm_predict(email_text, lstm_model, lstm_tokenizer)
            predicted_category = categories[lstm_pred]
            st.write(f"LSTM Prediction: {predicted_category}")

        # st.write(f"Predicted Category: {predicted_category}")

        # Retrieve and display the formatted response based on the predicted category
        response = get_formatted_response(email_text, predicted_category)
        st.write(f"Suggested Response:\n{response}")

if __name__ == "__main__":
    main()