# Email Classification for University HODs

## Student Information
- **Name:** ANKIT YADAV
- **Qualification :** MTech ( CSE ) with research focus on NLP

## Project Overview
This project implements an AI-powered Email Management System for Head of Departments (HODs) at a university. The system classifies incoming emails into different categories and handles them accordingly, streamlining the email management process.

## Project Flowchart
![flow chart](mermaid.png)
## Model Selection and Approach

### Models Implemented
1. **Fine-tuned Pretrained Model:** bert-base-uncased
2. **Custom Neural Network:** LSTM

### Rationale for Model Selection

1. **Contextual Understanding**: BERT’s bidirectional architecture captures the full context of emails, which is essential for distinguishing between similar categories like student, corporate, and research emails.
2. **Efficient and Effective**: The base model (12 layers, 110M parameters) balances strong performance with manageable computational costs, making it suitable for most classification tasks.
3. **Fine-Tuning Capability**: BERT can be fine-tuned easily on specific tasks like email classification, allowing the model to adapt to nuances in different email types.
4. **Uncased Model**: Using the uncased version simplifies the model by ignoring case distinctions, which aren’t critical for email classification, and reduces vocabulary size.


## Dataset Information
- The dataset contains 632 synthetically generated emails, evenly distributed across three categories: **student**, **research**, and **corporate**.
- Emails were created using **ChatGPT** and **Claude** models, ensuring a balanced dataset for classification tasks.
- **70 prompts** each for **research** and **student** responses were generated using **Retrieval-Augmented Generation (RAG)** to assist with response generation.

## Model Performance

### Fine-tuned Pretrained Model
| Class      | Precision | Recall | F1-Score | Support |
|------------|-----------|---------|----------|---------|
| Student    | 0.97      | 0.97    | 0.97     | 33      |
| Corporate  | 0.94      | 0.94    | 0.94     | 51      |
| Research   | 0.93      | 0.93    | 0.93     | 43      |
| ---------- | --------- | ------- | -------- | ------- |
| Accuracy   |           |         | 0.94     | 127     |
| Macro Avg  | 0.95      | 0.95    | 0.95     | 127     |
| Weighted Avg| 0.94     | 0.94    | 0.94     | 127     |

### Custom Neural Network
| Category   | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| Student    | 0.97      | 0.95   | 0.96     | 41      |
| Corporate  | 0.92      | 0.95   | 0.94     | 38      |
| Research   | 0.96      | 0.96   | 0.96     | 48      |
| ---------- | --------- | -------| -------- | ------- |
| Accuracy   |           |        | 0.95     | 127     |
| Macro Avg  | 0.95      | 0.95   | 0.95     | 127     |
| Weighted Avg | 0.95    | 0.95   | 0.95     | 127     |




