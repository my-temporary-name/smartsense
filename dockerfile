FROM python:3.11.10-slim-bookworm

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "user_interface.py"]
