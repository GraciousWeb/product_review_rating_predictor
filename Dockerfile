FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY predictor.pkl tfidf.pkl new_app.py ./
EXPOSE 8501
CMD ["streamlit", "run", "new_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]