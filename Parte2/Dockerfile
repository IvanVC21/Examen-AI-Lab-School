FROM python:3.10.11-slim-buster

WORKDIR /examen_parte2

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY api.py  .

COPY /lasso_opt_model.pkl  .

EXPOSE 80

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "80"]