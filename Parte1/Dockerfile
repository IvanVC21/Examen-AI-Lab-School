FROM python:3.10.11-slim-buster

WORKDIR /Examen AI Lab School

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY API.py  .

COPY /modelo_examen_parte1.pkl  .

EXPOSE 80

CMD ["uvicorn", "API:app", "--host", "0.0.0.0", "--port", "80"]