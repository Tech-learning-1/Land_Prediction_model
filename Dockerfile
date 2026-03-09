From python:3.12-alpine

WORKDIR /app
COPY requirements.txt .

RUN python -m pip install --upgrade pip setup-tools wheel
RUN python -m pip install -r requirement.txt

COPY . .

EXPOSE 5001

CMD ["python", "app.py"]