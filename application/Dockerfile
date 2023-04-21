FROM python:3.9-slim
WORKDIR /app
ADD ./requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -v -r /app/requirements.txt
ADD ./src/ /app

CMD [ "python", "./server.py" ]
