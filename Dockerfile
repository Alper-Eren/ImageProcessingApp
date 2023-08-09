FROM ubuntu:latest

RUN apt-get update 
RUN apt-get install -y python3 python3-pip

WORKDIR /app
COPY . /app

RUN pip3 install pyqt5

CMD ["python3", "main.py"]
