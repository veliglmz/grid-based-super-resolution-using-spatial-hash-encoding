FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive 

WORKDIR /app

COPY requirements.txt .

RUN apt update

RUN apt-get install -y python3.8 python3-pip

RUN python3.8 -m pip install -r requirements.txt

COPY . .

CMD ["python3.8", "main.py", "-c", "config.json"]