ARG BASE_IMAGE=12.3.2-base-ubuntu22.04
FROM nvidia/cuda:${BASE_IMAGE}

ENV DEBIAN_FRONTEND=nonintercative

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-distutils python3-pip libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6


RUN ln -fs /usr/bin/python3.10 /usr/bin/python3 && \
    ln -fs /usr/bin/python3 /usr/bin/python
    
COPY . .

RUN pip install -r requirements.txt
