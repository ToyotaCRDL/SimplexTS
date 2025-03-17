FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get -y upgrade \
    && apt-get install -y --no-install-recommends \
    build-essential checkinstall apt-utils autoconf \
    sudo git nano curl wget tree tmux htop bmon iotop xsel xclip tzdata vim ca-certificates \
    tar zip unzip bzip2 xz-utils \
    python3-dev python3-venv \
    libssl-dev libffi-dev libbz2-dev libdb-dev libnss3-dev libreadline-dev libgdbm-dev \
    liblzma-dev libncursesw5-dev libsqlite3-dev zlib1g-dev uuid-dev \
    libc6-dev libglib2.0-0 libcurl4-openssl-dev libxml2-dev libxslt1-dev libjpeg-dev libpng-dev \
    libsm6 libxext6 libxrender1 libxrender-dev libgl1-mesa-glx libgl1-mesa-dev libopencv-dev \
    make cpp gcc g++ perl cmake ccache libtool texinfo dpkg-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get clean

RUN wget https://www.python.org/ftp/python/3.8.10/Python-3.8.10.tgz
RUN tar -xzvf Python-3.8.10.tgz
RUN cd Python-3.8.10/ && ./configure --enable-optimizations && make -j 12 && make altinstall
RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3.8 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.8 1
RUN update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.8 1
RUN update-alternatives --install /usr/bin/pip3 pip3 /usr/local/bin/pip3.8 1
RUN rm -f Python-3.8.10.tgz
RUN rm -rf Python-3.8.10/

ARG USER
RUN useradd -m -s /bin/bash ${USER}
USER ${USER}

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

WORKDIR /home/${USER}

COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt
RUN rm -f requirements.txt