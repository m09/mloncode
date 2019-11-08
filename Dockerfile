FROM ubuntu:18.04

RUN mkdir /mloncode

WORKDIR /mloncode

ARG DEBIAN_FRONTEND=noninteractive

ENV MLONCODE_BBLFSHD_NAME mloncode_bblfshd
ENV MLONCODE_BBLFSHD_PORT 9432

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    apt-utils \
    ca-certificates \
    locales \
    && echo "en_US.UTF-8 UTF-8" > /etc/locale.gen \
    && locale-gen \
    && apt-get remove -y .*-doc .*-man >/dev/null \
    && apt-get autoremove --purge -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libxml2 \
    libxml2-dev \
    gcc \
    g++ \
    python3.7 \
    python3.7-dev \
    python3-distutils \
    wget \
    && wget -O - https://bootstrap.pypa.io/get-pip.py | python3.7 \
    && pip3.7 install https://download.pytorch.org/whl/cpu/torch-1.3.0%2Bcpu-cp37-cp37m-linux_x86_64.whl \
    && pip3.7 install dgl \
    && pip3.7 install --no-cache-dir -r requirements.txt \
    && apt-get remove -y \
    libxml2-dev \
    gcc \
    g++ \
    python3.7-dev \
    wget \
    && apt-get remove -y .*-doc .*-man >/dev/null \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY setup.py .
COPY README.md .
COPY mloncode ./mloncode

RUN pip3.7 install --no-cache-dir .

ENTRYPOINT ["mloncode"]
