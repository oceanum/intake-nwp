FROM ubuntu:22.04

RUN echo "--------------- Installing system packages ---------------" &&\
    apt update && apt upgrade -y && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt -y install \
        curl \
        git \
        libeccodes-dev \
        python3-pip && \
    apt clean all

RUN echo "--------------- Installing python packages ---------------"
ENV REPO="/source/intake-nwp"
RUN mkdir -p $REPO

COPY pyproject.toml README.rst HISTORY.rst $REPO/
COPY intake_nwp $REPO/intake_nwp
COPY tests $REPO/tests
RUN cd $REPO && \
    pip install -U pip && \
	pip install -e . --no-cache-dir
