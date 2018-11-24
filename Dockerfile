FROM ubuntu:latest
MAINTAINER Xinxin Tang "xinxin.tang92@gmail.com"

ARG DEBIAN_FRONTEND=noninteractive
ARG CRAN_MIRROR=https://cran.revolutionanalytics.com/

RUN apt-get update -y
RUN apt-get install -y tzdata \ 
                       python3-pip \
                       python3-dev \ 
                       build-essential \
                       r-base \
                       r-base-dev \ 
                       sudo       

COPY . /app
WORKDIR /app

RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3"]
CMD ["app.py"]
