FROM ubuntu:latest

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip



COPY ./requirements.txt /requirements.txt
WORKDIR /
RUN pip install --requirement requirements.txt

RUN python3 -m nltk.downloader stopwords
RUN python3 -m nltk.downloader punkt
RUN python3 -m nltk.downloader wordnet

#ADD src /src
#ADD tests /tests
#ADD movie_classifier.py movie_classifier.py

COPY . /

RUN chmod +x tests/test_all.sh
RUN tests/test_all.sh

