FROM python:3.6
ENV PYTHONUNBUFFERED=1

RUN mkdir -p /code
VOLUME /code
EXPOSE 8000

ADD . /code
WORKDIR /code
RUN pip install -r requirements.txt
