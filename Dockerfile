FROM python:3.10-bookworm

WORKDIR /src

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY ./src .

CMD ["python3", "run.py"]