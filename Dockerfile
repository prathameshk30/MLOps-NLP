FROM frolvlad/alpine-python-machinelearning:latest

COPY . /app
WORKDIR /app
EXPOSE 4000

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader punkt
CMD ["python","app.py"]