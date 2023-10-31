FROM ultralytics/yolov5:latest-cpu

ENV PYTHONUNBUFFERED=True \
    PORT=9090 \
    WORKERS=2 \
    THREADS=4

# To match ultralytics/yolov5 WORKDIR
WORKDIR /usr/src/app
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY _wsgi.py model.py *.pt ./

CMD exec gunicorn --preload --bind :$PORT --workers $WORKERS --threads $THREADS --timeout 0 _wsgi:app
