FROM agrigorev/zoomcamp-model:3.8.12-slim

RUN pip install pipenv

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["app.py", "./"]

EXPOSE 9697

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9697", "app:app"]