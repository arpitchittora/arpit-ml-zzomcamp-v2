FROM python:3.8.12-slim
FROM continuumio/miniconda3

# Create the environment:
COPY environment.yml .

RUN conda env create -f environment.yml

# Initialize conda in bash config fiiles:
RUN conda init bash

WORKDIR /app

SHELL ["conda", "run", "-n", "mlmidtermimdbrating", "/bin/bash", "-c"]

RUN echo "Make sure conda mlmidtermimdbrating is installed:"

COPY ["app.py", "model_1.0.bin", "entrypoint.sh", "./"]

EXPOSE 9696

ENTRYPOINT ["./entrypoint.sh"]