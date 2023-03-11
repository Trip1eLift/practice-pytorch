FROM continuumio/miniconda3

USER root

RUN bash -c "/opt/conda/bin/conda install jupyter -y"

RUN bash -c "mkdir /opt/notebooks"

# WORKDIR /app

COPY ./requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt

COPY ./src ./opt/notebooks/

EXPOSE 8888

CMD ["bash", "-c", \
  "/opt/conda/bin/jupyter notebook \
  --notebook-dir=/opt/notebooks --ip='*' --port=8888 \
  --no-browser --allow-root" \
]