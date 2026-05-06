FROM ghcr.io/pan-webis-de/pan25-generative-authorship-baselines

WORKDIR /app

RUN python3 -m pip config set global.break-system-packages true \
    && python3 -m pip install --no-cache-dir sentencepiece protobuf safetensors

COPY predict.py /app/predict.py
COPY validate_submission.py /app/validate_submission.py
COPY models/ /app/models/
COPY deberta_results/checkpoint-2964/ /app/models/deberta/

ENTRYPOINT ["python3", "/app/predict.py"]
