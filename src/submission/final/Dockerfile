FROM --platform=linux/amd64 pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

ENV PYTHONUNBUFFERED=1
ARG SUBMISSION_VERSION=local
LABEL mama-synth.submission.version="${SUBMISSION_VERSION}"

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

WORKDIR /opt/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install --no-cache-dir --requirement /opt/app/requirements.txt

COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user resources/model.ckpt /opt/app/model.ckpt

RUN printf "%s\n" "$SUBMISSION_VERSION" > /opt/app/submission_version.txt \
    && chown user:user /opt/app/submission_version.txt \
    && mkdir -p /output \
    && chown user:user /output

USER user

ENTRYPOINT ["python", "inference.py"]
