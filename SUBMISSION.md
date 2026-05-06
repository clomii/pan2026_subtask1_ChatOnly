# PAN 2025 Subtask 1 Submission

This repository now contains a TIRA-compatible inference entrypoint:

```bash
python3 /app/predict.py $inputDataset/dataset.jsonl $outputDir
```

The script reads a JSONL file with `id` and `text`, then writes one prediction
file named `prediction.jsonl` to the output directory. Each line has this form:

```json
{"id": "sample-id", "label": 0.731}
```

## Local Docker smoke test

PowerShell:

```powershell
docker build -t pan25-subtask1 .
New-Item -ItemType Directory -Force .\submission_out | Out-Null
docker run --rm `
  -v ${PWD}\data\val.jsonl:/input/dataset.jsonl `
  -v ${PWD}\submission_out:/out `
  pan25-subtask1 /input/dataset.jsonl /out
```

Validate the output:

```powershell
docker run --rm `
  -v ${PWD}\data\val.jsonl:/input/dataset.jsonl `
  -v ${PWD}\submission_out:/out `
  --entrypoint python3 `
  pan25-subtask1 /app/validate_submission.py /input/dataset.jsonl /out/prediction.jsonl
```

## TIRA command

Use this command for the PAN Subtask 1 code submission:

```bash
python3 /app/predict.py $inputDataset/dataset.jsonl $outputDir
```

For a dry run with the TIRA client:

```bash
tira-cli code-submission --dry-run \
  --path . \
  --task generative-ai-authorship-verification-panclef-2025 \
  --dataset pan25-generative-ai-detection-smoke-test-20250428-training \
  --command 'python3 /app/predict.py $inputDataset/dataset.jsonl $outputDir'
```

## N-gram blend

The Docker image is ready with the DeBERTa checkpoint and the stronger
serialized N-gram model at `models/ngram_pipeline.joblib`. If this model ever
needs to be regenerated, run:

```bash
python3 train_custom_model.py --model-out models/ngram_pipeline.joblib
```

`predict.py` automatically blends this model with DeBERTa. If the file is
missing, the submission still runs with DeBERTa only.
