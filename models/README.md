# Optional serialized models

`predict.py` always uses the packaged DeBERTa checkpoint from
`deberta_results/checkpoint-2964`.

If `models/ngram_pipeline.joblib` exists, `predict.py` also loads it and blends it
with DeBERTa. Generate it with:

```bash
python3 train_custom_model.py --model-out models/ngram_pipeline.joblib
```
