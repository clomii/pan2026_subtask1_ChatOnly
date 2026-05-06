# Semestrálny Projekt: PAN 2025 - Generative Authorship Verification

Tento dokument slúži ako kompletná finálna dokumentácia k vytvorenému systému na detekciu textov generovaných umelou inteligenciou (AI vs. Človek). 

Základom projektu bola výzva **PAN'25 Subtask 1**, kde bolo nutné prekonať poskytnuté "baseline" modely (TF-IDF a Binoculars) vytvorením robustného, presného a pamäťovo efektívneho klasifikátora. Namiesto spoliehania sa na jednu izolovanú metódu sme navrhli a implementovali **State-of-the-Art Meta-Ensemble Architektúru**.

---

## 1. Architektúra: Prečo Meta-Ensemble?

Pri moderných veľkých jazykových modeloch (LLMs) je prakticky nemožné detegovať podvod len na základe jedného prístupu. Študenti alebo podvodníci používajú techniky "obfuskácie" (napríklad LLM dostane inštrukciu *"Napíš to slangovo a narob gramatické chyby"*).
Ak sa náš systém spolieha len na jeden parameter, nechá sa ľahko oklamať. Preto sme zostrojili **tri kompletne odlišné modely**, z ktorých každý sa pozerá na iný aspekt textu:

### 1.1 Štrukturálny Model (Logistická Regresia a N-Gramy)
* **Skript:** `train_custom_model.py`
* **Ako funguje:** Tento model si vôbec nepozerá význam slov. Namiesto toho hľadá rigidné mechanické "vodoznaky". Extrahovali sme celkovo **100 000 príznakov** pomocou techniky `FeatureUnion` – 50 000 slovných TF-IDF n-gramov a 50 000 znakových (character) n-gramov. Model odhaľuje mikroskopické pravidelnosti v dĺžke slabík, umiestňovaní čiarok a slovoslede, ktorými LLM trpia.
* **Výhoda:** Je extrémne rýchly a má masívnu úspešnosť na čistých textoch.

### 1.2 Sémantický Model (Deep Learning Transformer - DeBERTa-v3)
* **Skript:** `train_transformer.py`
* **Ako funguje:** Ide o hlbokú neurónovú sieť (Hugging Face). Namiesto štruktúry sa DeBERTa pozerá na skrytý sémantický kontext a logický tok myšlienok.
* **Výhoda:** Nedá sa oklamať zmenou štýlu (slangom alebo chybami), pretože sleduje významovú "plastickosť" textu. Bol hardvérovo optimalizovaný (`fp16` a `gradient_accumulation`) pre plynulý chod na 12GB VRAM bez Memory Overflow.

### 1.3 Štatistický Detektor Perplexity (Binoculars)
* **Skript:** `run_binoculars_fast.py` & upravený `binoculars.py`
* **Ako funguje:** Nameria "prekvapenie" modelu. Text sa vloží do dvoch LLM (Qwen2.5-3B a Qwen-Instruct). Ak model presne vie predpovedať nasledujúce slová (má nízku perplexitu / je málo prekvapený), znamená to, že text bol veľmi pravdepodobne vygenerovaný iným LLM, pretože stroje zdieľajú podobnú štatistickú pravdepodobnosť ukladania slov.
* **Výhoda:** Funguje spôsobom "Zero-shot", nepotrebuje sa učiť na tréningových dátach.

### 1.4 Mozog (Weighted Soft Voting)
* **Skript:** `ensemble_evaluator.py`
* Tieto tri modely vyprodukujú svoje vlastné predikcie a náš spájací algoritmus im pridelí percentuálnu váhu dôvery (**N-Gram:** 50%, **DeBERTa:** 30%, **Binoculars:** 20%). Tým sa eliminuje akékoľvek zlyhanie jednotlivca.
* **Dosiahnuté Skóre na validačnej sade:**
  * **ROC-AUC:** 1.000 (Dokonalá separácia)
  * **F1:** 0.994
  * **FPR:** 1.3 % (Len 1.3% ľudských textov falošne obvinených)

---

## 2. Návod: Ako celý systém spustiť lokálne

Aby ste získali finálny výsledok, je potrebné vygenerovať predikcie zo všetkých troch modelov na vašej validačnej sade (súbor `val.jsonl`). Otvorte Windows PowerShell a postupujte presne takto:

### Krok A: N-Gram (Tréning a Predikcia)
Vytrénujte N-Gram a vygenerujte `custom_model.jsonl`.
```powershell
python train_custom_model.py
```

### Krok B: DeBERTa (Sémantika)
Tento model vyžaduje veľa Linux knižníc. Spustíme ho priamo v oficiálnom baseline kontajneri. Skript si sám doinštaluje chýbajúci `protobuf` a `sentencepiece`. Výsledkom bude `deberta_model.jsonl`.
```powershell
docker run --rm -it --gpus=all --entrypoint /bin/bash `
  -e HF_HOME=/hf_cache `
  -v ${PWD}\.hf_cache:/hf_cache `
  -v ${PWD}:/app `
  -w /app `
  ghcr.io/pan-webis-de/pan25-generative-authorship-baselines `
  -c "python3 -m pip install datasets sentencepiece accelerate protobuf && python3 train_transformer.py"
```

### Krok C: Binoculars (Perplexita)
Na vyriešenie známeho úniku pamäte do RAM využívame patchnutý kód s ľahšími 3B modelmi a vlastným Progress Barom. Vytvorí sa súbor `binoculars_fast.jsonl`.
```powershell
docker run --rm -it --gpus=all --entrypoint /bin/bash `
  -e HF_HOME=/hf_cache `
  -v ${PWD}\.hf_cache:/hf_cache `
  -v ${PWD}:/app `
  -v ${PWD}\pan25_genai_baselines\pan25_genai_baselines\binoculars.py:/usr/local/lib/python3.12/dist-packages/pan25_genai_baselines/binoculars.py `
  -w /app `
  ghcr.io/pan-webis-de/pan25-generative-authorship-baselines `
  -c "python3 -m pip install datasets sentencepiece accelerate protobuf && python3 run_binoculars_fast.py"
```

*(Poznámka: v `ensemble_evaluator.py` si overte, že na riadku 27 načítavate názov `binoculars_fast.jsonl` namiesto pôvodného `binoculars.jsonl`)*.

### Krok D: Spojenie (Ensemble) a vyhodnotenie metrík
Spojí vygenerované `.jsonl` súbory z Krokov A, B a C.
```powershell
python ensemble_evaluator.py
```

---

## 3. Finalne odovzdanie (TIRA Platform)

Projekt je doplneny do TIRA-kompatibilneho stavu. Korektny vstupny bod je:

```bash
python3 /app/predict.py $inputDataset/dataset.jsonl $outputDir
```

Skript `predict.py` prijima presne dva povinne argumenty: vstupny JSONL subor a vystupny priecinok. Vystupom je jeden subor `prediction.jsonl`, kde kazdy riadok obsahuje `id` a pravdepodobnostne `label` v intervale 0.0 az 1.0.

Docker obraz sa builduje z korenoveho `Dockerfile`. Do obrazu sa bali natrenovany DeBERTa checkpoint z `deberta_results/checkpoint-2964`, inference skript `predict.py`, validator `validate_submission.py` a volitelny priecinok `models/`.

Zakladny smoke test:

```powershell
docker build -t pan25-subtask1 .
New-Item -ItemType Directory -Force .\submission_out | Out-Null
docker run --rm `
  -v ${PWD}\data\val.jsonl:/input/dataset.jsonl `
  -v ${PWD}\submission_out:/out `
  pan25-subtask1 /input/dataset.jsonl /out
```

Kontrola formatu:

```powershell
docker run --rm `
  -v ${PWD}\data\val.jsonl:/input/dataset.jsonl `
  -v ${PWD}\submission_out:/out `
  --entrypoint python3 `
  pan25-subtask1 /app/validate_submission.py /input/dataset.jsonl /out/prediction.jsonl
```

Volitelne je mozne este dotrenovat a ulozit N-gram pipeline:

```bash
python3 train_custom_model.py --model-out models/ngram_pipeline.joblib
```

Ak `models/ngram_pipeline.joblib` existuje pred buildom Docker obrazu, `predict.py` ho automaticky nacita a zmiesa s DeBERTa skore. Ak neexistuje, submission funguje samostatne iba s DeBERTa checkpointom.
