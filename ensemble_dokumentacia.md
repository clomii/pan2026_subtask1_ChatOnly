# Meta-Ensemble Architektúra pre PAN'25

Tento dokument vysvetľuje pokročilú stratégiu pre detekciu textov generovaných umelou inteligenciou. Zatiaľ čo samostatný N-Gram model dosahuje skvelé výsledky, pre najlepšie možné umiestnenie v reálnej súťaži (kde testovací dataset obsahuje skryté triky a obfuskácie) je absolútnou nutnosťou nasadiť **Ensemble Model** (kombináciu viacerých rozdielnych modelov).

Z tohto dôvodu sme vyvinuli skripty `train_transformer.py` a `ensemble_evaluator.py`.

---

## 1. Sémantický model: `train_transformer.py`

Tento skript učí hlbokú neurónovú sieť (tzv. Transformer) odlíšiť ľudský text od strojového. Používa model **DeBERTa-v3** od spoločnosti Microsoft.

### Prečo je to dobré?
* **Pochopenie významu (Sémantika):** Kým náš N-Gram model (*custom_model*) hľadá len presné štrukturálne zhody (časté bodky, čiarky a konkrétne frázy), DeBERTa číta text s ohľadom na kontext. Ak LLM model oklamete príkazom *"Napíš to slangovo a s chybami"*, N-Gram zlyhá, pretože nevidel vodoznak. DeBERTa ale odhalí neprirodzený strojový význam za týmto slangom.
* **Hardvérová optimalizácia pre bežné GPU:** Trénovať Deep Learning modely s 512 tokenmi na 23 000 textoch bežne vyžaduje masívne serverové grafiky. V tomto skripte sú zapnuté dve dôležité techniky, vďaka ktorým to pobeží aj na kartách s 12GB VRAM (napr. RTX 4070 Ti):
  * `fp16=True`: Zapína 16-bitovú polovičnú presnosť výpočtov. Výrazne to znižuje spotrebu pamäte a dramaticky zrýchľuje procesor Tensor jadier vo vašej Nvidia karte.
  * `gradient_accumulation_steps=2`: Model trénuje v "minidávkach" po 8 textoch (aby nepretiekol pamäťou), no aktualizuje svoje neuróny až po 16 textoch. Dosiahneme tak lepšiu stabilitu bez pádu pre nedostatok VRAM.

### Ako to spustiť v Dockeri?
Keďže model potrebuje knižnice z rodiny Hugging Face, spúšťa sa zvnútra oficiálneho kontajnera pomocou príkazu:

```powershell
docker run --rm -it --gpus=all --entrypoint /bin/bash `
  -e HF_HOME=/hf_cache `
  -v ${PWD}\.hf_cache:/hf_cache `
  -v ${PWD}:/app `
  -w /app `
  ghcr.io/pan-webis-de/pan25-generative-authorship-baselines `
  -c "python3 -m pip install datasets sentencepiece accelerate protobuf && python3 train_transformer.py"
```

---

## 2. Finálne prepojenie: `ensemble_evaluator.py`

Toto je "mozog", ktorý na konci všetko spája dokopy. Namiesto spoliehania sa len na jeden konkrétny prístup, tento skript zhromaždí dáta z troch úplne odlišných paradigiem a vytvorí **Meta-Ensemble stratégiu**.

### Ako to funguje?
Zoberie výstupné `.jsonl` súbory s pravdepodobnosťami a použije techniku **Weighted Soft Voting** (Vážené spriemerovanie pravdepodobností). 

Skript zlúči názory troch "expertov":
1. **N-Gram Model (50% váha):** Náš rýchly model vynikajúci na mechanické štrukturálne vodoznaky a formátovanie interpunkcie.
2. **DeBERTa Model (30% váha):** Transformer excelentný v odhaľovaní zvláštnej sémantiky a kontextu aj v prípade, že je text štylisticky obfuskovaný (zamaskovaný).
3. **Binoculars Baseline (20% váha):** Zero-Shot štatistický detektor perplexity. Ten vôbec nevie, o čom text je, iba meria, či dané slová zapadajú do matematickej rovnice bežného LLaMA/Falcon modelu. Ak je to príliš "predvídateľné", určite to písala AI.

Spriemerovaním týchto hodnôt vznikne ultimátny finálny skórovací systém, ktorý automaticky prejde hodnotiacim procesom PAN'25 (vyhodnotenie ROC-AUC, Brier skóre atď.).

### Ako to spustiť?
Tento skript už nepotrebuje zložité AI knižnice ani Docker. Stačí ho spustiť v bežnom systéme (napríklad cez Python alebo len kliknutím vo vašom vývojovom prostredí), akonáhle máte vygenerované `.jsonl` súbory z predchádzajúcich modelov:

```powershell
python ensemble_evaluator.py
```

---

## 3. Presný návod Krok za Krokom (Step-by-Step)

Pre dosiahnutie maximálneho Meta-Ensemble skóre musíte vygenerovať všetky tri `.jsonl` súbory s predikciami a na záver ich spojiť. Postupujte presne v tomto poradí:

**Krok 1: Vytvorenie N-Gram predikcií**
* Najprv vytrénujete náš rýchly Logistický model, ktorý vytvorí štrukturálnu analýzu textu.
* **Príkaz (vo Windows PowerShell):**
  ```powershell
  python train_custom_model.py
  ```
* **Výsledok:** V priečinku sa objaví súbor `custom_model.jsonl`.

**Krok 2: Vytvorenie Binoculars predikcií**
* Získate perplexity skóre z masívnych LLM modelov. Tento proces beží v kontajneri s našou záplatou na pamäť, ktorá zamedzí pádu pri presune tenzorov.
* **Príkaz:**
  ```powershell
  docker run --rm --gpus=all -e HF_HOME=/hf_cache -v ${PWD}\.hf_cache:/hf_cache -v ${PWD}\data\val.jsonl:/val.jsonl -v ${PWD}:/out -v ${PWD}\pan25_genai_baselines\pan25_genai_baselines\binoculars.py:/usr/local/lib/python3.12/dist-packages/pan25_genai_baselines/binoculars.py ghcr.io/pan-webis-de/pan25-generative-authorship-baselines binoculars /val.jsonl /out
  ```
* **Výsledok:** V priečinku sa objaví súbor `binoculars.jsonl`. *(Môže to chvíľu trvať)*.

**Krok 3: Vytvorenie DeBERTa predikcií**
* Vytrénujete sémantický Deep Learning model. Na stiahnutie knižníc a spustenie tréningu využijeme rovnaký Docker kontajner, do ktorého pošleme dávkový príkaz.
* **Príkaz:**
  ```powershell
  docker run --rm -it --gpus=all --entrypoint /bin/bash -e HF_HOME=/hf_cache -v ${PWD}\.hf_cache:/hf_cache -v ${PWD}:/app -w /app ghcr.io/pan-webis-de/pan25-generative-authorship-baselines -c "python3 -m pip install datasets sentencepiece accelerate protobuf && python3 train_transformer.py"
  ```
* **Výsledok:** Po dokončení epoch tréningu sa vytvorí súbor `deberta_model.jsonl`.

**Krok 4: Finálne zlúčenie a Vyhodnotenie (Meta-Ensemble)**
* Akonáhle máte v priečinku všetky tri súbory (`custom_model.jsonl`, `binoculars.jsonl`, `deberta_model.jsonl`), prichádza na rad zlúčenie. Skript ich načíta, spočíta im vážený priemer (Weighted Soft Vote) a vytlačí finálne prekonávajúce ROC-AUC skóre.
* **Príkaz (vo Windows PowerShell):**
  ```powershell
  python ensemble_evaluator.py
  ```
* **Výsledok:** Vytvorí sa ultimátny súbor `ensemble_model.jsonl` a v termináli uvidíte oficiálne metriky pre PAN'25.
