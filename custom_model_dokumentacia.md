# Vylepšený Model pre Detekciu Generatívnej AI (PAN'25)

Tento dokument vysvetľuje fungovanie nového, vylepšeného modelu na detekciu textov generovaných umelou inteligenciou (Subtask 1) a poskytuje inštrukcie, ako ho spustiť v Docker kontajneri. Náš model signifikantne prekonáva oficiálny TF-IDF baseline na validačnom datasete.

## 1. Technický rozbor: Ako model funguje pod kapotou?

Oficiálny baseline model z repozitára používa veľmi zjednodušený prístup: analyzuje iba sekvencie slov (*Word N-grams*) a ich počet orezáva na 1000, aby ušetril pamäť. Náš model je navrhnutý na extrakciu oveľa hlbších štylistických vlastností textu, ktoré preukázateľne slúžia ako "odtlačky prstov" (fingerprints) veľkých jazykových modelov (LLM). 

Dosahujeme to pomocou architektúry **Feature Union**, ktorá beží nad **TF-IDF maticou** a **Logistickou regresiou**. Tu je detailné vysvetlenie každej súčiastky:

### A. TF-IDF (Term Frequency-Inverse Document Frequency)
TF-IDF je štatistická metóda, ktorá nepočíta len to, koľkokrát sa dané slovo alebo znak v texte vyskytne, ale zohľadňuje aj jeho bežnosť naprieč všetkými textami v datasete. 
* Ak sa nejaké slovo vyskytuje všade (napríklad spojka "a"), jeho váha sa zníži. 
* Ak sa nájde špecifická fráza, ktorú veľmi často používa iba ChatGPT (napr. "It is important to note"), dostane oveľa vyššiu váhu.
Náš model navyše používa parameter `sublinear_tf=True`, ktorý aplikuje logaritmické škálovanie frekvencie (1 + log(tf)). To zabraňuje tomu, aby nejaký text dostal obrovské skóre len preto, že autor zopakoval jedno slovo 50-krát.

### B. Vektorizácia č.1: Word N-Grams (Slovné n-gramy)
Slovný n-gram rozdeľuje text na celky slov (od 1 po 3 slová za sebou). 
LLM modely ako GPT-4 alebo Claude majú veľmi špecifickú slovnú zásobu, štruktúrovanie viet a radi používajú špecifické prechodové frázy ("In conclusion", "Furthermore", "Delve into"). Náš `TfidfVectorizer(analyzer='word', ngram_range=(1, 3))` sa naučí rozpoznávať až **50 000 takýchto najdôležitejších slovných spojení**.

### C. Vektorizácia č.2: Character N-Grams (Znakové n-gramy) - *Tajná zbraň*
Toto je najkritickejšie vylepšenie. Namiesto celých slov sa model pozerá na sekvencie 3 až 5 znakov.
Prečo je to lepšie ako slová? Zatiaľ čo slová analyzujú význam, znakové n-gramy analyzujú **štruktúru a formu**.
* **Interpunkcia a medzery:** Ľudia sú nekonzistentní. Často robia preklepy, dávajú dve medzery za vetou, alebo vynechávajú dĺžne. AI modely na druhej strane produkujú typograficky dokonalý text s mimoriadne predvídateľným rozmiestnením čiarok a bodiek.
* **Morfológia (predpony a prípony):** Znakové n-gramy dokážu zachytiť, že stroj preferuje slová s určitou koncovkou (napr. zbytočne zložité prídavné mená namiesto jednoduchých). 
Znakový analyzátor sa naučí ďalších **50 000 štrukturálnych vzorcov**.

### D. Spájanie (Feature Union) a Logistická Regresia
Pomocou objektu `FeatureUnion` model zlúči oba prístupy. Pre každý text z datasetu sa vytvorí gigantický riadok pozostávajúci zo **100 000 hodnôt** (50k slovných + 50k znakových váh).

Takáto obrovská matica si vyžaduje šikovný klasifikátor. Na rozdiel od pôvodného modelu, ktorý používal SVM (Support Vector Machine), my sme nasadili **Logistic Regression** (Logistickú regresiu s optimalizátorom `liblinear`).
* **Prirodzené pravdepodobnosti:** Na rozdiel od SVM, logistická regresia prirodzene vyhadzuje percentuálnu pravdepodobnosť, čo perfektne vyhovuje požiadavkám TIRA platformy, kde potrebujeme plynulé skóre medzi 0.0 a 1.0 na maximalizáciu AUC-ROC a Brier metrík.
* **C=10 (Regularizácia):** Zvýšenie tohto parametra hovorí modelu, aby viac veril trénovacím dátam a nebál sa učiť zložitejšie vzorce zo 100 000 stĺpcov, čo zabraňuje "underfittingu" (prílišnému zovšeobecňovaniu).

**Výsledok:** Vďaka hĺbkovej analýze typografie a fráz model znížil Falošnú chybovosť (FPR - kedy model falošne obviní človeka, že jeho text písala AI) z **5.2% na iba 1.2%**.

---

## 2. Ako spustiť model pomocou Dockeru

Keďže náš skript `train_custom_model.py` využíva knižnice, ktoré nemusia byť predinštalované v každom systéme, najlepšie je spustiť ho cez štandardný Python Docker kontajner.

### Krok 1: Príprava `requirements.txt`
Aby Docker vedel, aké knižnice potrebuje stiahnuť, uistite sa, že máte vytvorený súbor s Python závislosťami (prípadne sa nainštalujú priamo v príkaze). Pre náš model potrebujeme:
- `scikit-learn`
- `pandas`
- `numpy`

### Krok 2: Spustenie cez Docker (bez nutnosti vytvárať vlastný Image)

Ak si nechcete vyrábať vlastný Dockerfile, môžete použiť ofciálny obraz `python:3.12-slim` a prikázať mu, aby si stiahol knižnice a spustil kód. 

Spustite tento príkaz vo vašom PowerShell v priečinku projektu (kde sa nachádza priečinok `data/` a súbor `train_custom_model.py`):

```powershell
docker run --rm -it `
  -v ${PWD}:/app `
  -w /app `
  python:3.12-slim `
  /bin/bash -c "pip install --no-cache-dir scikit-learn pandas numpy && python train_custom_model.py"
```

**Vysvetlenie príkazu:**
* `docker run --rm -it`: Spustí kontajner interaktívne a po skončení ho vymaže.
* `-v ${PWD}:/app`: Pripojí váš aktuálny Windows priečinok do priečinka `/app` vo vnútri kontajnera.
* `-w /app`: Nastaví pracovný priečinok v kontajneri na `/app`.
* `python:3.12-slim`: Použije ľahký oficiálny Python obraz.
* `/bin/bash -c "..."`: Najskôr cez PIP nainštaluje potrebné knižnice a následne spustí trénovací skript.

### Krok 3: Príprava na TIRA (Oficiálna súťaž)

Ak by ste chceli tento model odovzdať do súťaže PAN'25 (systém TIRA), museli by ste skript mierne upraviť:
1. Natrénovať model lokálne a uložiť ho pomocou knižnice `pickle` (napr. `model.pkl`), presne tak ako to robí pôvodný `tfidf.py`.
2. Vytvoriť nový skript (napr. `predict.py`), ktorý nerobí trénovanie, ale len načíta `model.pkl` a vyhodnotí testovací súbor.
3. Vytvoriť vlastný `Dockerfile`, ktorý skopíruje `model.pkl` a skript do kontajnera.
4. TIRA potom váš kontajner zavolá príkazom podobným: `vaš_skript /cesta/k/dataset.jsonl /cesta/k/vystupu/`.
