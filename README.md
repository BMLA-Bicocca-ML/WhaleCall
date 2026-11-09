# Marine Bioacoustic Binary Pipeline

Pipeline di pre-processing e gestione dataset per la classificazione binaria di bioacustica marina (Mammiferi Marini vs Rumore di Fondo) tramite spettrogrammi Mel in formato tensore, progettata per l'addestramento di reti neurali (SNN).

---

## Panoramica del Progetto
Questo repository contiene gli script e la struttura dati per il caricamento, la normalizzazione e l'elaborazione di file audio subacquei. 
L'approccio implementato è **binario globale**:
**Classe 1 (Mammiferi Marini):** Comprende un totale di 32 specie di mammiferi marini estratte da dataset dedicati.
**Classe 0 (Rumore di Fondo):** Estratto da dataset di riferimento (ShipsEar su Hugging Face).

### Nota Applicativa ("Rete Sentinella")
Questa pipeline è pensata non solo per l'addestramento diretto, ma anche per fungere da modello binario "sentinella" (a bassissimo consumo energetico per dispositivi embedded o microfoni subacquei h24). Una volta rilevata la presenza di un animale marino tramite questa rete, è possibile attivare una seconda rete più complessa per l'identificazione della singola specie tra le 32 disponibili.

---

## Caratteristiche del Dataset
**Formato di input:** File `.parquet` contenenti audio grezzi.
**Formato di output:** Spettrogrammi Mel normalizzati in tensori PyTorch con shape fissa `[1, 64, 128]`.
**Distribuzione e Bilanciamento:** Circa **27.098 chunk totali**, suddivisi in ~16.508 campioni per i mammiferi marini e ~10.590 campioni per il rumore di fondo, un rapporto vicino al 60/40.
**Normalizzazione:** Parametri di normalizzazione Min-Max (`db_min = -100.00`, `db_max = 14.00`) calcolati in modo **assoluto e globale** sull'intero dataset (specie + rumore). Questo garantisce che tutte le registrazioni mantengano la stessa scala energetica, evitando distorsioni o fenomeni di clipping sugli spettrogrammi.

---

## Struttura del Repository

```text
├── Dataset.py                       # Classe PyTorch Dataset per l'indicizzazione e il caricamento dei tensori
├── elaborazione_dati_parquet.py     # Script con la logica di trasformazione e normalizzazione globale
├── main.py                          # Script di test e avvio per la verifica del DataLoader e delle distribuzioni
├── train1.parquet                   # File parquet train 1 (specie marine)
├── train2.parquet                   # File parquet train 2 (specie marine)
├── train3.parquet                   # File parquet train 3 (specie marine)
├── train_rumore_1_labelled.parquet  # File parquet di rumore di fondo etichettato
└── README.md                        # Documentazione del progetto