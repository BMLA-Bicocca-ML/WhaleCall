import io
import pandas as pd
import soundfile
import torch
import torchaudio

# Parametri fissi
TARGET_SAMPLE_RATE = 16000
DURATA_TARGET_SECONDI = 1.0
CAMPIONI_TARGET = int(TARGET_SAMPLE_RATE * DURATA_TARGET_SECONDI)


def process_audio_bytes(byte_audio):
    """Elabora i byte grezzi di un file .Parquet (come scaricabili da huggingface) restituendo la lista dei tensori degli spettrogrammi.
    """
    try:
        # Leggiamo i byte in memoria usando io.BytesIO
        audio_array, sample_rate = soundfile.read(io.BytesIO(byte_audio), dtype="float32")

        # Conversione da NumPy a Tensor
        forma_onda = torch.from_numpy(audio_array)

        # Monocanale
        if forma_onda.ndim == 1:
            forma_onda = forma_onda.unsqueeze(0)
        # Multicanale
        else:
            forma_onda = forma_onda.transpose(0, 1)

    except Exception as errore:
        print(f"Audio inleggibile ignorato: {errore}")
        return []

    # Conversione in mono se necessario
    if forma_onda.shape[0] > 1:
        forma_onda = forma_onda.mean(dim=0, keepdim=True)

    # Ricampionamento
    if sample_rate != TARGET_SAMPLE_RATE:
        ricampionatore = torchaudio.transforms.Resample(sample_rate, TARGET_SAMPLE_RATE)
        forma_onda = ricampionatore(forma_onda)

    num_campioni = forma_onda.shape[1]
    lista_matrici = []

    # Da audio corto a padding
    if num_campioni < CAMPIONI_TARGET:
        pad_dimensione = CAMPIONI_TARGET - num_campioni
        forma_onda_paddata = torch.nn.functional.pad(forma_onda, (0, pad_dimensione))
        matrice = estrai_spettrogramma(forma_onda_paddata, TARGET_SAMPLE_RATE)
        lista_matrici.append(matrice)

    # DA audio lungo a chunking con padding sull'ultimo pezzetto
    else:
        passo = CAMPIONI_TARGET
        for inizio in range(0, num_campioni, passo):
            fine = min(inizio + CAMPIONI_TARGET, num_campioni)
            chunk = forma_onda[:, inizio:fine]

            if chunk.shape[1] < CAMPIONI_TARGET:
                pad_dimensione = CAMPIONI_TARGET - chunk.shape[1]
                chunk = torch.nn.functional.pad(chunk, (0, pad_dimensione))

            matrice = estrai_spettrogramma(chunk, TARGET_SAMPLE_RATE)
            lista_matrici.append(matrice)

    return lista_matrici


def estrai_spettrogramma(forma_onda, sample_rate):

    trasformazione_mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=400,hop_length=122, n_mels=64, center=False,)

    spettrogramma = trasformazione_mel(forma_onda)
    conversione_db = torchaudio.transforms.AmplitudeToDB(stype="power")
    spettrogramma = conversione_db(spettrogramma)

    # Il min e max sono stati trovati osservendo il min e massimo dei decibel dello spettrogramma mel, il min è il min_ass e il max e il 99 percentile
    min_val = -100.0
    max_val = 14.0

    range_val = max_val - min_val
    if range_val < 1e-6:
        spettrogramma_norm = torch.zeros_like(spettrogramma)
    else:
        spettrogramma_norm = (spettrogramma - min_val) / range_val

    return spettrogramma_norm

