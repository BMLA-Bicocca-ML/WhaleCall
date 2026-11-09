import pandas as pd
import torch
from torch.utils.data import Dataset
from elaborazione_dati_parquet import process_audio_bytes


class MarineDatasetBinary(Dataset):
    def __init__(self, liste_file_specie, liste_file_rumore):
        """
        Dataset binario per più file Parquet.
        - liste_file_specie: lista di percorsi dei file con i mammiferi (Label 1)
        - liste_file_rumore: lista di percorsi dei file con il rumore di fondo (Label 0)
        """
        self.dfs = []
        self.campioni_index = []

        # Carichiamo e mettiamo indice dei file delle specie (Label 1)
        print("Indicizzazione file specie marine (Label 1)...")
        for percorso in liste_file_specie:
            df = pd.read_parquet(percorso)
            df_idx = len(self.dfs)
            self.dfs.append(df)

            for idx, row in df.iterrows():
                byte_audio = row.audio["bytes"]
                matrici_temporanee = process_audio_bytes(byte_audio)
                for chunk_idx in range(len(matrici_temporanee)):
                    self.campioni_index.append((df_idx, idx, chunk_idx, 1))

        # Carichiamo e mettiamo indice dei file di rumore (Label 0)
        print("Indicizzazione file rumore di fondo (Label 0)...")
        for percorso in liste_file_rumore:
            df = pd.read_parquet(percorso)
            df_idx = len(self.dfs)
            self.dfs.append(df)

            for idx, row in df.iterrows():
                byte_audio = row.audio["bytes"]
                matrici_temporanee = process_audio_bytes(byte_audio)
                for chunk_idx in range(len(matrici_temporanee)):
                    self.campioni_index.append((df_idx, idx, chunk_idx, 0))

        print(f"Totale complessivo chunk indicizzati: {len(self.campioni_index)}")

    def __len__(self):
        return len(self.campioni_index)

    def __getitem__(self, idx):
        df_idx, riga_idx, chunk_idx, label = self.campioni_index[idx]
        row = self.dfs[df_idx].iloc[riga_idx]
        
        byte_audio = row.audio["bytes"]
        matrici = process_audio_bytes(byte_audio)
        
        spettrogramma = matrici[chunk_idx]
        return spettrogramma, torch.tensor(label, dtype=torch.float32)