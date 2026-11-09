from collections import Counter
from torch.utils.data import DataLoader
from Dataset import MarineDatasetBinary

if __name__ == "__main__":
    # Liste dei file di specie marine e file di rumore
    file_specie = ["train1.parquet",  "train2.parquet", "train3.parquet"]
    
    file_rumore = ["train_rumore_1_labelled.parquet"]

    # Inizializziamo il dataset combinato
    dataset = MarineDatasetBinary(file_specie, file_rumore)

    # DataLoader con shuffle per mescolare specie e rumore casualmente
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    print(f"Dataset pronto! Totale batch per epoca: {len(dataloader)}")
# Controllo rapido nel main
# Conta le etichette di tutto il dataset
etichette = [label for _, _, _, label in dataset.campioni_index]
conteggio = Counter(etichette)
print(f"Distribuzione classi -> Rumore (0): {conteggio[0]} | Specie (1): {conteggio[1]}")

# Prende un singolo spettrogramma per verificare shape
spettrogramma, label = dataset[0]
print(f"Shape singolo tensore: {spettrogramma.shape}")