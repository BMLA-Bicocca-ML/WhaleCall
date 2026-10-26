"""
Spiking Neural Network (SNN) minimale per classificazione di spettrogrammi Mel.

Architettura: Linear -> LIF -> Linear -> LIF, applicata passo per passo
lungo l'asse temporale dello spettrogramma. L'output è il firing rate
medio (spike per time step) di ciascun neurone di uscita, usato come logit.
"""

# ============================================================
# PACCHETTI
# ============================================================
# !pip install snntorch      
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate   # gradienti surrogati per la backprop sugli spike


# ============================================================
# DEFINIZIONE DELLA SPIKING NEURAL NETWORK
# ============================================================
class MinimalSNN(nn.Module):
    """
    SNN a un solo hidden layer con neuroni Leaky Integrate-and-Fire (LIF).

    Parameters
    ----------
    beta : float, default 0.9
        Fattore di decadimento del potenziale di membrana (0 < beta < 1).
        Valori alti = memoria più lunga; U[t+1] = beta * U[t] + I[t+1].
    hidden : int, default 32
        Numero di neuroni dello strato nascosto.
    n_classes : int, default 2
        Numero di classi = numero di neuroni di output.
    num_steps : int, default 25
        Usato qui come pendenza (slope) del surrogato fast sigmoid.
        NB: non è il numero di time step della simulazione, che è
        determinato dalla lunghezza temporale dell'input (vedi forward).
    n_bandeMel : int, default 64
        Numero di bande Mel dello spettrogramma = dimensione di input
        del primo strato lineare.
    """

    def __init__(self, beta=0.9, hidden=32, n_classes=2, num_steps=25, n_bandeMel=64):
        super().__init__()

        # La funzione spike (Heaviside) non è derivabile: in backward si usa
        # la derivata di una fast sigmoid come gradiente surrogato.
        sg = surrogate.fast_sigmoid(slope=num_steps)

        # Strato 1: proiezione delle bande Mel -> corrente in ingresso ai LIF nascosti
        self.fc1 = nn.Linear(n_bandeMel, hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=sg)

        # Strato 2 (output): neuroni nascosti -> un neurone LIF per classe
        self.fc2 = nn.Linear(hidden, n_classes)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=sg)  # , learn_beta=True)
        # learn_beta=True renderebbe beta un parametro appreso (opzionale)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor, shape [B, 1, 64, T]
            Batch di spettrogrammi Mel (B campioni, 1 canale,
            64 bande Mel, T frame temporali; qui T = 128).

        Returns
        -------
        torch.Tensor, shape [B, n_classes]
            Firing rate medio di ogni neurone di output sui T time step,
            utilizzabile come punteggio per classe.
        """
        # [B, 1, 64, T] -> [B, 64, T] -> [T, B, 64]
        # Il tempo diventa la dimensione di iterazione esterna.
        x = x.squeeze(1).permute(2, 0, 1)

        # Inizializza i potenziali di membrana a zero (uno per strato)
        mem1 = self.lif1.init_leaky()
        mem_o = self.lif_out.init_leaky()

        out = []  # spike di output ad ogni time step
        for xt in x:  # xt: [B, 64], frame al tempo t
            # Strato nascosto: corrente = fc1(xt), emette spike h e aggiorna mem1
            h, mem1 = self.lif1(self.fc1(xt), mem1)
            # Strato di output: riceve gli spike nascosti, emette spk e aggiorna mem_o
            spk, mem_o = self.lif_out(self.fc2(h), mem_o)
            out.append(spk)

        # [T, B, n_classes] -> media sul tempo -> [B, n_classes] (rate coding)
        return torch.stack(out).mean(0)


# ============================================================
# MODELLO, LOSS E OTTIMIZZATORE
# ============================================================
model = MinimalSNN()

# Hinge loss multiclasse: penalizza se il rate della classe corretta non supera
# quello delle altre di almeno `margin`. Adatta a output limitati in [0, 1]
# (i firing rate), dove una cross-entropy sarebbe poco naturale.
criterion = nn.MultiMarginLoss(margin=0.3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ============================================================
# TRAINING
# ============================================================
def train_epoch(loader):
    """
    Esegue una epoch di training sull'intero DataLoader.

    Parameters
    ----------
    loader : torch.utils.data.DataLoader
        Restituisce batch (x, y) con x: [B, 1, 64, T] e y: [B] (etichette intere).

    Returns
    -------
    (float, float)
        Loss media per campione e accuracy sull'epoca.
    """
    model.train()
    tot, correct, loss_sum = 0, 0, 0.0   # contatori cumulativi sull'epoca

    for x, y in loader:
        rates = model(x)                  # firing rate [B, n_classes]
        loss = criterion(rates, y)

        optimizer.zero_grad()             # azzera i gradienti del batch precedente
        loss.backward()                   # BPTT attraverso i time step (con gradiente surrogato)
        optimizer.step()

        # Pesa la loss per la dimensione del batch (l'ultimo può essere più piccolo)
        loss_sum += loss.item() * len(y)
        correct += (rates.argmax(1) == y).sum().item()  # classe = neurone con rate maggiore
        tot += len(y)

    return loss_sum / tot, correct / tot
