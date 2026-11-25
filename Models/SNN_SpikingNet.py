"""
Spiking Neural Network (SNN) minimale per classificazione di spettrogrammi Mel.

Architettura: Linear -> LIF -> Linear -> LIF, applicata passo per passo
lungo l'asse temporale dello spettrogramma. L'output è il firing rate
medio (spike per time step) di ciascun neurone di uscita, usato come logit.
"""

# ============================================================
# PACCHETTI
# ============================================================
#!pip install optuna
#!pip install snntorch

import pickle, numpy as np, optuna, torch, snntorch as snn, torch.nn as nn, torch.nn.functional as F
from snntorch import surrogate, utils
from torch.utils.data import DataLoader, Dataset

# ============================================================
# DATI
# ============================================================
#train_ds = materialize(train_dataset)
#test_ds  = materialize(test_dataset)

train_dl = DataLoader(train_ds, 256, shuffle=True)
test_dl  = DataLoader(test_ds, 256)
# ============================================================
# DEFINIZIONE DELLA SPIKING NEURAL NETWORK
# ============================================================
# 3) MODELLO (SNN con dinamica temporale di membrana) ---------------------
spike_fn = surrogate.fast_sigmoid(slope=25)      # Heaviside in avanti, sigmoide veloce all'indietro

class LIF(nn.Module):
    def __init__(s, beta, thr=1.0):
        super().__init__()
        s.beta, s.thr = beta, thr
    def forward(s, cur):                         # cur: (T, B, n) = correnti I[t]
        mem = torch.zeros_like(cur[0])           # membrana nuova a ogni chiamata: nessuno stato condiviso
        out = []
        for c in cur:                            # unico ciclo sul tempo
            mem = s.beta * mem + c               # U[t] = beta*U[t-1] + I[t]
            spk = spike_fn(mem - s.thr)          # spike si/no (surrogato in backward)
            mem = mem - spk * s.thr              # reset per sottrazione
            out.append(spk)
        return torch.stack(out)                  

class SNN(nn.Module):
    def __init__(s, n_hidden, n_layers, beta, t_pool=1): # t_pool in caso volessimo fare pooling in futuro, t_pool = 1 non lo fa
        super().__init__()
        dims = [64] + [n_hidden] * n_layers      # 64 bande Mel in ingresso
        s.lin = nn.ModuleList(nn.Linear(dims[i], dims[i+1]) for i in range(n_layers))
        s.lif = nn.ModuleList(LIF(beta) for _ in range(n_layers))
        s.head = nn.Linear(n_hidden, 1)
        s.t_pool = t_pool

    def forward(s, x):                           # x: (B, 1, 64, 128)
        x = x.squeeze(1)                         # (B, 64, 128)
        if s.t_pool > 1:
            x = F.avg_pool1d(x, s.t_pool)
        h = x.permute(2, 0, 1)                   # (T, B, 64)
        for lin, lif in zip(s.lin, s.lif):
            h = lif(lin(h))                      # Linear su tutti i T, poi dinamica di membrana
        return s.head(h.mean(0)).squeeze(-1)     # combinazione lineare dello spike rate -> logit (B,)

def make_model(n_hidden, n_layers, beta, t_pool=1):
    return SNN(n_hidden, n_layers, beta, t_pool).to(dev)

loss_fn = nn.BCEWithLogitsLoss()   # sigmoide + BCE un neurone come output è sufficiente

def f1_score(p,t):
    tp = (p & t).sum()
    fp = (p & ~t).sum()
    fn = (~p & t).sum()
    f1 = 2 * tp / (2 * tp + fp + fn).clamp(min=1)
    return f1

@torch.no_grad()
def evaluate(model, dl):
    model.eval()                                   # modalità inferenza (buona pratica, qui neutra)
    # CICLO 4/5: mini-batch di dl. no_grad (decoratore): niente backward, quindi non serve costruire
    # il grafo computazionale. Risparmia memoria e tempo, senza cambiare i valori calcolati.
    z = torch.cat([model(xb.float()) for xb, _ in dl])      # logit di tutto il dl, restano su GPU
    y = torch.cat([yb for _, yb in dl]).float()
    loss = loss_fn(z, y)                           # media corretta sui campioni (non media di medie per batch)
    p = z > 0
    t =  y > 0.5                          # logit > 0 (cioè p > 0.5) -> predizione booleana
    f1 = f1_score(p,t)
    return loss.item(), f1.item()                  # un solo trasferimento per valutazione


def fit(model, lr, tr_dl, va_dl, epochs, patience=5, trial=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = float("inf")
    best_f1 = 0.0
    state = None
    wait = 0      # miglior val loss, F1 a quell'epoca, snapshot dei pesi, epoche senza miglioramento
    # CICLO 2: epoche. Una epoca = un passaggio completo sul train + una valutazione sul val.
    for ep in range(epochs):
        model.train()                              # modalità training (qui nessun effetto: niente BN/dropout)
        # CICLO 3: mini-batch del train (256 chunk alla volta, shuffle=True ad ogni epoca).
        # Qui i gradienti SERVONO, quindi niente no_grad.
        for xb, yb in tr_dl:
            opt.zero_grad(set_to_none=True)        # PyTorch accumula i gradienti: vanno azzerati a ogni step
                                                   # (set_to_none evita di scrivere zeri: più veloce)
            loss = loss_fn(model(xb.float()), yb)  
            loss.backward()                        # backprop (attraverso lo spike via gradiente surrogato)
            opt.step()                             # aggiornamento Adam
        # CICLO 4: val; evaluate è già in no_grad e restituisce val loss dell'epoca + F1
        v, f = evaluate(model, va_dl)

    #==== BO ======#
        if trial:                                  # solo durante BO
            trial.report(-v, ep)                   # comunica a Optuna il valore intermedio (segno - : si massimizza)
            if trial.should_prune():               # MedianPruner: sotto la mediana dei trial a quell'epoca?
                raise optuna.TrialPruned()         # interrompe il trial

    #=== Early-Stop ====#
        if v < best:                               # miglioramento: salvo il checkpoint e azzero il contatore
            # .clone() è necessario: state_dict() restituisce riferimenti ai tensori vivi,
            # che l'ottimizzatore continuerebbe a modificare. Senza copia, "il best" sarebbe l'ultimo.
            best = v
            best_f1 = f
            wait =  0
            state = {k: t.clone() for k, t in model.state_dict().items()}
        elif (wait := wait + 1) >= patience:       # walrus: incrementa e confronta in un colpo solo
            break                                  # early stopping
    model.load_state_dict(state)                   # ripristina i pesi dell'epoca migliore, non dell'ultima
    return best_f1                                 # F1 all'epoca di miglior val loss: niente passata extra
# ============================================================
# BAYESIAN OPTIMIZATION
# ============================================================
val_ds = materialize(val_dataset)
val_dl = DataLoader(val_ds, 256)

# --- split BO 80/20 sui tensori di val ---
Xv, yv = val_ds.tensors
perm = torch.randperm(len(yv), generator=torch.Generator().manual_seed(26))
n_tr = int(0.8 * len(yv))
tr_i, va_i = perm[:n_tr], perm[n_tr:]
bo_tr = DataLoader(TensorDataset(Xv[tr_i], yv[tr_i]), 256, shuffle=True)
bo_va = DataLoader(TensorDataset(Xv[va_i], yv[va_i]), 256)

optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(t):
    model = make_model(t.suggest_int("n_hidden", 4, 32, log=True),
                       t.suggest_int("n_layers", 1, 7),
                       beta=t.suggest_float("beta", 0.5, 0.99),
                       t_pool=2)
    return fit(model, t.suggest_float("lr", 1e-4, 1e-2, log=True),
               bo_tr, bo_va, epochs=300, patience=3, trial=t)

study = optuna.create_study(direction="maximize",
                            #sampler=optuna.samplers.TPESampler(seed=14, n_startup_trials=10, n_ei_candidates=24),
                            sampler=optuna.samplers.GPSampler(seed=14),
                            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=3))
study.optimize(objective, n_trials=20, timeout=25 * 60)
bp = study.best_params
print(bp, study.best_value)

# ============================================================
# TEST
# ============================================================
model = make_model(bp["n_hidden"], bp["n_layers"], bp["beta"], t_pool=2)
fit(model, bp["lr"], train_dl, val_dl, epochs=300, patience=8)


print("train F1:", round(evaluate(model, DataLoader(train_ds, 256, shuffle=False))[1],4),
      "| val F1:", round(evaluate(model, val_dl)[1], 4),
      "| test F1:", round(evaluate(model, test_dl)[1], 4))

torch.save(model.state_dict(), "modello_snn.pth")
pickle.dump({"campioni_index": ci, "best_params": bp}, open("dataset_marine_metadata.pkl", "wb"))

