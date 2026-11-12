"""
Valutazione della rete spiking "balena / non balena".

Lo script prende la rete già addestrata, la fa girare sul test set e misura
quanto bene distingue i clip con il verso di balena da quelli senza.

Formato dei file (da concordare )
----------------------------------------------
Dati   : file .pt con un dizionario {"X": tensore, "y": tensore}
         X -> [N, T, feature]  N clip, T passi temporali, valori di spike 0/1
         y -> [N]               0 = non balena, 1 = balena
Modello: file .py con una funzione build_model() che restituisce la rete.
         Ad ogni passo la rete riceve [batch, feature] e restituisce gli
         spike di uscita [batch, 2], oppure la coppia (spike, membrana).
Pesi   : file .pt con lo state_dict salvato durante l'addestramento.

Uso
---
    python evaluate.py --model model.py --weights best_model.pt \
                       --test test.pt --val val.pt --out risultati

    # media e deviazione standard su più addestramenti (seed diversi)
    python evaluate.py --aggregate risultati_seed*/metrics.json
"""

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import metrics as skm

NON_BALENA, BALENA = 0, 1
CLASS_NAMES = ["Non balena", "Balena"]


# ---------------------------------------------------------------------------
# 1. Caricamento
# ---------------------------------------------------------------------------

def load_model(model_file: str, weights_file: str, device: str) -> torch.nn.Module:
    """Costruisce la rete con build_model() e carica i pesi addestrati."""
    spec = importlib.util.spec_from_file_location("model_definition", model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    net = module.build_model()
    state = torch.load(weights_file, map_location=device)
    net.load_state_dict(state.get("model_state_dict", state))
    return net.to(device).eval()


def load_dataset(path: str) -> tuple[torch.Tensor, np.ndarray]:
    """Restituisce X come [N, T, feature] e le etichette y come array numpy."""
    data = torch.load(path, map_location="cpu")
    X, y = data["X"].float(), data["y"].long().numpy()
    assert len(X) == len(y), f"{path}: {len(X)} clip ma {len(y)} etichette"
    return X, y


# ---------------------------------------------------------------------------
# 2. Inferenza: dagli spike a un punteggio per clip
# ---------------------------------------------------------------------------

def reset_neurons(net: torch.nn.Module) -> None:
    """Azzera i potenziali di membrana prima di ogni nuovo batch."""
    from snntorch import utils
    utils.reset(net)


@torch.no_grad()
def count_output_spikes(net, X, batch_size, device) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Fa girare la rete su tutti i clip e restituisce:
      spike_counts [N, 2] : quante volte ha sparato ciascun neurone di uscita
      membrane     [N, 2] : potenziale di membrana medio (None se non c'è)
    """
    all_counts, all_membranes = [], []

    for start in range(0, len(X), batch_size):
        batch = X[start:start + batch_size].to(device)   # [B, T, feature]
        reset_neurons(net)

        spikes, membranes = [], []
        for t in range(batch.shape[1]):                   # un passo alla volta
            output = net(batch[:, t])
            if isinstance(output, tuple):
                spikes.append(output[0])
                membranes.append(output[1])
            else:
                spikes.append(output)

        all_counts.append(torch.stack(spikes).sum(dim=0).cpu())
        if membranes:
            all_membranes.append(torch.stack(membranes).mean(dim=0).cpu())

    counts = torch.cat(all_counts).numpy()
    membrane = torch.cat(all_membranes).numpy() if all_membranes else None
    return counts, membrane


def whale_score(counts: np.ndarray, membrane: np.ndarray | None) -> np.ndarray:
    """
    Punteggio "balena" tra 0 e 1 = quota di spike del neurone balena.

    Esempio: 8 spike "balena" e 2 "non balena" -> 8 / 10 = 0.8.
    Se nessuno dei due neuroni spara si usa il potenziale di membrana
    come spareggio; senza membrana il punteggio resta 0.5 (incertezza).
    """
    total = counts.sum(axis=1)
    score = np.full(len(counts), 0.5)

    fired = total > 0
    score[fired] = counts[fired, BALENA] / total[fired]

    silent = ~fired
    if membrane is not None and silent.any():
        difference = membrane[silent, BALENA] - membrane[silent, NON_BALENA]
        score[silent] = 1 / (1 + np.exp(-difference))
    return score


# ---------------------------------------------------------------------------
# 3. Metriche
# ---------------------------------------------------------------------------

def choose_threshold(y_val: np.ndarray, score_val: np.ndarray) -> float:
    """Sceglie sul validation set la soglia che massimizza F1."""
    precision, recall, thresholds = skm.precision_recall_curve(y_val, score_val)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[np.argmax(f1[:-1])])


def auc_confidence_interval(y, score, n_resamples=1000, seed=0) -> tuple[float, float]:
    """Intervallo di confidenza al 95% della ROC-AUC con il bootstrap."""
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_resamples):
        sample = rng.integers(0, len(y), len(y))
        if len(np.unique(y[sample])) == 2:
            aucs.append(skm.roc_auc_score(y[sample], score[sample]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def compute_metrics(y, score, threshold, counts) -> dict:
    """Calcola tutte le metriche descritte nel documento di accompagnamento."""
    predicted = (score >= threshold).astype(int)
    tn, fp, fn, tp = skm.confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    auc_low, auc_high = auc_confidence_interval(y, score)

    return {
        "clip": int(len(y)),
        "clip_balena": int(y.sum()),
        "soglia": round(float(threshold), 4),
        "roc_auc": float(skm.roc_auc_score(y, score)),
        "roc_auc_ic95": [auc_low, auc_high],
        "pr_auc": float(skm.average_precision_score(y, score)),
        "accuracy": float(skm.accuracy_score(y, predicted)),
        "balanced_accuracy": float(skm.balanced_accuracy_score(y, predicted)),
        "precision": float(skm.precision_score(y, predicted, zero_division=0)),
        "recall": float(skm.recall_score(y, predicted, zero_division=0)),
        "specificita": float(tn / max(tn + fp, 1)),
        "f1": float(skm.f1_score(y, predicted, zero_division=0)),
        "matrice_confusione": {"VP": int(tp), "FP": int(fp), "FN": int(fn), "VN": int(tn)},
        "spike_medi_per_clip": float(counts.sum(axis=1).mean()),
        "clip_senza_spike": int((counts.sum(axis=1) == 0).sum()),
        "accuracy_rispondendo_sempre_no": float((y == NON_BALENA).mean()),
    }


def verdict(m: dict, baseline_auc: float | None) -> list[str]:
    """Traduce i numeri in giudizi leggibili."""
    notes = []
    low, high = m["roc_auc_ic95"]

    if low <= 0.5:
        notes.append("NON BUONO: l'AUC non è distinguibile dal caso (0.5).")
    else:
        notes.append(f"Meglio del caso: AUC {m['roc_auc']:.3f} (IC95% {low:.3f}-{high:.3f}).")

    if baseline_auc is not None:
        gap = m["roc_auc"] - baseline_auc
        if gap >= 0:
            notes.append(f"Supera il baseline ({baseline_auc:.3f}) di {gap:+.3f}.")
        elif gap > -0.03:
            notes.append(f"Vicina al baseline ({baseline_auc:.3f}, {gap:+.3f}): buon risultato.")
        else:
            notes.append(f"Sotto il baseline ({baseline_auc:.3f}) di {-gap:.3f}.")

    if m["accuracy"] <= m["accuracy_rispondendo_sempre_no"]:
        notes.append("Attenzione: l'accuracy non batte chi risponde sempre 'non balena'.")
    if m["clip_senza_spike"] > 0.2 * m["clip"]:
        notes.append("Attenzione: molti clip senza spike in uscita (neuroni 'morti'?).")
    return notes


# ---------------------------------------------------------------------------
# 4. Grafici e file di output
# ---------------------------------------------------------------------------

def plot_roc(y, score, auc, out_dir):
    fpr, tpr, _ = skm.roc_curve(y, score)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"Rete spiking (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Caso (AUC = 0.5)")
    ax.set(xlabel="Tasso di falsi positivi", ylabel="Recall (tasso di veri positivi)",
           title="Curva ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "roc.png", dpi=150)
    plt.close(fig)


def plot_precision_recall(y, score, ap, out_dir):
    precision, recall, _ = skm.precision_recall_curve(y, score)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision, lw=2, label=f"Rete spiking (AP = {ap:.3f})")
    ax.axhline(y.mean(), ls="--", color="grey", label=f"Caso ({y.mean():.2f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Curva Precision-Recall")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "precision_recall.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm, threshold, out_dir):
    matrix = np.array([[cm["VN"], cm["FP"]], [cm["FN"], cm["VP"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
            ax.text(j, i, matrix[i, j], ha="center", va="center", color=color, fontsize=14)
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set(xlabel="Predetto", ylabel="Reale", title=f"Matrice di confusione (soglia {threshold:.2f})")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_score_distribution(y, score, threshold, out_dir):
    bins = np.linspace(0, 1, 26)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(score[y == NON_BALENA], bins=bins, alpha=0.6, label="Non balena")
    ax.hist(score[y == BALENA], bins=bins, alpha=0.6, label="Balena")
    ax.axvline(threshold, color="black", ls="--", label="Soglia")
    ax.set(xlabel="Punteggio balena", ylabel="Numero di clip", title="Separazione tra le classi")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "score_distribution.png", dpi=150)
    plt.close(fig)


def save_predictions(y, score, threshold, counts, out_dir):
    """Una riga per clip: utile per andare a vedere quali clip sbaglia."""
    with open(out_dir / "predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip", "etichetta", "punteggio", "predetto",
                         "spike_non_balena", "spike_balena"])
        for i in range(len(y)):
            writer.writerow([i, y[i], f"{score[i]:.4f}", int(score[i] >= threshold),
                             int(counts[i, 0]), int(counts[i, 1])])


def print_report(m: dict, notes: list[str]):
    cm = m["matrice_confusione"]
    low, high = m["roc_auc_ic95"]
    print(f"\nTest set: {m['clip']} clip ({m['clip_balena']} con balena)")
    print(f"  ROC-AUC            {m['roc_auc']:.4f}   IC95% {low:.4f}-{high:.4f}")
    print(f"  PR-AUC             {m['pr_auc']:.4f}")
    print(f"  Soglia             {m['soglia']:.3f}")
    for name in ["accuracy", "balanced_accuracy", "precision", "recall", "specificita", "f1"]:
        print(f"  {name:<18} {m[name]:.4f}")
    print(f"  Confusione         VP={cm['VP']} FP={cm['FP']} FN={cm['FN']} VN={cm['VN']}")
    print(f"  Spike medi/clip    {m['spike_medi_per_clip']:.2f}")
    print("\nVerdetto:")
    for note in notes:
        print(f"  - {note}")


# ---------------------------------------------------------------------------
# 5. Aggregazione di più addestramenti
# ---------------------------------------------------------------------------

def aggregate_runs(metric_files: list[str]):
    runs = [json.loads(Path(f).read_text()) for f in metric_files]
    print(f"\nMedia ± deviazione standard su {len(runs)} run")
    for name in ["roc_auc", "pr_auc", "balanced_accuracy", "precision", "recall", "f1"]:
        values = np.array([run[name] for run in runs])
        std = values.std(ddof=1) if len(values) > 1 else 0.0
        print(f"  {name:<18} {values.mean():.4f} ± {std:.4f}")


# ---------------------------------------------------------------------------
# Programma principale
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Valuta la rete spiking balena / non balena.")
    p.add_argument("--model", help="file .py con build_model()")
    p.add_argument("--weights", help="pesi addestrati (.pt)")
    p.add_argument("--test", help="test set (.pt)")
    p.add_argument("--val", help="validation set (.pt) per scegliere la soglia")
    p.add_argument("--threshold", type=float, help="soglia fissa al posto di quella scelta")
    p.add_argument("--baseline-auc", type=float, help="AUC di un modello di confronto")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--out", default="risultati", help="cartella dei risultati")
    p.add_argument("--aggregate", nargs="+", help="file metrics.json da mediare")
    return p.parse_args()


def main():
    args = parse_args()

    if args.aggregate:
        aggregate_runs(args.aggregate)
        return

    if not (args.model and args.weights and args.test):
        raise SystemExit("Servono --model, --weights e --test.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    net = load_model(args.model, args.weights, device)

    # Soglia di decisione: fissata a mano, scelta sul validation set, o 0.5.
    if args.threshold is not None:
        threshold = args.threshold
    elif args.val:
        X_val, y_val = load_dataset(args.val)
        threshold = choose_threshold(y_val, whale_score(*count_output_spikes(
            net, X_val, args.batch_size, device)))
    else:
        threshold = 0.5

    # Valutazione sul test set.
    X_test, y_test = load_dataset(args.test)
    counts, membrane = count_output_spikes(net, X_test, args.batch_size, device)
    score = whale_score(counts, membrane)

    m = compute_metrics(y_test, score, threshold, counts)
    m["verdetto"] = verdict(m, args.baseline_auc)

    # Salvataggio.
    (out_dir / "metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False))
    save_predictions(y_test, score, threshold, counts, out_dir)
    plot_roc(y_test, score, m["roc_auc"], out_dir)
    plot_precision_recall(y_test, score, m["pr_auc"], out_dir)
    plot_confusion_matrix(m["matrice_confusione"], threshold, out_dir)
    plot_score_distribution(y_test, score, threshold, out_dir)

    print_report(m, m["verdetto"])
    print(f"\nRisultati salvati in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
