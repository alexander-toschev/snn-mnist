# evaluation.py
from __future__ import annotations
import torch
from bindsnet.network.monitors import Monitor
from bindsnet.datasets import MNIST
from torchvision import transforms

from snn_mnist_net import SNNMeter

# ===== extra evaluation utilities =====
from typing import Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from counts_readout import collect_counts_plus_fast, make_mnist_datasets
from readout_models import tfidf_from_counts, train_mlp_readout

__all__ = ["probe_readouts_counts"]

@torch.no_grad()
def evaluate_on_mnist(net, input_layer, lif_layer, encoder, label_map, T: int = 200, top_k: int = 3, n_test: int = 1000, seed: int = 999):
    for c in net.connections.values():
        if hasattr(c, "update_rule"):
            c.update_rule.nu = (torch.as_tensor(0.0), torch.as_tensor(0.0))

    lif_mon = Monitor(lif_layer, state_vars=("s",), time=T); net.add_monitor(lif_mon, name="lif_test_tmp")
    transform = transforms.Compose([transforms.ToTensor()])
    ds_test = MNIST(root="./data", train=False, download=True, transform=transform)
    idxs = list(range(min(n_test, len(ds_test))))

    correct = 0
    meter = SNNMeter()
    lm_t = torch.as_tensor(label_map, dtype=torch.long, device=lif_layer.s.device if hasattr(lif_layer,'s') else None)

    for i in idxs:
        torch.manual_seed(seed + i)
        x = ds_test[i]["image"]; y = int(ds_test[i]["label"])
        spikes_in = encoder(x)
        net.run(inputs={"Input": spikes_in}, time=T)

        s_full = lif_mon.get("s")              # [T,1,N]
        s2 = s_full[:,0,:].to(torch.float32)   # [T,N]
        valid_mask = (lm_t >= 0)

        if not valid_mask.any():
            pred = -1
        else:
            s2m = s2.clone(); s2m[:, ~valid_mask] = -1e9
            active_t = (s2[:, valid_mask].sum(dim=1) > 0)
            if active_t.any():
                winners_t = s2m[active_t].argmax(dim=1)
                class_seq = lm_t[winners_t].clamp_min(0)
                votes = torch.bincount(class_seq, minlength=10)
                pred = int(votes.argmax().item())
            else:
                pred = -1

        if pred == y: correct += 1

        k_meter = min(3, lif_layer.n)
        counts = s2.sum(0)
        winners_for_meter = torch.topk(counts, k=k_meter).indices.tolist() if counts.sum() > 0 else None
        meter.log_sample(s_full, spikes_in, lif_layer.n, T, winners=winners_for_meter)

        net.reset_state_variables(); lif_mon.reset_state_variables()

    acc = correct / len(idxs); rpt = meter.report()
    net.monitors.pop("lif_test_tmp", None)
    print(f"TEST accuracy: {acc:.3f}  | spikes/sample={rpt['spikes_per_sample']:.2f}  energy≈{rpt['energy_proxy_per_sample']:.1f}")
    return {"accuracy": acc, **rpt}


def _train_linear_readout(
    Xtr: torch.Tensor, ytr: torch.Tensor,
    Xte: torch.Tensor, yte: torch.Tensor,
    epochs: int = 25, batch_size: int = 512,
    lr: float = 1e-3, weight_decay: float = 1e-4
) -> Tuple[nn.Module, float]:
    """Простая линейная голова (softmax). Возвращает (model, acc_test)."""
    in_dim = Xtr.shape[1]
    head = nn.Linear(in_dim, 10)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    head.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad(); loss = crit(head(xb), yb); loss.backward(); opt.step()
    head.eval()
    with torch.no_grad():
        acc = (head(Xte).argmax(1) == yte).float().mean().item()
    return head, acc

def _pca_whiten_counts(
    Xtr_counts: torch.Tensor, Xte_counts: torch.Tensor, var_keep: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    PCA по ковариации и whitening для counts. Возвращает (Ztr_w, Zte_w, eigvals, W, d).
    """
    Xtr_c = torch.log1p(Xtr_counts)
    Xte_c = torch.log1p(Xte_counts)
    mu = Xtr_c.mean(0, keepdim=True)
    sd = Xtr_c.std(0, keepdim=True).clamp_min(1e-6)
    Xtr_z = (Xtr_c - mu) / sd
    Xte_z = (Xte_c - mu) / sd

    C = (Xtr_z.T @ Xtr_z) / (Xtr_z.shape[0] - 1)
    eigvals, eigvecs = torch.linalg.eigh(C)   # по возрастанию
    idx = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[idx]; eigvecs = eigvecs[:, idx]
    cum = torch.cumsum(eigvals, dim=0) / eigvals.sum()
    d = int((cum < var_keep).sum().item() + 1)
    W = eigvecs[:, :d]
    Ztr = Xtr_z @ W
    Zte = Xte_z @ W
    Ztr_w = Ztr / torch.sqrt(eigvals[:d] + 1e-8)
    Zte_w = Zte / torch.sqrt(eigvals[:d] + 1e-8)
    return Ztr_w, Zte_w, eigvals, W, d

def probe_readouts_counts(
    Xtr: torch.Tensor, ytr: torch.Tensor,
    Xte: torch.Tensor, yte: torch.Tensor,
    n_hidden: int,
    mlp_hidden: int = 256, mlp_epochs: int = 30
) -> Dict[str, float]:
    """
    Пробует 3 пайплайна над spike-counts:
      1) counts -> log1p -> zscore -> Linear
      2) counts -> PCA(95%)+whiten -> Linear
      3) counts -> TF-IDF -> zscore -> MLP
    Возвращает dict с точностями.
    """
    N = n_hidden

    # 1) counts zscore + Linear
    Xtr_c = torch.log1p(Xtr[:, :N]); Xte_c = torch.log1p(Xte[:, :N])
    mu = Xtr_c.mean(0, keepdim=True); sd = Xtr_c.std(0, keepdim=True).clamp_min(1e-6)
    Xtr_z = (Xtr_c - mu) / sd; Xte_z = (Xte_c - mu) / sd
    _, acc_counts_lin = _train_linear_readout(Xtr_z, ytr, Xte_z, yte)

    # 2) PCA-whiten + Linear
    Ztr_w, Zte_w, *_ = _pca_whiten_counts(Xtr[:, :N], Xte[:, :N], var_keep=0.95)
    _, acc_pca_lin = _train_linear_readout(Ztr_w, ytr, Zte_w, yte)

    # 3) TF-IDF + MLP
    Xtr_n, Xte_n, _, _, _ = tfidf_from_counts(Xtr[:, :N], Xte[:, :N])
    _, acc_tfidf_mlp = train_mlp_readout(
        Xtr_n, ytr, Xte_n, yte, hidden=mlp_hidden, dropout=0.2,
        epochs=mlp_epochs, batch_size=256
    )
    return {
        "counts_zscore+Linear": acc_counts_lin,
        "PCAwhiten+Linear": acc_pca_lin,
        "TFIDF+MLP": acc_tfidf_mlp,
    }

def eval_readouts_from_net(
    net, lif_layer, encoder, cfg, label_map=None,
    n_train_counts: int = 60000, n_test_counts: int = 10000
) -> Dict[str, float]:
    """
    Собирает counts(+WTA-hist) через counts_readout.collect_counts_plus,
    затем гоняет probe_readouts_counts над counts и возвращает метрики.
    """
    ds_train, ds_test = make_mnist_datasets()
    Xtr, ytr = collect_counts_plus_fast(net, lif_layer, encoder, ds_train, n_train_counts, T=cfg.time, label_map=label_map,  move_net=True, batch_size=128,
        encoder_rate_boost=cfg.encoder_rate_boost)
    Xte, yte = collect_counts_plus_fast(net, lif_layer, encoder, ds_test,  n_test_counts,  T=cfg.time, label_map=label_map,  move_net=True, batch_size=128,
        encoder_rate_boost=cfg.encoder_rate_boost)
    N = lif_layer.n
    accs = probe_readouts_counts(Xtr, ytr, Xte, yte, n_hidden=N)
    return accs

