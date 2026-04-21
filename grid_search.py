# grid_search.py
import itertools, csv, time, math, random
from copy import deepcopy
from pathlib import Path

from snn_mnist_net import Cfg, run_experiment

def frange(start: float, stop: float, step: float):
    vals, v = [], start
    while v <= stop + 1e-12:
        vals.append(round(v, 6))
        v += step
    return vals

def make_w_init_pairs(lo_vals, hi_vals, min_span=0.15):
    pairs = []
    for lo in lo_vals:
        for hi in hi_vals:
            if hi > lo and (hi - lo) >= min_span:
                pairs.append((round(lo,6), round(hi,6)))
    return pairs

def proxy_score(hhi, winners_unique, spikes_per_sample, n_hidden):
    h = max(0.0, min(hhi or 0.0, 0.30)) / 0.30
    if winners_unique is None:
        w = 0.5
    else:
        du = abs(float(winners_unique) - 50.0)
        w = max(0.0, 1.0 - du/30.0)
    s = 1.0 - math.exp(-float(spikes_per_sample or 0.0)/40.0)
    return 0.6*h + 0.3*w + 0.1*s

def run_one_combo(base_cfg: Cfg, overrides: dict, N_train: int, num_threads=None):
    cfg = deepcopy(base_cfg)
    for k,v in overrides.items():
        if k == "w_init_pairs":
            cfg.w_init_lo, cfg.w_init_hi = map(float, v)
        else:
            setattr(cfg, k, v)
    cfg.N = int(N_train)
    cfg.enable_inhibition_at_start = True
    cfg.encoder = "poisson"

    try:
        if num_threads is not None:
            import torch
            torch.set_num_threads(int(num_threads))
    except Exception:
        pass

    t0 = time.time()
    try:
        out, connection, lif_layer, net, encoder = run_experiment(cfg, verbose=False, progress=False)
    except TypeError:
        out, connection, lif_layer, net, encoder = run_experiment(cfg, verbose=False)
    dt = round(time.time() - t0, 2)

    spikes  = float(out.get("spikes_per_sample", float("nan")))
    winners = int(out.get("winners_unique", -1))
    hhi     = float(out.get("winner_HHI", float("nan")))
    energy  = float(out.get("energy_proxy_per_sample", float("nan")))
    score   = proxy_score(hhi, winners, spikes, cfg.n_hidden)

    row = [
        float(getattr(cfg,"inhib_strength", float("nan"))),
        float(getattr(cfg,"w_init_lo", float("nan"))),
        float(getattr(cfg,"w_init_hi", float("nan"))),
        float(getattr(cfg,"poisson_rate_scale", float("nan"))),
        float(getattr(cfg,"thresh_init", float("nan"))),
        cfg.N,
        round(score, 3), round(spikes, 2), winners, round(hhi, 3), round(energy, 1),
        dt
    ]
    return row

from tqdm.auto import tqdm

def grid_search_network_v2(
    base_cfg: Cfg,
    grid: dict,
    csv_path: str = "grid_net_dense_results.csv",
    max_combos: int | None = None,
    sample_mode: str = "auto",
    resume: bool = True,
    num_threads: int | None = None
):
    keys = []
    spaces = []
    for k in ["inhib_strength","w_init_pairs","poisson_rate_scale","thresh_init"]:
        if k in grid:
            keys.append(k); spaces.append(grid[k])

    full_space = list(itertools.product(*spaces))
    total = len(full_space)

    if max_combos is None or total <= max_combos or sample_mode == "full":
        combos = full_space
    else:
        random.seed(getattr(base_cfg, "seed", 42))
        combos = random.sample(full_space, k=max_combos) if sample_mode in ("auto","random") else full_space[:max_combos]

    header = [
        "inhib_strength","w_init_lo","w_init_hi","poisson_rate_scale","thresh_init",
        "N_train","proxy_score","spikes_per_sample","winners_unique","winner_HHI","energy_proxy_per_sample",
        "runtime_sec"
    ]
    done_keys = set()
    p = Path(csv_path)
    if resume and p.exists():
        try:
            import pandas as pd
            old = pd.read_csv(p)
            if set(header).issubset(old.columns):
                for _,r in old.iterrows():
                    dk = (float(r["inhib_strength"]), float(r["w_init_lo"]), float(r["w_init_hi"]),
                          float(r["poisson_rate_scale"]), float(r["thresh_init"]), int(r["N_train"]))
                    done_keys.add(dk)
        except Exception:
            pass

    new_file = not p.exists() or (not resume)
    if new_file and p.exists():
        p.unlink(missing_ok=True)
    f = open(p, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new_file:
        w.writerow(header); f.flush()

    N_train = int(grid.get("N_train", getattr(base_cfg, "N", 400)))
    print(f"[grid] total={total}, run={len(combos)} (resume={resume}, already_done={len(done_keys)})")
    print(f"[grid] N_train={N_train}, csv='{csv_path}'")

    rows = []
    for combo in tqdm(combos, desc="Grid (network)", ncols=110):
        overrides = dict(zip(keys, combo))
        wlo, whi = overrides["w_init_pairs"]
        dk = (float(overrides.get("inhib_strength", float("nan"))),
              float(wlo), float(whi),
              float(overrides.get("poisson_rate_scale", float("nan"))),
              float(overrides.get("thresh_init", float("nan"))),
              N_train)
        if resume and dk in done_keys:
            continue

        row = run_one_combo(base_cfg, overrides, N_train=N_train, num_threads=num_threads)
        rows.append(row)
        w.writerow(row); f.flush()

    f.close()

    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        df = df.sort_values("proxy_score", ascending=False)
        print("\n=== TOP-10 (proxy_score) ===")
        for _, r in df.head(10).iterrows():
            print(
                f"inh={r.inhib_strength:.3f}  w=({r.w_init_lo:.2f},{r.w_init_hi:.2f})  "
                f"rate={r.poisson_rate_scale:.3f}  thr={r.thresh_init:.2f}  "
                f"N={int(r.N_train)}  proxy={r.proxy_score:.3f}  "
                f"sp={r.spikes_per_sample:.1f}  uniq={int(r.winners_unique)}  HHI={r.winner_HHI:.3f}  t={r.runtime_sec:.2f}s"
            )
    except Exception as e:
        print("[grid] summary print failed:", e)

    return rows
