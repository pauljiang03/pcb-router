"""Train the net-ordering model by distilling multi-start routing into a tiny
per-net scorer (pairwise ranking loss); writes models/ordering.npz.
Run: python scripts/train_ordering.py
"""
import sys
import time
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from envs.board import (load_te_example, load_te_excel, make_challenge,
                        ChallengeSpec, equal_length_placement, spread_placement,
                        rotate_board, rotate_points)
from envs.routing import route_all_traces
from envs.ordering import net_features, predict_order

FAST = dict(n_starts=1, max_iters=12, repair_passes=1)
OUT = pathlib.Path("models/ordering.npz")


def boards(train=True):
    """(board, placed) instances; held-out set uses unseen seeds/rotations."""
    out = []
    seeds = (0, 1, 2, 3) if train else (6, 7)
    for s in seeds:
        b = load_te_example(20, seed=s)
        out.append((b, equal_length_placement(b, 20)))
        out.append((b, spread_placement(b, 20)))
    cano = load_te_excel()
    k = 0 if train else 2
    rb = rotate_board(cano, k)
    out.append((rb, rotate_points(equal_length_placement(cano, 20), cano, k)))
    for gaps in ((3,) if train else (4,)):
        b, pl = make_challenge(ChallengeSpec(num_traces=16, n_gaps=gaps,
                                             board_size=120.0))
        out.append((b, pl))
    return out


def best_order(board, placed, n_orders=10, rng=None):
    """Route under K orders (informed + random); return the best order and its (fails, length) score."""
    n = len(placed)
    d = [np.hypot(placed[i][0] - board.traces[i].start_x,
                  placed[i][1] - board.traces[i].start_y) for i in range(n)]
    base = list(range(n))
    orders = [base, sorted(base, key=lambda i: -d[i]),
              sorted(base, key=lambda i: d[i])]
    while len(orders) < n_orders:
        orders.append(list(rng.permutation(n)))
    scored = []
    for od in orders:
        _p, L, f = route_all_traces(board, placed, order_hint=od,
                                    n_starts=1, max_iters=12, repair_passes=0)
        scored.append(((f, sum(x for x in L if x < 1e9)), od))
    scored.sort(key=lambda t: t[0])
    return scored[0][1], scored[0][0]


def main():
    rng = np.random.RandomState(0)
    X, R, sizes = [], [], []
    t0 = time.time()
    for board, placed in boards(train=True):
        od, (f, tot) = best_order(board, placed, rng=rng)
        rank = np.empty(len(od))
        for pos, i in enumerate(od):
            rank[i] = pos
        X.append(net_features(board, placed))
        R.append(rank)
        sizes.append(len(od))
        print(f"  dataset board n={len(od)} best fails={f} len={tot:.0f}mm",
              flush=True)
    Xc = np.concatenate(X)
    mu, sd = Xc.mean(0), Xc.std(0) + 1e-9
    print(f"dataset: {len(X)} boards, {len(Xc)} nets ({time.time()-t0:.0f}s)")

    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Linear(Xc.shape[1], 32), torch.nn.ReLU(),
        torch.nn.Linear(32, 32), torch.nn.ReLU(),
        torch.nn.Linear(32, 1))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    Xt = [torch.tensor((x - mu) / sd, dtype=torch.float32) for x in X]
    Rt = [torch.tensor(r, dtype=torch.float32) for r in R]
    for epoch in range(300):
        loss = torch.zeros(())
        for xt, rt in zip(Xt, Rt):
            s = net(xt).ravel()
            i, j = torch.meshgrid(torch.arange(len(rt)),
                                  torch.arange(len(rt)), indexing="ij")
            earlier = (rt[i] < rt[j])                 # i routed before j
            # earlier-routed nets must score HIGHER (pairwise hinge)
            loss = loss + torch.relu(1.0 - (s[i] - s[j]))[earlier].mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 100 == 0:
            print(f"  epoch {epoch}: loss {float(loss):.3f}", flush=True)

    W = [p.detach().numpy() for p in net.parameters()]
    OUT.parent.mkdir(exist_ok=True)
    np.savez(OUT, W1=W[0].T, b1=W[1], W2=W[2].T, b2=W[3], W3=W[4].T, b3=W[5],
             mu=mu, sd=sd)
    print(f"saved {OUT}")

    print("\nheld-out: learned-first n_starts=3  vs  default n_starts=6")
    for board, placed in boards(train=False):
        n = len(placed)
        od = predict_order(board, placed)
        t0 = time.time()
        _p, _L, f1 = route_all_traces(board, placed, order_hint=od, n_starts=3)
        tl = time.time() - t0
        t0 = time.time()
        _p, _L, f0 = route_all_traces(board, placed, n_starts=6)
        td = time.time() - t0
        print(f"  n={n}: learned {n-f1}/{n} in {tl:.1f}s | "
              f"default {n-f0}/{n} in {td:.1f}s", flush=True)


if __name__ == "__main__":
    main()
