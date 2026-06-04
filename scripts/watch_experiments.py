#!/usr/bin/env python3
"""
Watch all 4 experiments at a glance — one combined, auto-refreshing table.

Reads each experiments/<name>/run.out, pulls the latest Update + Eval lines,
and prints a single dashboard in the terminal. Stdlib only.

    python scripts/watch_experiments.py            # refresh every 5s
    python scripts/watch_experiments.py --once     # print once and exit
    python scripts/watch_experiments.py -n 2       # refresh every 2s
"""
import argparse
import os
import re
import sys
import time

EXPS = ["baseline", "monotone", "heads", "layers"]  # default; overridden by auto-discovery
TOTAL = 10_000_000  # overridden by --total

UPD = re.compile(
    r"Update (\d+) \| Timesteps: ([\d,]+) \| Episodes: (\d+) \| "
    r"Mean Reward:\s*(-?[\d.]+) \| LR: ([\d.e+-]+) \| "
    r"Policy Loss: (-?[\d.]+) \| Value Loss: (-?[\d.]+) \| Belief Loss: (-?[\d.]+)"
)
EVAL = re.compile(r"P0 Win Rate: ([\d.]+)% \| P1 Win Rate: ([\d.]+)% \| Mean Length: ([\d.]+)")


def tail(path, n=400):
    try:
        with open(path, "r", errors="ignore") as f:
            return f.readlines()[-n:]
    except FileNotFoundError:
        return []


def last_match(lines, rx):
    for ln in reversed(lines):
        m = rx.search(ln)
        if m:
            return m
    return None


def bar(frac, width=18):
    frac = max(0.0, min(1.0, frac))
    filled = int(frac * width)
    return "█" * filled + "·" * (width - filled)


def discover(root):
    """All subdirs of root that have a run.out, sorted."""
    try:
        names = [d for d in sorted(os.listdir(root))
                 if os.path.exists(os.path.join(root, d, "run.out"))]
    except FileNotFoundError:
        names = []
    return names or EXPS


def snapshot(root):
    rows = []
    for e in discover(root):
        lines = tail(os.path.join(root, e, "run.out"))
        u = last_match(lines, UPD)
        ev = last_match(lines, EVAL)
        if not u:
            rows.append((e, None))
            continue
        ts = int(u.group(2).replace(",", ""))
        rows.append((e, {
            "update": int(u.group(1)),
            "ts": ts,
            "frac": ts / TOTAL,
            "reward": float(u.group(4)),
            "ploss": float(u.group(6)),
            "vloss": float(u.group(7)),
            "bloss": float(u.group(8)),
            "winrate": float(ev.group(1)) if ev else None,
            "mlen": float(ev.group(3)) if ev else None,
        }))
    return rows


def render(root):
    rows = snapshot(root)

    def row_text(name, d):
        if d is None:
            return (f"{name:9s} {'(starting…)':20s} {'-':>11s} {'-':>8s} "
                    f"{'-':>8s} {'-':>8s} {'-':>5s} {'-':>6s}")
        prog = f"{bar(d['frac'])} {d['frac']*100:4.1f}%"
        win = f"{d['winrate']:.0f}" if d["winrate"] is not None else "-"
        mlen = f"{d['mlen']:.1f}" if d["mlen"] is not None else "-"
        return (f"{name:9s} {prog:20s} {d['ts']:>11,d} {d['reward']:>8.2f} "
                f"{d['ploss']:>8.3f} {d['vloss']:>8.3f} {win:>5s} {mlen:>6s}")

    title = "Da Vinci Code — 4-way experiment comparison"
    hdr = (f"{'exp':9s} {'progress':20s} {'steps':>11s} {'reward':>8s} "
           f"{'p_loss':>8s} {'v_loss':>8s} {'win%':>5s} {'mlen':>6s}")
    body = [row_text(n, d) for n, d in rows]
    width = max([len(hdr), len(title)] + [len(b) for b in body])

    def line(s):
        return "│ " + s.ljust(width) + " │"

    out = ["┌" + "─" * (width + 2) + "┐",
           line(title),
           "├" + "─" * (width + 2) + "┤",
           line(hdr),
           "├" + "─" * (width + 2) + "┤"]
    out += [line(b) for b in body]
    out.append("└" + "─" * (width + 2) + "┘")
    out.append("  reward is NOT comparable across reward-modes (monotone has no ±10 win/lose);")
    out.append("  for 'who is best' use win% / mean-length, or run compare_experiments.py after 10M.")
    return "\n".join(out)


def main():
    global TOTAL
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="experiments")
    p.add_argument("-n", "--interval", type=float, default=5.0)
    p.add_argument("--once", action="store_true")
    p.add_argument("--total", type=int, default=TOTAL, help="total timesteps for progress %%")
    args = p.parse_args()
    TOTAL = args.total
    if args.once:
        print(render(args.root))
        return
    try:
        while True:
            sys.stdout.write("\033[2J\033[H")  # clear screen + home
            print(render(args.root))
            print(f"\n  refreshing every {args.interval:g}s … Ctrl-C to quit")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
