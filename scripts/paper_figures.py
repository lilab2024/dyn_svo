"""Figures for the paper, generated from committed result files and recorded data.

  python scripts/paper_figures.py
Writes to paper/figures (one panel per figure): test_position_ratio.pdf, theory_gain_I.pdf, theory_gain_II.pdf,
test_results_primary.pdf, test_results_r5.pdf, chain_validation.pdf, data_vehicle_speed.pdf
and data_spectrum.pdf and figures_provenance.json. Reads the test summaries, theory_v1, theory_v2
the chain validation listed in config/paper_sources.yaml and refuses them unless their md5 match; reads the recorded
development data after verify_data full (development experiments only). Nothing is written if any source is refused.
Requires a clean repository. PDF metadata dates are removed so the same input gives the same file.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dyn_svo import provenance as pv
from dyn_svo.config import load_config
from dyn_svo.data import manifest as mf

CONFIGS = ("test_freeze", "paper_sources", "data_split", "data_manifest", "metrics")
EXPS = ("B", "D", "H")
MARKERS = {"B": "o", "D": "s", "H": "^"}
U16 = math.pi / 16
# Curves stay distinguishable in black-and-white print and for colour-blind readers:
# Okabe-Ito colours, and in addition a distinct marker and line style per curve.
OKABE_ITO = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#000000", "#F0E442")
LINESTYLES = ("-", "--", "-.", ":", (0, (5, 1, 1, 1, 1, 1)), (0, (8, 2)), (0, (2, 2)), (0, (4, 1, 1, 1)), "-")
MARKER_SEQ = ("o", "s", "^", "v", "D", "P", "X", "*", "h")


def curve_style(i, n_points=None, n_markers=8):
    """Colour, line style and marker of curve i; markers on about n_markers points of a dense curve."""
    st = {"color": OKABE_ITO[i % len(OKABE_ITO)], "linestyle": LINESTYLES[i % len(LINESTYLES)],
          "marker": MARKER_SEQ[i % len(MARKER_SEQ)], "markersize": 3.5, "markerfacecolor": "white", "markeredgewidth": 0.8}
    if n_points is not None and n_points > n_markers:
        st["markevery"] = (i % 3 * max(1, n_points // (3 * n_markers)), max(1, n_points // n_markers))
    return st


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def position_ratios(r5):
    """exp(L_j) of the dynamic configuration per test experiment."""
    pe = r5["h2"]["per_experiment"]
    return {e: [math.exp(x) for x in pe[e]["position_log_ratio"]] for e in EXPS}


def position_ratio_figure(r5, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = position_ratios(r5)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for i, e in enumerate(EXPS):
        st = curve_style(i)
        st["marker"] = MARKERS[e]
        ax.plot(range(1, len(data[e]) + 1), data[e], linewidth=1, label="Experiment " + e, **st)
    ax.axhline(1.0, color="0.4", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Follower position behind the AV (1 = directly behind)")
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("Follower oscillation ratio" + chr(10) + "to the all-human run")
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    h, l = ax.get_legend_handles_labels()
    h.append(_ref_handle("Ratio = 1 (no improvement)"))
    _legend_below(fig, h, l + [h[-1].get_label()], ncol=2, rows=2)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    return data


def gain_figure(v2, regime, path):
    """|G| of the AV loop over the period for the phi grid in one regime (theory_v2)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    gc = v2["sensitivity"][regime]["gain_curves"]
    periods, curves = gc["periods_s"], gc["curves"]
    ax.axvspan(20.0, 64.0, color="0.92", linewidth=0)
    ax.axhline(1.0, color="0.4", linewidth=0.8, linestyle="--")
    n = len(curves)
    for i, c in enumerate(curves):
        st = curve_style(i, n_points=len(periods), n_markers=7)
        st["color"] = plt.cm.cividis(0.8 * (1 - i / max(n - 1, 1)))   # light to dark with phi, also in grey
        ax.plot(periods, c["gain"], linewidth=1, label="%g" % round(c["phi_over_pi16"], 6), **st)
    _period_axis(ax, min(periods), max(periods))
    ax.set_ylabel("Speed gain |G| (regime " + regime + ")", fontsize=8)
    ax.set_xlabel("Period of the leader" + chr(8217) + "s speed oscillation (s)", fontsize=8)
    ax.tick_params(labelsize=7)
    data = {"%g" % round(c["phi_over_pi16"], 6): [round(min(c["gain"]), 6), round(max(c["gain"]), 6)] for c in curves}
    h, l = ax.get_legend_handles_labels()
    l = ["$" + chr(92) + "phi = 0$" if x == "0" else "$" + chr(92) + "phi = " + ("" if x == "1" else x) + chr(92) + "pi/16$"
         for x in l]
    h = h + [_band_handle((20.0, 64.0)), _ref_handle("|G| = 1 (no amplification)")]
    l = l + [h[-2].get_label(), h[-1].get_label()]
    _legend_below(fig, h, l, ncol=3, rows=4)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    return data


def _legend_below(fig, handles, labels, ncol, rows=1, **kw):
    """Legend centred below the axes, never over the data; returns the space it takes."""
    h_in = fig.get_size_inches()[1]
    frac = (0.17 * rows + 0.06) / h_in
    fig.tight_layout(rect=(0, frac, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False, fontsize=6, columnspacing=1.0,
               handletextpad=0.4, **kw)
    return frac


PERIOD_TICKS = (10, 20, 50, 100, 200, 500)


def _period_axis(ax, lo, hi):
    """Log period axis with plain-number ticks at 10, 20, 50, 100, 200 and 500 s within [lo, hi]."""
    from matplotlib.ticker import FixedLocator, NullFormatter, FixedFormatter
    ax.set_xscale("log")
    ticks = [x for x in PERIOD_TICKS if lo <= x <= hi]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([str(x) for x in ticks]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(lo, hi)


def _band_handle(band):
    """Legend entry for the shaded wave band (labels easy to read)."""
    from matplotlib.patches import Patch
    return Patch(facecolor="0.92", edgecolor="0.7", linewidth=0.5,
                 label="Wave band (%g" % band[0] + chr(8211) + "%g s)" % band[1])


def _ref_handle(label):
    """Legend entry for the dashed reference line at one."""
    from matplotlib.lines import Line2D
    return Line2D([], [], color="0.4", linewidth=0.8, linestyle="--", label=label)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, path, dpi=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if dpi is None:
        fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})
    else:
        fig.savefig(path, metadata={"CreationDate": None, "ModDate": None}, dpi=dpi)


def controller_label(name):
    """Readable label of a test controller from its configuration name."""
    from dyn_svo.experiments import configs as cf
    if name == "fs":
        return "FollowerStopper"
    if name == "pi":
        return "PI"
    if name == "fixed_match":
        return "MPC, matched fixed $" + chr(92) + "phi$"
    by = {c["name"]: c for c in cf.av_configs()}
    c = by[name]
    if name.startswith("fixed_"):
        k = round(c["phi"] / U16, 6)
        return "MPC, fixed $" + chr(92) + "phi = 0$" if k == 0 else "MPC, fixed $" + chr(92) + "phi = %g" % k + chr(92) + "pi/16$"
    return "MPC, dynamic $" + chr(92) + "phi_t$"


def test_results_figure(docs, variant, path):
    """FOR of every test controller per experiment for one analysis (primary or r5); filled marker when every
    guardrail passed."""
    plt = _plt()
    def rank(n):
        return ({"fixed_match": 2, "fs": 3, "pi": 4}.get(n, 1 if n.startswith("dyn_") else 0), n)
    names = sorted({k.split("|")[0] for k in docs[variant]["experiment_log_for"]}, key=rank)
    fig, ax = plt.subplots(figsize=(3.5, 1.5 + 0.42 * len(names)))
    data = {}
    for i, n in enumerate(names):
        y = len(names) - 1 - i
        for e in EXPS:
            key = n + "|" + e
            lf = docs[variant]["experiment_log_for"].get(key)
            if lf is None:
                continue
            ok = bool(docs[variant]["guardrails"][key]["passes"])
            fr = math.exp(lf)
            ax.plot([fr], [y + {"B": 0.2, "D": 0.0, "H": -0.2}[e]], linestyle="none", marker=MARKERS[e],
                    markersize=4, markeredgecolor="0.1", markerfacecolor="0.1" if ok else "white")
            data[key] = [round(fr, 6), ok]
    ax.axvline(1.0, color="0.4", linewidth=0.8, linestyle="--")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([controller_label(n) for n in reversed(names)], fontsize=7)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("Follower oscillation ratio, FOR" + chr(10) + "(1 = as behind a human driver)", fontsize=8)
    handles = [ax.plot([], [], linestyle="none", marker=MARKERS[e], markersize=4, markeredgecolor="0.1",
                       markerfacecolor="0.1", label="Experiment " + e)[0] for e in EXPS]
    handles.append(ax.plot([], [], linestyle="none", marker="s", markersize=4, markeredgecolor="0.1",
                           markerfacecolor="0.1", label="Filled: all guardrails passed")[0])
    handles.append(ax.plot([], [], linestyle="none", marker="s", markersize=4, markeredgecolor="0.1",
                           markerfacecolor="white", label="Hollow: a guardrail failed")[0])
    _legend_below(fig, handles, [x.get_label() for x in handles], ncol=2, rows=3)
    _save(fig, path)
    plt.close(fig)
    return data


def chain_validation_figure(cv, path):
    """Simulated over recorded oscillation by follower position, mean over episodes per experiment."""
    plt = _plt()
    ok = {r["exp"]: r for r in cv["summary"]}
    sums, counts = {}, {}
    for ep in cv["episodes"]:
        e = ep["exp"]
        s, c = sums.setdefault(e, []), counts.setdefault(e, [])
        for j, x in enumerate(ep["log_ratio_by_position"]):
            if x is None or not math.isfinite(x):
                continue
            while len(s) <= j:
                s.append(0.0)
                c.append(0)
            s[j] += x
            c[j] += 1
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    data = {}
    for i, e in enumerate(sorted(sums)):
        ys = [math.exp(s / c) for s, c in zip(sums[e], counts[e]) if c > 0]
        passed = all(ok[e][k] for k in ("ok_sigma", "ok_slope", "ok_speed"))
        ax.plot(range(1, len(ys) + 1), ys, linewidth=1,
                label="Experiment " + e + ("" if passed else " (not all criteria met)"), **curve_style(i))
        data[e] = [round(y, 6) for y in ys]
    ax.axhline(1.0, color="0.4", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Follower position behind the replayed leader", fontsize=8)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("Oscillation ratio," + chr(10) + "simulated / recorded", fontsize=8)
    ax.tick_params(labelsize=7)
    h, l = ax.get_legend_handles_labels()
    h.append(_ref_handle("Ratio = 1 (simulation = recording)"))
    _legend_below(fig, h, l + [h[-1].get_label()], ncol=2, rows=3)
    _save(fig, path)
    plt.close(fig)
    return data


def vehicle_speed_matrix(win, ring_length, t_span=300.0, step=3):
    """Speed of every vehicle over the first t_span seconds of the usable window, every step-th frame.
    Rows are ordered by position on the ring at the first sample, front first, so each row follows the row above
    (the first row follows the last). Returns (vehicle ids, times from the window start in s, matrix)."""
    import numpy as np
    t0 = float(win["t"].min())
    d = win[(win["k"] % step == 0) & (win["t"] <= t0 + t_span)]
    first = d[d["k"] == d["k"].min()]
    if set(first["veh_id"]) != set(d["veh_id"]):
        raise ValueError("not every vehicle is present at the first sample of the window")
    order = list(first.assign(pos=np.mod(first["x"].to_numpy(), ring_length)).sort_values("pos", ascending=False)["veh_id"])
    m = d.pivot(index="veh_id", columns="t", values="v").reindex(order)
    return order, m.columns.to_numpy(dtype=float) - t0, m.to_numpy(dtype=float)


def data_vehicle_speed_figure(win, ring_length, path, t_span=300.0):
    """Vehicle-time raster of speed for one development experiment at 10 Hz, no interpolation, colour range p2-p98
    of the speeds shown (replaces the time-space dots)."""
    import numpy as np
    from matplotlib.colors import Normalize
    from matplotlib.ticker import MaxNLocator
    plt = _plt()
    order, tt, m = vehicle_speed_matrix(win, ring_length, t_span)
    vals = m[np.isfinite(m)]
    lo, hi = (float(x) for x in np.percentile(vals, [2, 98]))
    dt = float(np.median(np.diff(tt)))
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    im = ax.imshow(m, aspect="auto", cmap="viridis", norm=Normalize(lo, hi), interpolation="nearest",
                   extent=(tt[0] - dt / 2, tt[-1] + dt / 2, len(order) + 0.5, 0.5))
    cb = fig.colorbar(im, ax=ax, pad=0.02, extend="both")
    cb.set_label("Speed (m/s)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Vehicle (each follows the one above)", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    _save(fig, path, dpi=300)   # the raster is sampled at 300 dpi
    plt.close(fig)
    return {"experiment": str(win["exp"].iloc[0]), "t_span_s": t_span, "rate_hz": round(1.0 / dt, 6),
            "vehicles": len(order), "order": [int(v) for v in order], "missing_cells": int(np.size(m) - np.size(vals)),
            "speed_clip_p2_p98_m_s": [round(lo, 4), round(hi, 4)]}


def data_spectrum_figure(spectra, band, path, period_range=(10.0, 120.0)):
    """Speed spectra of the development experiments over the period range of the spectrum analysis
    with the wave band. Returns the dominant period per experiment by the rule of
    scripts/wave_spectrum.py."""
    import numpy as np
    from dyn_svo.analysis import spectrum as spm
    plt = _plt()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.axvspan(band[0], band[1], color="0.92", linewidth=0)
    data = {}
    for i, e in enumerate(sorted(spectra)):
        f, psd = spectra[e]
        keep = (f > 0) & (f >= 1.0 / period_range[1]) & (f <= 1.0 / period_range[0])
        per, p = 1.0 / f[keep], psd[keep] / float(np.max(psd[keep]))
        ax.plot(per, p, linewidth=1, label=e, **curve_style(i, n_points=len(per), n_markers=10))
        data[e] = round(spm.dominant_period(f, psd, tuple(period_range))[0], 2)
    _period_axis(ax, period_range[0], period_range[1])
    ax.set_xlabel("Period of the speed oscillation (s)", fontsize=8)
    ax.set_ylabel("Normalized speed spectrum", fontsize=8)
    ax.tick_params(labelsize=7)
    h, l = ax.get_legend_handles_labels()
    h, l = h + [_band_handle(band)], ["Experiment " + x for x in l] + [_band_handle(band).get_label()]
    _legend_below(fig, h, l, ncol=3, rows=2)
    _save(fig, path)
    plt.close(fig)
    return data


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    out = Path(args.out_dir) if args.out_dir else mf.data_root() / "dyn_svo" / "paper" / "figures"
    res = mf.data_root() / "dyn_svo" / "results"
    data_check = mf.verify_data("full")   # reads recorded development data
    prov = pv.provenance(CONFIGS, data_check)
    pv.require_clean(prov)
    src = load_config("paper_sources")
    f = src["test_summaries"]["r5"]
    if md5(res / f["path"]) != f["md5"]:
        raise SystemExit("test summary refused (md5 differs from config/paper_sources.yaml): " + f["path"])
    r5 = json.loads((res / f["path"]).read_text(encoding="utf-8"))
    if r5.get("variant") != "r5" or r5.get("commits") != [src["test_commit"]] or r5.get("h2") is None:
        raise SystemExit("test summary refused: variant, commit or H2 missing")
    t2 = (src.get("theory") or {}).get("v2")
    if t2 is None:
        raise SystemExit("theory_v2 not listed in config/paper_sources.yaml (nothing written)")
    if md5(res / t2["path"]) != t2["md5"]:
        raise SystemExit("theory result refused (md5 differs from config/paper_sources.yaml): " + t2["path"])
    v2 = json.loads((res / t2["path"]).read_text(encoding="utf-8"))
    if v2.get("provenance", {}).get("dirty") is not False or set(v2.get("sensitivity", {})) != {"I", "II"}:
        raise SystemExit("theory_v2 refused: dirty provenance or regimes missing")
    docs, rels = {"r5": r5}, {"r5": f}
    fp = src["test_summaries"]["primary"]
    if md5(res / fp["path"]) != fp["md5"]:
        raise SystemExit("test summary refused (md5 differs from config/paper_sources.yaml): " + fp["path"])
    docs["primary"] = json.loads((res / fp["path"]).read_text(encoding="utf-8"))
    rels["primary"] = fp
    if docs["primary"].get("variant") != "primary" or docs["primary"].get("commits") != [src["test_commit"]]:
        raise SystemExit("test summary refused: variant or commit of the primary analysis")
    fcv = (src.get("validation") or {}).get("chainval")
    t1 = (src.get("theory") or {}).get("v1")
    for name, ff in (("chain validation", fcv), ("theory_v1", t1)):
        if ff is None:
            raise SystemExit(name + " not listed in config/paper_sources.yaml (nothing written)")
        if md5(res / ff["path"]) != ff["md5"]:
            raise SystemExit(name + " refused (md5 differs from config/paper_sources.yaml): " + ff["path"])
    cv = json.loads((res / fcv["path"]).read_text(encoding="utf-8"))
    band = json.loads((res / t1["path"]).read_text(encoding="utf-8"))["settings"]["band_s"]
    from dyn_svo.analysis import spectrum as spm
    from dyn_svo.data import loader as ld
    from dyn_svo.data import split as sp
    dev = sp.experiments("dev")
    frames = {e: ld.load_experiment(e) for e in dev}
    sc = load_config("metrics")["spectrum"]
    spectra = {e: spm.experiment_spectrum(frames[e], segment_s=sc["segment_s"], overlap=sc["overlap"],
                                          nfft_factor=sc["nfft_factor"])[:2] for e in dev}
    wave_exp = dev[0]
    fw = frames[wave_exp]
    win = fw[fw["in_window"]][["exp", "veh_id", "k", "t", "x", "v"]]
    pdf = out / "test_position_ratio.pdf"
    data = position_ratio_figure(r5, pdf)
    gdata = {r: gain_figure(v2, r, out / ("theory_gain_" + r + ".pdf")) for r in ("I", "II")}
    rdata = {v: test_results_figure(docs, v, out / ("test_results_" + v + ".pdf")) for v in ("primary", "r5")}
    vpdf, tpdf, spdf = out / "chain_validation.pdf", out / "data_vehicle_speed.pdf", out / "data_spectrum.pdf"
    vdata = chain_validation_figure(cv, vpdf)
    tdata = data_vehicle_speed_figure(win, ld.RING_LENGTH_M, tpdf)
    sdata = data_spectrum_figure(spectra, band, spdf, period_range=tuple(sc["period_range_s"]))
    record = {"provenance": prov, "figures": {"test_position_ratio.pdf": {
        "md5": md5(pdf), "source": "results/" + f["path"], "source_md5": f["md5"],
        "data": {e: [round(x, 6) for x in v] for e, v in data.items()}},
        "chain_validation.pdf": {"md5": md5(vpdf), "source": "results/" + fcv["path"], "source_md5": fcv["md5"],
                                 "data_ratio_by_position": vdata},
        "data_vehicle_speed.pdf": {"md5": md5(tpdf), "data": "recorded development experiment " + wave_exp,
                                   "shown": tdata},
        "data_spectrum.pdf": {"md5": md5(spdf), "data": "recorded development experiments " + ", ".join(dev),
                              "band_s": band, "band_source_md5": t1["md5"], "dominant_period_shown_s": sdata}}}
    for r in ("I", "II"):
        q = out / ("theory_gain_" + r + ".pdf")
        record["figures"][q.name] = {"md5": md5(q), "source": "results/" + t2["path"], "source_md5": t2["md5"],
                                     "data_min_max_gain": gdata[r]}
    for v in ("primary", "r5"):
        q = out / ("test_results_" + v + ".pdf")
        record["figures"][q.name] = {"md5": md5(q), "source": "results/" + rels[v]["path"], "source_md5": rels[v]["md5"],
                                     "data_for_passes": rdata[v]}
    (out / "figures_provenance.json").write_text(json.dumps(record, indent=1), encoding="utf-8", newline="")
    print("written:", pdf, md5(pdf))
    for name in sorted(record["figures"]):
        print("written:", out / name, record["figures"][name]["md5"])


if __name__ == "__main__":
    main()
