import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle

# ==============================
# Seaborn theme (whitegrid)
# ==============================
sns.set_theme(style="whitegrid")

# ==============================
# RCParams — journal-ready for Overleaf/PDF
# ==============================
def set_journal_rcparams(layout="double"):
    """
    layout: 'single' (one-column figure) or 'double' (two-column figure).
    """
    if layout == "single":
        figsize = (6.5, 3.8)
        base, lbl, ttl, tick, leg = 13, 15, 15, 12, 12
    else:  # 'double' recommended
        figsize = (12, 4.8)
        base, lbl, ttl, tick, leg = 14, 16, 16, 13, 13

    plt.rcParams.update({
        "figure.figsize": figsize,
        "font.family": "serif",      # align with LaTeX
        "font.size": base,
        "axes.labelsize": lbl,
        "axes.titlesize": ttl,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "legend.fontsize": leg,
        "lines.linewidth": 1.5,
        "lines.markersize": 9,
        "legend.frameon": False,
        "grid.alpha": 0.3,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,          # proper font embedding for Overleaf
        "ps.fonttype": 42
    })

# ==============================
# Compute per-budget means/std and absolute deltas
# ==============================
def compute_deltas(datasets):
    """
    datasets[name] = {
        "f1_base": array per-client,
        "eod_base": array per-client,
        "f1": [array per-client @b=0.05, @0.10, @0.15],
        "eod": [array per-client @b=0.05, @0.10, @0.15]
    }
    Returns: delta_f1, delta_eod, std_f1, std_eod (dicts of np.array with shape (n_budgets,))
    """
    delta_f1, delta_eod = {}, {}
    std_f1, std_eod = {}, {}

    for name, data in datasets.items():
        f1_base  = float(np.mean(data["f1_base"]))   # no-fairness baseline (mean over clients)
        eod_base = float(np.mean(data["eod_base"]))

        mean_f1  = np.array([arr.mean() for arr in data["f1"]])
        mean_eod = np.array([arr.mean() for arr in data["eod"]])

        std_f1[name]  = np.array([arr.std(ddof=1) for arr in data["f1"]])
        std_eod[name] = np.array([arr.std(ddof=1) for arr in data["eod"]])

        # Absolute deltas: ΔF1 (>0 = performance loss), ΔEOD (>0 = fairness gain)
        delta_f1[name]  = f1_base - mean_f1
        delta_eod[name] = eod_base - mean_eod

    return delta_f1, delta_eod, std_f1, std_eod

# ==============================
# Two-panel plot: ΔF1 (left) and ΔEOD (right)
# ==============================
def plot_budget_analysis(
    budget, delta_f1, delta_eod, std_f1, std_eod,
    dataset_order=None, outfile=None
):
    # Enforce a deterministic order in legend/plots if provided
    names = dataset_order if dataset_order else list(delta_f1.keys())

    # B/W-friendly styles
    base_markers   = ['o','s','^','D','v','P','X']
    base_linestyles = ['-','--','-.',':']
    markers = cycle(base_markers)
    linestyles = cycle(base_linestyles)

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)  # shared X (budget)

    # ---- Panel (a): ΔF1 ----
    for i, name in enumerate(names):
        m, ls = next(markers), next(linestyles)
        # alternate filled/empty markers for grayscale clarity
        mfc = 'none' if i % 2 else None
        ax1.errorbar(
            budget, delta_f1[name], yerr=std_f1[name],
            marker=m, linestyle=ls, linewidth=2.2, capsize=5,
            label=name, markerfacecolor=mfc
        )
    ax1.set_xlabel("Budget")
    ax1.set_ylabel("ΔF1 (absolute loss)")
    ax1.set_title("Performance loss")
    ax1.set_xticks(budget)
    ax1.set_ylim(bottom=0)             # start Y from 0
    ax1.grid(True, alpha=0.3)
    # Panel label
    ax1.text(-0.12, 1.05, "(a)", transform=ax1.transAxes, fontsize=16, fontweight="bold")

    # ---- Panel (b): ΔEOD ----
    markers   = cycle(base_markers)
    linestyles = cycle(base_linestyles)
    for i, name in enumerate(names):
        m, ls = next(markers), next(linestyles)
        mfc = 'none' if i % 2 else None
        ax2.errorbar(
            budget, delta_eod[name], yerr=std_eod[name],
            marker=m, linestyle=ls, linewidth=2.2, capsize=5,
            label=name, markerfacecolor=mfc
        )
    ax2.set_xlabel("Budget")
    ax2.set_ylabel("ΔEOD (absolute gain)")
    ax2.set_title("Fairness gain")
    ax2.set_xticks(budget)
    ax2.set_ylim(bottom=0)             # start Y from 0
    ax2.grid(True, alpha=0.3)
    ax2.text(-0.12, 1.05, "(b)", transform=ax2.transAxes, fontsize=16, fontweight="bold")

    # Legend to the right of the right panel
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(
        handles, labels,
        loc="upper left",
        bbox_to_anchor=(1.05, 1.0),
        borderaxespad=0
    )

    # Leave room on the right for the legend
    plt.tight_layout(rect=[0, 0, 0.95, 1])

    if outfile:
        plt.savefig(outfile, bbox_inches="tight")
    plt.show()

# ==============================
# Example — with your data
# ==============================
if __name__ == "__main__":
    set_journal_rcparams(layout="double")

    budget = np.array([0.05, 0.10, 0.15])

    datasets = {
        "Income": {
            "f1_base": np.array([0.80,0.78,0.77,0.79,0.79,0.78,0.79,0.79,0.80,0.81]),
            "eod_base": np.array([0.47,0.52,0.95,0.50,0.63,0.49,0.84,0.43,0.58,0.77]),
            "f1": [
                np.array([0.76,0.73,0.70,0.75,0.76,0.74,0.76,0.75,0.74,0.76]),
                np.array([0.69,0.71,0.66,0.70,0.72,0.69,0.67,0.70,0.72,0.70]),
                np.array([0.69,0.71,0.66,0.70,0.72,0.69,0.67,0.70,0.71,0.65])
            ],
            "eod": [
                np.array([0.28,0.24,0.38,0.25,0.32,0.32,0.35,0.23,0.27,0.41]),
                np.array([0.20,0.19,0.20,0.19,0.19,0.20,0.20,0.18,0.20,0.29]),
                np.array([0.20,0.19,0.16,0.19,0.19,0.20,0.18,0.17,0.17,0.18])
            ]
        },

        "Employment": {
            "f1_base": np.array([0.77,0.79,0.79,0.78,0.80,0.79,0.79,0.79,0.80,0.79]),
            "eod_base": np.array([0.32,0.47,0.48,0.31,0.42,0.43,0.37,0.43,0.54,0.54]),
            "f1": [
                np.array([0.74,0.75,0.72,0.73,0.73,0.73,0.72,0.74,0.75,0.75]),
                np.array([0.73,0.73,0.70,0.71,0.70,0.68,0.72,0.72,0.73,0.72]),
                np.array([0.73,0.72,0.70,0.71,0.70,0.68,0.72,0.72,0.72,0.72])
            ],
            "eod": [
                np.array([0.28,0.19,0.23,0.26,0.27,0.26,0.27,0.20,0.25,0.30]),
                np.array([0.20,0.18,0.20,0.18,0.20,0.20,0.20,0.20,0.19,0.20]),
                np.array([0.20,0.18,0.18,0.19,0.20,0.20,0.20,0.19,0.18,0.18])
            ]
        },

        "Compas": {
            "f1_base": np.array([0.74,0.78,0.75,0.75,0.74,0.74,0.79,0.73,0.77,0.74]),
            "eod_base": np.array([0.72,0.78,0.54,0.46,0.78,0.42,0.55,0.69,0.46,0.59]),
            "f1": [
                np.array([0.72,0.75,0.72,0.69,0.67,0.69,0.75,0.67,0.72,0.69]),
                np.array([0.63,0.66,0.66,0.69,0.63,0.69,0.75,0.68,0.72,0.64]),
                np.array([0.63,0.63,0.63,0.69,0.63,0.69,0.75,0.68,0.71,0.65])
            ],
            "eod": [
                np.array([0.35,0.47,0.36,0.18,0.26,0.18,0.19,0.18,0.15,0.28]),
                np.array([0.20,0.30,0.28,0.19,0.20,0.18,0.19,0.19,0.15,0.17]),
                np.array([0.20,0.19,0.20,0.19,0.19,0.20,0.18,0.19,0.17,0.18])
            ]
        },

        "MEPS": {
            "f1_base": np.array([0.87,0.81,0.82,0.80,0.81,0.83,0.84,0.82,0.82,0.84]),
            "eod_base": np.array([0.77,0.79,0.93,0.68,0.81,0.66,0.75,0.70,0.90,0.61]),
            "f1": [
                np.array([0.80,0.79,0.77,0.77,0.73,0.77,0.77,0.77,0.77,0.79]),
                np.array([0.75,0.70,0.77,0.77,0.73,0.77,0.77,0.76,0.76,0.79]),
                np.array([0.72,0.70,0.77,0.77,0.73,0.77,0.77,0.76,0.76,0.78])
            ],
            "eod": [
                np.array([0.27,0.32,0.19,0.19,0.19,0.20,0.20,0.19,0.16,0.16]),
                np.array([0.22,0.20,0.17,0.19,0.18,0.19,0.20,0.19,0.16,0.13]),
                np.array([0.18,0.20,0.17,0.19,0.18,0.19,0.20,0.19,0.17,0.15])
            ]
        },

        "Income-3": {
            "f1_base": np.array([0.69,0.68,0.69,0.69,0.70,0.70,0.68,0.68,0.70,0.70]),
            "eod_base": np.array([0.63,0.65,0.63,0.70,0.90,0.70,0.67,0.75,0.79,0.73]),
            "f1": [
                np.array([0.65,0.62,0.66,0.65,0.66,0.64,0.64,0.66,0.64,0.66]),
                np.array([0.62,0.59,0.61,0.59,0.58,0.60,0.56,0.62,0.58,0.59]),
                np.array([0.53,0.52,0.55,0.52,0.53,0.57,0.49,0.55,0.51,0.52])
            ],
            "eod": [
                np.array([0.32,0.45,0.48,0.43,0.53,0.38,0.39,0.40,0.47,0.43]),
                np.array([0.30,0.31,0.39,0.36,0.31,0.32,0.37,0.35,0.32,0.26]),
                np.array([0.26,0.19,0.31,0.20,0.22,0.29,0.27,0.29,0.23,0.19])
            ]
        },

         "Education": {
            "f1_base": np.array([0.68,0.68,0.66,0.68,0.71,0.69,0.69,0.68,0.68,0.71]),
            "eod_base": np.array([0.70,0.66,0.75,0.83,0.89,0.78,0.76,0.78,0.81,0.90]),
            "f1": [
                np.array([0.66,0.65,0.64,0.62,0.65,0.65,0.67,0.65,0.65,0.67]),
                np.array([0.60,0.60,0.58,0.58,0.63,0.58,0.57,0.57,0.59,0.59]),
                np.array([0.52,0.54,0.58,0.58,0.63,0.53,0.55,0.57,0.59,0.58])
            ],
            "eod": [
                np.array([0.49,0.41,0.26,0.28,0.41,0.36,0.38,0.30,0.32,0.53]),
                np.array([0.34,0.27,0.14,0.18,0.17,0.25,0.25,0.18,0.20,0.20]),
                np.array([0.15,0.13,0.14,0.17,0.15,0.11,0.15,0.18,0.20,0.19])
            ]
        },
    }

    # Compute deltas and std
    delta_f1, delta_eod, std_f1, std_eod = compute_deltas(datasets)

    # Optional: control legend order explicitly
    order = ["Income", "Employment", "Compas", "MEPS","Income-3","Education"]

    # Generate figure and save PDF
    plot_budget_analysis(
        budget, delta_f1, delta_eod, std_f1, std_eod,
        dataset_order=order,
        outfile="budget_analysis.pdf"
    )
