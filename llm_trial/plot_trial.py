"""Figure for the blinded LLM trial: predicted vs actual knee for numeric analogs, LLM A and LLM B."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "reports" / "severson_b2" / "llm_trial"
plt.style.use(str(ROOT / "journal.mplstyle"))
mm = 1 / 25.4
B = pd.read_csv(R / "scored.csv", index_col=0)
typ = B["actual"] > 300
panels = [("analogs_k8", "numeric analogs (8 cells)", "#D55E00"),
          ("llm_A", "LLM A: numbers + analogs", "#0072B2"),
          ("llm_B", "LLM B: + chemistry, protocol physics,\nmechanism knowledge", "#009E73")]
fig, axes = plt.subplots(1, 3, figsize=(170 * mm, 66 * mm), layout="constrained", sharey=True)
for ax, (m, title, col) in zip(axes, panels):
    e = (B[m] - B["actual"]).abs()
    ax.plot([50, 430], [50, 430], color="#6b6b6b", lw=0.8, ls="--")
    ax.scatter(B.loc[typ, "actual"], B.loc[typ, m], s=14, color=col, alpha=0.8, label="typical cells")
    ax.scatter(B.loc[~typ, "actual"], B.loc[~typ, m], s=40, marker="D", color=col, edgecolor="black", lw=0.6,
               label="short-lived cells")
    ax.set_xlim(50, 430); ax.set_ylim(50, 430)
    ax.set_title(f"{title}\nMAE {e.mean():.0f} (typical {e[typ].mean():.0f}, short-lived {e[~typ].mean():.0f})",
                 fontsize=7.5, loc="left")
    ax.set_xlabel("actual knee cycle")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("predicted from first 100 cycles")
axes[0].legend(fontsize=7, loc="upper left")
fig.savefig(R / "llm_trial.png"); fig.savefig(R / "llm_trial.pdf")
