import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

INPUT_FILE = "meccha_chameleon_sentiment.csv"
OUTPUT_FILE = "sentiment_dashboard.png"

# Load results
df = pd.read_csv(INPUT_FILE)

counts = df["sentiment"].value_counts()
positive = counts.get("Positive", 0)
neutral = counts.get("Neutral", 0)
negative = counts.get("Negative", 0)
total = len(df)
avg_score = df["polarity_score"].mean()

labels = ["Positive", "Neutral", "Negative"]
values = [positive, neutral, negative]
percentages = [v / total * 100 for v in values]
colors = ["#1baf7a", "#888780", "#e34948"]

# Dark theme
BG = "#1a1a19"
CARD = "#242423"
TEXT_PRIMARY = "#ffffff"
TEXT_SECONDARY = "#c3c2b7"
TEXT_MUTED = "#898781"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["text.color"] = TEXT_PRIMARY

fig = plt.figure(figsize=(14, 8), facecolor=BG)
fig.patch.set_facecolor(BG)

gs = GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4,
              left=0.06, right=0.97, top=0.88, bottom=0.08)

# --- Title ---
fig.text(0.06, 0.95, "MECCHA CHAMELEON — Steam Review Sentiment Analysis",
         fontsize=15, fontweight="bold", color=TEXT_PRIMARY, va="top")
fig.text(0.06, 0.91, f"31,665 English reviews analyzed with TextBlob NLP",
         fontsize=11, color=TEXT_SECONDARY, va="top")

# --- Metric cards (top row) ---
metric_data = [
    ("POSITIVE", f"{positive:,}", f"{percentages[0]:.1f}%", "#1baf7a"),
    ("NEUTRAL",  f"{neutral:,}",  f"{percentages[1]:.1f}%", "#888780"),
    ("NEGATIVE", f"{negative:,}", f"{percentages[2]:.1f}%", "#e34948"),
]

for i, (label, val, pct, color) in enumerate(metric_data):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor(CARD)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.text(0.12, 0.78, label, transform=ax.transAxes,
            fontsize=9, color=color, fontweight="bold")
    ax.text(0.12, 0.38, val, transform=ax.transAxes,
            fontsize=22, color=TEXT_PRIMARY, fontweight="bold")
    ax.text(0.12, 0.12, f"{pct} of all reviews", transform=ax.transAxes,
            fontsize=9, color=TEXT_MUTED)

# --- Donut chart ---
ax_pie = fig.add_subplot(gs[1:, 0])
ax_pie.set_facecolor(BG)

wedges, _ = ax_pie.pie(
    percentages,
    colors=colors,
    startangle=140,
    wedgeprops={"width": 0.55, "edgecolor": BG, "linewidth": 3},
)
ax_pie.text(0, 0.08, f"{avg_score:.3f}", ha="center", va="center",
            fontsize=20, fontweight="bold", color=TEXT_PRIMARY)
ax_pie.text(0, -0.22, "avg score", ha="center", va="center",
            fontsize=9, color=TEXT_MUTED)

legend_patches = [mpatches.Patch(color=c, label=f"{l}  {p:.1f}%")
                  for c, l, p in zip(colors, labels, percentages)]
ax_pie.legend(handles=legend_patches, loc="lower center",
              bbox_to_anchor=(0.5, -0.12), ncol=1,
              frameon=False, fontsize=9,
              labelcolor=TEXT_SECONDARY)
ax_pie.set_title("Distribution", color=TEXT_MUTED, fontsize=9,
                 fontweight="bold", pad=10, loc="left")

# --- Horizontal bar chart ---
ax_bar = fig.add_subplot(gs[1, 1:])
ax_bar.set_facecolor(CARD)
for spine in ax_bar.spines.values():
    spine.set_visible(False)

bar_labels = ["Positive", "Neutral", "Negative"]
bar_values = [positive, neutral, negative]
bar_colors = ["#1baf7a", "#888780", "#e34948"]
y_pos = [2, 1, 0]

bars = ax_bar.barh(y_pos, bar_values, color=bar_colors, height=0.5,
                   edgecolor="none")
for bar, val in zip(bars, bar_values):
    ax_bar.text(bar.get_width() + 150, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=10,
                color=TEXT_PRIMARY, fontweight="bold")

ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(bar_labels, color=TEXT_SECONDARY, fontsize=10)
ax_bar.set_xticks([])
ax_bar.set_xlim(0, max(bar_values) * 1.22)
ax_bar.tick_params(left=False)
ax_bar.set_title("Review counts", color=TEXT_MUTED, fontsize=9,
                 fontweight="bold", pad=10, loc="left")
ax_bar.set_facecolor(CARD)
fig.add_artist(plt.Rectangle(
    (ax_bar.get_position().x0 - 0.01, ax_bar.get_position().y0 - 0.02),
    ax_bar.get_position().width + 0.02,
    ax_bar.get_position().height + 0.04,
    transform=fig.transFigure, color=CARD, zorder=-1
))

# --- Polarity scale ---
ax_scale = fig.add_subplot(gs[2, 1:])
ax_scale.set_facecolor(CARD)
for spine in ax_scale.spines.values():
    spine.set_visible(False)
ax_scale.set_xticks([])
ax_scale.set_yticks([])
ax_scale.set_title("Average polarity score", color=TEXT_MUTED, fontsize=9,
                   fontweight="bold", pad=10, loc="left")

from matplotlib.patches import FancyArrowPatch
import numpy as np

gradient = np.linspace(0, 1, 256).reshape(1, -1)
ax_scale.imshow(gradient, aspect="auto", extent=[-1, 1, 0.3, 0.7],
                cmap=plt.cm.RdYlGn, alpha=0.7)
ax_scale.axvline(avg_score, color=TEXT_PRIMARY, linewidth=2.5, ymin=0.1, ymax=0.9)
ax_scale.plot(avg_score, 0.5, "o", color=TEXT_PRIMARY, markersize=10, zorder=5)

ax_scale.text(-1, 0.05, "-1.0\nvery negative", ha="center", fontsize=8,
              color=TEXT_MUTED, va="top")
ax_scale.text(0, 0.05, "0.0\nneutral", ha="center", fontsize=8,
              color=TEXT_MUTED, va="top")
ax_scale.text(1, 0.05, "+1.0\nvery positive", ha="center", fontsize=8,
              color=TEXT_MUTED, va="top")
ax_scale.text(avg_score, 0.88, f"{avg_score:.3f}", ha="center", fontsize=11,
              color="#1baf7a", fontweight="bold", va="top")
ax_scale.set_xlim(-1.15, 1.15)
ax_scale.set_ylim(0, 1)

plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Dashboard saved as '{OUTPUT_FILE}'")
plt.show()