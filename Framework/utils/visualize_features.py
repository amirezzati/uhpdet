import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_joint_pdf(df, feature_x, feature_y, out_path):
    """
    Empirical joint PDF (PMF) of two DISCRETE features, saved as a heatmap PNG.
    P(X = x, Y = y) for all observed (x, y).
    """
    # counts of each (x, y)
    joint = (
        df
        .groupby([feature_x, feature_y])
        .size()
        .reset_index(name="count")
    )

    # turn counts into probabilities
    total = len(df)
    joint["prob"] = joint["count"] / total

    # make a 2D table: rows = feature_y values, cols = feature_x values
    pdf_mat = joint.pivot(
        index=feature_y,
        columns=feature_x,
        values="prob"
    ).fillna(0.0).sort_index().sort_index(axis=1)

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        pdf_mat,
        annot=True,        # show numbers in cells
        fmt=".3f",         # probability format
        cbar_kws={"label": "P(X, Y)"}
    )

    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.title(f"Joint PDF of {feature_x} and {feature_y}")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved joint PDF plot to: {out_path}")



def plot_joint_pdf_pretty(
    df,
    feature_x,
    feature_y,
    out_path,
    bin_step=0.05,          # e.g. 0.05 or 0.10; use 0.01 for very fine
    tick_step=0.10,         # show ticks every 0.10 (keeps axes readable)
    annot=True,
    annot_fmt=".3f",
    mask_below=0.0,         # e.g. 0.002 to hide tiny cells
    dpi=300
):
    """
    Paper-style empirical joint PMF heatmap of two numeric features.
    Features are binned to a fixed step so axis labels are clean.
    """

    # --- 1) Coerce to numeric & drop NaNs
    x = pd.to_numeric(df[feature_x], errors="coerce")
    y = pd.to_numeric(df[feature_y], errors="coerce")
    sub = df.loc[x.notna() & y.notna(), [feature_x, feature_y]].copy()

    # --- 2) Bin to fixed step (e.g., 0.05) and round to 2 decimals
    # example: bin_step=0.05 -> values become 0.00, 0.05, 0.10, ...
    sub[feature_x] = (np.round(sub[feature_x] / bin_step) * bin_step).round(2)
    sub[feature_y] = (np.round(sub[feature_y] / bin_step) * bin_step).round(2)

    # --- 3) Joint probabilities
    joint = sub.groupby([feature_x, feature_y]).size().reset_index(name="count")
    joint["prob"] = joint["count"] / len(sub)

    pdf_mat = (
        joint.pivot(index=feature_y, columns=feature_x, values="prob")
        .fillna(0.0)
        .sort_index()
        .sort_index(axis=1)
    )

    # --- 4) Build nice ticks (every tick_step) but keep full matrix
    x_vals = pdf_mat.columns.values.astype(float)
    y_vals = pdf_mat.index.values.astype(float)

    def pick_ticks(vals, step):
        # choose ticks that are close to multiples of `step`
        # (robust to floating errors)
        eps = step / 10
        return [i for i, v in enumerate(vals) if abs((v / step) - round(v / step)) < eps]

    xtick_idx = pick_ticks(x_vals, tick_step)
    ytick_idx = pick_ticks(y_vals, tick_step)

    # --- 5) Styling
    sns.set_theme(style="white", context="paper")  # good default for papers

    fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=dpi)

    mask = None
    if mask_below > 0:
        mask = pdf_mat.values < mask_below

    hm = sns.heatmap(
        pdf_mat,
        ax=ax,
        cmap=sns.cubehelix_palette(as_cmap=True),             # nicer than default; change if you want
        square=True,
        linewidths=0.3,
        linecolor="white",
        annot=annot,
        fmt=annot_fmt,
        annot_kws={"fontsize": 7},
        mask=mask,
        cbar_kws={"label": r"$P(X, Y)$", "shrink": 0.9},
        vmin=0.0
    )

    # axis labels + title
    ax.set_xlabel(feature_x, fontsize=11)
    ax.set_ylabel(feature_y, fontsize=11)
    ax.set_title(f"Joint PMF of {feature_x} and {feature_y}", fontsize=12, pad=10)

    # set tick positions and labels (2 decimals)
    ax.set_xticks([i + 0.5 for i in xtick_idx])
    ax.set_xticklabels([f"{x_vals[i]:.2f}" for i in xtick_idx], rotation=45, ha="right", fontsize=9)

    ax.set_yticks([i + 0.5 for i in ytick_idx])
    ax.set_yticklabels([f"{y_vals[i]:.2f}" for i in ytick_idx], rotation=0, fontsize=9)

    # tighten layout & save
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved pretty joint PMF plot to: {out_path}")




def main(dataset: str, model_name: str):

    # ================================
    # Map model-name → folder path
    # ================================
    # if model_name == "llava":
    #     model_path = "llava-hf/llava-1.5-7b-hf"
    # elif model_name == "blip":
    #     model_path = "Salesforce/instructblip-vicuna-7b"
    # elif model_name == "qwen":
    #     model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    # else:
    #     raise ValueError(f"Unknown model-name: {model_name}")

    # ================================
    # Build file path
    # ================================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "Results", "Features")
    file_name = f"data_features.csv"
    file_path = os.path.join(base_dir, model_name, dataset, file_name)

    print(f"Reading features from: {file_path}")

    # Load CSV
    df = pd.read_csv(file_path)

    df['FSA_ITC'] = df['QA_FSA'] * df['QA_ITC']

    # Feature columns (assumes first col = image_id, second = something else, last = hal)
    print(df.columns)
    # exit(0)
    features = [x for x in df.columns[4:].tolist() if 'QA' in x][:4] #+ [df.columns[-1]]

    print(features)
    # exit(0)

    # ==========================================
    # Output directories
    # ==========================================
    base_out_dir = os.path.join(base_dir, model_name, dataset, "plots")
    scatter_dir = os.path.join(base_out_dir, "scatter")
    hist_dir = os.path.join(base_out_dir, "hist")
    joint_dir = os.path.join(base_out_dir, "joint_pdf")

    os.makedirs(scatter_dir, exist_ok=True)
    os.makedirs(hist_dir, exist_ok=True)
    os.makedirs(joint_dir, exist_ok=True)

    # ==========================================
    # Joint PDF
    # ==========================================
    df0 = df[df["hal_label"] == 0]
    df1 = df[df["hal_label"] == 1]
    for fx, fy in itertools.combinations(features, 2):
        # filenames (safe-ish)
        base = f"{fx}__{fy}".replace("/", "_")

        out0 = os.path.join(joint_dir, f"joint_{base}_hal0.png")
        out1 = os.path.join(joint_dir, f"joint_{base}_hal1.png")

        # hal=0
        plot_joint_pdf_pretty(
            df0, fx, fy, out0,
            bin_step=0.2,
            tick_step=0.1,
            mask_below=0.0,
            annot=True,
            annot_fmt=".3f",
            dpi=300
        )

        # hal=1
        plot_joint_pdf_pretty(
            df1, fx, fy, out1,
            bin_step=0.2,
            tick_step=0.1,
            mask_below=0.0,
            annot=True,
            annot_fmt=".3f",
            dpi=300
        )

        print(f"Done: {fx} vs {fy}")


    # plot_joint_pdf_pretty(
    #     df[df["hal_label"] == 0],
    #     "QA_ITA",
    #     "QA_FSC",
    #     os.path.join(joint_dir, "joint_pretty_QA_ITA_QA_FSC_hal0.png"),
    #     bin_step=0.05,    # try 0.05 or 0.10 (paper-friendly)
    #     tick_step=0.10,   # show fewer ticks
    #     mask_below=0.002, # optional: hide tiny probs
    #     annot=True,
    #     annot_fmt=".3f",
    # )
    # plot_joint_pdf_pretty(
    #     df[df["hal_label"] == 1],
    #     "QA_ITA",
    #     "QA_FSC",
    #     os.path.join(joint_dir, "joint_pretty_QA_ITA_QA_FSC_hal0.png"),
    #     bin_step=0.05,    # try 0.05 or 0.10 (paper-friendly)
    #     tick_step=0.10,   # show fewer ticks
    #     mask_below=0.002, # optional: hide tiny probs
    #     annot=True,
    #     annot_fmt=".3f",
    # )

    plot_joint_pdf(
        df[df["hal_label"] == 0],
        "QA_ITA",
        "QA_FSC",
        os.path.join(joint_dir, "joint_pdf_SCS_ITA_SCS_FSC_hal0.png")
    )
    plot_joint_pdf(
        df[df["hal_label"] == 1],
        "QA_ITA",
        "QA_FSC",
        os.path.join(joint_dir, "joint_pdf_SCS_ITA_SCS_FSC_hal1.png")
    )
    plot_joint_pdf(
        df[df["hal_label"] == 0],
        "QA_FSC",
        "QA_FSA",
        os.path.join(joint_dir, "joint_pdf_SCS_FSC_SCS_FSA_hal0.png")
    )
    plot_joint_pdf(
        df[df["hal_label"] == 1],
        "QA_FSC",
        "QA_FSA",
        os.path.join(joint_dir, "joint_pdf_SCS_FSC_SCS_FSA_hal1.png")
    )
    plot_joint_pdf(
        df[df["hal_label"] == 0],
        "QA_ITC",
        "QA_FSA",
        os.path.join(joint_dir, "joint_pdf_SCS_ITC_SCS_FSA_hal0.png")
    )
    plot_joint_pdf(
        df[df["hal_label"] == 1],
        "QA_ITC",
        "QA_FSA",
        os.path.join(joint_dir, "joint_pdf_SCS_ITC_SCS_FSA_hal1.png")
    )
    # return
    

    # ==========================================
    # 1) Pairwise scatter plots with % per class
    # ==========================================
    # for i in range(len(features)):
    #     for j in range(i + 1, len(features)):
    #         feature_1 = features[i]
    #         feature_2 = features[j]

    #         pair_df = df[[feature_1, feature_2, "hal"]].copy()

    #         # total count for each class (to compute percentage of that class)
    #         class_totals = pair_df["hal"].value_counts().to_dict()

    #         # how many points at each (x, y, hal)
    #         grouped = (
    #             pair_df
    #             .groupby([feature_1, feature_2, "hal"])
    #             .size()
    #             .reset_index(name="count")
    #         )

    #         # percentage of that class that lies at this (x, y)
    #         grouped["perc"] = grouped.apply(
    #             lambda row: 100.0 * row["count"] / class_totals[row["hal"]],
    #             axis=1,
    #         )

    #         # pivot so each (x, y) has two columns: perc_hal0, perc_hal1
    #         pivot = grouped.pivot_table(
    #             index=[feature_1, feature_2],
    #             columns="hal",
    #             values="perc",
    #             fill_value=0.0,
    #         )

    #         pivot = pivot.rename(columns={0: "perc_hal0", 1: "perc_hal1"}).reset_index()

    #         plt.figure(figsize=(8, 6))

    #         # normal scatter of all points, colored by hal
    #         scatter = plt.scatter(
    #             pair_df[feature_1],
    #             pair_df[feature_2],
    #             c=pair_df["hal"],
    #             cmap="coolwarm",
    #             alpha=0.7,
    #         )

    #         # one label per (x, y): "<hal0%> , <hal1%>"
    #         for _, row in pivot.iterrows():
    #             x = row[feature_1]
    #             y = row[feature_2]
    #             p0 = row.get("perc_hal0", 0.0)
    #             p1 = row.get("perc_hal1", 0.0)
    #             label = f"{p0:.1f} , {p1:.1f}"
    #             plt.text(x, y, label, fontsize=7, ha="center", va="center")

    #         plt.xlabel(feature_1)
    #         plt.ylabel(feature_2)
    #         plt.title(
    #             f"Scatter: {feature_1} vs {feature_2} (hal colored)\n"
    #             f"text = [% of hal=0 , % of hal=1]\n"
    #             f"({dataset}, {model_name})"
    #         )

    #         cbar = plt.colorbar(scatter, label="hal")
    #         cbar.set_ticks([0, 1])
    #         cbar.set_ticklabels(["0", "1"])

    #         out_file = f"scatter_{feature_1}_{feature_2}.png"
    #         plt.savefig(os.path.join(scatter_dir, out_file))
    #         plt.close()

    # ==========================================
    # 2) Histograms per feature for hal=0 vs hal=1
    # ==========================================
    for feature in features:
        plt.figure(figsize=(8, 6))

        vals_hal0 = df[df["hal_label"] == 0][feature].dropna()
        vals_hal1 = df[df["hal_label"] == 1][feature].dropna()

        # common bin edges for both classes
        all_vals = pd.concat([vals_hal0, vals_hal1])
        bins = np.linspace(all_vals.min(), all_vals.max(), 31)  # 30 bins

        # weights so that sum of bins for each class = 100 (%)
        w0 = np.ones_like(vals_hal0) / len(vals_hal0) * 100.0
        w1 = np.ones_like(vals_hal1) / len(vals_hal1) * 100.0

        plt.hist(
            vals_hal0,
            bins=bins,
            weights=w0,
            alpha=0.6,
            label="hal_label = 0",
        )
        plt.hist(
            vals_hal1,
            bins=bins,
            weights=w1,
            alpha=0.6,
            label="hal_label = 1",
        )

        plt.xlabel(feature)
        plt.ylabel("Percentage of class in bin (%)")
        plt.title(f"Histogram of {feature} by hal\n({dataset}, {model_name})")
        plt.legend()

        out_file = f"hist_{feature}.png"
        plt.savefig(os.path.join(hist_dir, out_file))
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="AMBER_yes_questions",
        help="Dataset folder name under Results/Features/<model>/ (e.g. AMBER_yes_questions).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="blip",
        choices=["llava", "blip", "qwen", "internvl"],
        help="Model name matching the subfolder under Results/Features/.",
    )

    args = parser.parse_args()

    print(f"Running scatter & hist plots for dataset={args.dataset}, model={args.model_name}")
    main(args.dataset, args.model_name)


# how to run
# python visualize_features.py --dataset PHD_yes_questions --model-name qwen