from pathlib import Path
from enum import Enum

import json
import pandas as pd
import matplotlib.pyplot as plt

PRODUCTS_GREEDY_SPECULATIVE_BATCH_SIZE = {
    1: "results_product_500_greedy_speculative_bs_1_report.txt", 
    4: "results_product_500_greedy_speculative_bs_4_report.txt",
    16: "results_product_500_greedy_speculative_bs_16_report.txt",
    32: "results_product_500_greedy_speculative_bs_32_report.txt"
}

PRODUCTS_BEAM_SEARCH_SPECULATIVE_BATCH_SIZE = {
    1: "results_product_500_beam_search_speculative_bs_1_report.txt", 
    2: "results_product_500_beam_search_speculative_bs_2_report.txt",
    3: "results_product_500_beam_search_speculative_bs_3_report.txt",
    4: "results_product_500_beam_search_speculative_bs_4_report.txt"
}

RETRO_BEAM_SEARCH_SPECULATIVE_BS_1_NBEST = {
    5: "results_retro_500_beam_search_speculative_bs_1_nbest_5_report.txt", 
    10: "results_retro_500_beam_search_speculative_bs_1_nbest_10_report.txt",
    15: "results_retro_500_beam_search_speculative_bs_1_nbest_15_report.txt",
    20: "results_retro_500_beam_search_speculative_bs_1_nbest_20_report.txt"
}

RETRO_BEAM_SEARCH_SPECULATIVE_NBEST_10_BATCH_SIZE = {
    1: "results_retro_500_beam_search_speculative_bs_1_nbest_10_report.txt",
    2: "results_retro_500_beam_search_speculative_bs_2_nbest_10_report.txt",
    4: "results_retro_500_beam_search_speculative_bs_4_nbest_10_report.txt",
    8: "results_retro_500_beam_search_speculative_bs_8_nbest_10_report.txt"
}

class Experiment(Enum):
    PRODUCTS_GREEDY_SPECULATIVE = 1
    PRODUCTS_BEAM_SEARCH_SPECULATIVE = 2
    RETRO_BEAM_SEARCH_SPECULATIVE_BS_1 = 3
    RETRO_BEAM_SEARCH_SPECULATIVE_NBEST_10 = 4

EXPERIMENTS = {
    Experiment.PRODUCTS_GREEDY_SPECULATIVE: PRODUCTS_GREEDY_SPECULATIVE_BATCH_SIZE,
    Experiment.PRODUCTS_BEAM_SEARCH_SPECULATIVE: PRODUCTS_BEAM_SEARCH_SPECULATIVE_BATCH_SIZE,
    Experiment.RETRO_BEAM_SEARCH_SPECULATIVE_BS_1: RETRO_BEAM_SEARCH_SPECULATIVE_BS_1_NBEST,
    Experiment.RETRO_BEAM_SEARCH_SPECULATIVE_NBEST_10: RETRO_BEAM_SEARCH_SPECULATIVE_NBEST_10_BATCH_SIZE,
}


def load_reports(experiment: Experiment) -> dict[int, pd.DataFrame]:
    report = {}
    for k, path in EXPERIMENTS[experiment].items():
        with open(Path(path), "r") as file:
            records = []
            for line in file.readlines():
                records.append(pd.DataFrame.from_dict(json.loads(line), orient="index").T)
        records = pd.concat(records).reset_index(drop=True)
        report[k] = records
    return report


def figure_products_greedy_speculative(
    ax,  # Array of axes
    major_text_size: int = 16, 
    minor_text_size: int = 14,
    marker_size: int = 8,
    alpha=1.0,
):
    # Products greedy speculative
    report = load_reports(Experiment.PRODUCTS_GREEDY_SPECULATIVE)
    batch_sizes = sorted(report.keys())
    axs = {}
    for i, batch_size in enumerate(batch_sizes):
        axs[batch_size] = ax[i]  # Just use the provided axes directly
    
    # Add 'A' label to the leftmost subplot
    axs[1].text(-0.25, 1.03, 'A', transform=axs[1].transAxes,
                fontsize=23, fontweight='bold', va='center')
    
    for batch_size in report.keys():
        results = report[batch_size]
        unique_n_drafts = sorted(results["n_drafts"].unique().tolist())
        for i in unique_n_drafts:
            axs[batch_size].plot(
                results[results["n_drafts"] == i]["draft_len"], 
                results[results["n_drafts"] == i]["total_seconds"], 
                "-s",
                markersize=marker_size,
                alpha=alpha,
                label=f"{i} drafts"
            )
        axs[batch_size].grid()
        axs[batch_size].set_ylim(5, 60)
        axs[batch_size].set_title(f"Batch size {batch_size}", size=minor_text_size)
        axs[batch_size].tick_params(axis='both', labelsize=minor_text_size)
        axs[batch_size].xaxis.label.set_size(minor_text_size)
        axs[batch_size].yaxis.label.set_size(minor_text_size)
        axs[batch_size].set_xlabel("Draft length")
        if batch_size != 1:  # Remove y-axis labels for all but first subplot
            axs[batch_size].set_yticklabels([])

    axs[1].set_ylabel("Total seconds")
    axs[32].legend(loc="upper right", fontsize=minor_text_size)
    return axs


def figure_products_beam_search_speculative(
    ax,  # Array of axes
    major_text_size: int = 16, 
    minor_text_size: int = 14,
    marker_size: int = 8,
    alpha=1.0,
):
    # Products greedy speculative
    report = load_reports(Experiment.PRODUCTS_BEAM_SEARCH_SPECULATIVE)
    batch_sizes = sorted(report.keys())
    axs = {}
    for i, batch_size in enumerate(batch_sizes):
        axs[batch_size] = ax[i]  # Just use the provided axes directly
    
    # Add 'B' label to the leftmost subplot
    axs[1].text(-0.25, 1.05, 'B', transform=axs[1].transAxes,
                fontsize=23, fontweight='bold', va='center')
    
    for batch_size in report.keys():
        results = report[batch_size]
        unique_n_drafts = sorted(results["n_drafts"].unique().tolist())
        for i in unique_n_drafts:
            axs[batch_size].plot(
                results[results["n_drafts"] == i]["draft_len"], 
                results[results["n_drafts"] == i]["total_seconds"], 
                "-s",
                markersize=marker_size,
                alpha=alpha,
                label=f"{i} drafts"
            )
        axs[batch_size].grid()
        axs[batch_size].set_ylim(60, 150)
        axs[batch_size].set_title(f"Batch size {batch_size}", size=minor_text_size)
        axs[batch_size].tick_params(axis='both', labelsize=minor_text_size)
        axs[batch_size].xaxis.label.set_size(minor_text_size)
        axs[batch_size].yaxis.label.set_size(minor_text_size)
        axs[batch_size].set_xlabel("Draft length")
        if batch_size != 1:  # Remove y-axis labels for all but first subplot
            axs[batch_size].set_yticklabels([])

    axs[1].set_ylabel("Total seconds")
    axs[4].legend(loc="upper left", fontsize=minor_text_size)
    return axs


def figure_retro_beam_search_speculative_bs_1(
    ax,  # Array of axes
    major_text_size: int = 16, 
    minor_text_size: int = 14,
    marker_size: int = 8,
    alpha=1.0,
):
    # Products greedy speculative
    report = load_reports(Experiment.RETRO_BEAM_SEARCH_SPECULATIVE_BS_1)
    n_best_values = sorted(report.keys())
    axs = {}
    for i, n_best in enumerate(n_best_values):
        axs[n_best] = ax[i]  # Just use the provided axes directly
    
    # Add 'C' label to the leftmost subplot
    axs[5].text(-0.25, 1.035, 'C', transform=axs[5].transAxes,
                fontsize=23, fontweight='bold', va='center')
    
    for n_best in report.keys():
        results = report[n_best]
        unique_n_drafts = sorted(results["n_drafts"].unique().tolist())
        for i in unique_n_drafts:
            axs[n_best].plot(
                results[results["n_drafts"] == i]["draft_len"], 
                results[results["n_drafts"] == i]["total_seconds"], 
                "-s",
                markersize=marker_size,
                alpha=alpha,
                label=f"{i} drafts"
            )
        axs[n_best].grid()
        axs[n_best].set_ylim(150, 410)
        axs[n_best].set_title(f"{n_best} best sequences", size=minor_text_size)
        axs[n_best].tick_params(axis='both', labelsize=minor_text_size)
        axs[n_best].xaxis.label.set_size(minor_text_size)
        axs[n_best].yaxis.label.set_size(minor_text_size)
        axs[n_best].set_xlabel("Draft length")
        if n_best != 5:  # Remove y-axis labels for all but first subplot
            axs[n_best].set_yticklabels([])

    axs[5].set_ylabel("Total seconds")
    axs[5].legend(loc="upper right", fontsize=minor_text_size)
    return axs

def figure_retro_beam_search_speculative_nbest_10(
    ax,  # Array of axes
    major_text_size: int = 16, 
    minor_text_size: int = 14,
    marker_size: int = 8,
    alpha=1.0,
):
    # Products greedy speculative
    report = load_reports(Experiment.RETRO_BEAM_SEARCH_SPECULATIVE_NBEST_10)
    batch_sizes = sorted(report.keys())
    axs = {}
    for i, batch_size in enumerate(batch_sizes):
        axs[batch_size] = ax[i]  # Just use the provided axes directly
    
    # Add 'D' label to the leftmost subplot
    axs[1].text(-0.25, 1.03, 'D', transform=axs[1].transAxes,
                fontsize=23, fontweight='bold', va='center')
    
    for batch_size in report.keys():
        results = report[batch_size]
        unique_n_drafts = sorted(results["n_drafts"].unique().tolist())
        for i in unique_n_drafts:
            axs[batch_size].plot(
                results[results["n_drafts"] == i]["draft_len"], 
                results[results["n_drafts"] == i]["total_seconds"], 
                "-s",
                markersize=marker_size,
                alpha=alpha,
                label=f"{i} drafts"
            )
        axs[batch_size].grid()
        axs[batch_size].set_ylim(40, 330)
        axs[batch_size].set_title(f"Batch size {batch_size}", size=minor_text_size)
        axs[batch_size].tick_params(axis='both', labelsize=minor_text_size)
        axs[batch_size].xaxis.label.set_size(minor_text_size)
        axs[batch_size].yaxis.label.set_size(minor_text_size)
        axs[batch_size].set_xlabel("Draft length")
        if batch_size != 1:  # Remove y-axis labels for all but first subplot
            axs[batch_size].set_yticklabels([])

    axs[1].set_ylabel("Total seconds")
    axs[1].legend(loc="lower left", fontsize=minor_text_size - 3)
    return axs


if __name__ == "__main__":
    fig = plt.figure(figsize=(15, 24))
    
    # Create a 2x4 grid of subplots
    gs = fig.add_gridspec(4, 4)
    
    # Create two rows of axes
    ax1 = [fig.add_subplot(gs[0, i]) for i in range(4)]
    ax2 = [fig.add_subplot(gs[1, i]) for i in range(4)]
    ax3 = [fig.add_subplot(gs[2, i]) for i in range(4)]
    ax4 = [fig.add_subplot(gs[3, i]) for i in range(4)]
    
    # Call the plotting functions with their respective axes
    marker_size = 9
    figure_products_greedy_speculative(ax1, marker_size=marker_size)
    figure_products_beam_search_speculative(ax2, marker_size=marker_size)
    figure_retro_beam_search_speculative_bs_1(ax3, marker_size=marker_size)
    figure_retro_beam_search_speculative_nbest_10(ax4, marker_size=marker_size)
    
    # Add overall title
    fig.suptitle(
        """Time it takes for the model to process 500 reactions with different hyperparameters.
        A - product prediction, greedy speculative.
        B - product prediction, speculative beam search.
        C - single-step retrosynthesis, speculative beam search, batch size 1.
        D - single-step retrosynthesis, speculative beam search, 10 best sequences.
        """, 
        size=18)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.3, wspace=0.05)
    plt.savefig("grid_search_summary.png", dpi=300, bbox_inches='tight')