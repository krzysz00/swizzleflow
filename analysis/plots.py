#!/usr/bin/env python3

import parsing
import extraction
import utils

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys

SWINV_TIMEOUTS = ['l3/trove-rcr-3', 'l3/trove-rcr-5', 'l3/trove-rcr-7', 'l3/2d-stencil-7', 'l3/trove-cr_sum-7']

def process_data(swizzleflow_raw, loads_raw, swinv_times):
    swizzleflow_info = extraction.humanize_names(swizzleflow_raw)
    loads_info = extraction.humanize_names(loads_raw)
    swinv_info = extraction.extract_swizzle_inventor_times(swinv_times)

    times = extraction.get_times(swizzleflow_info).drop(columns='key')
    load_times = extraction.get_times(loads_info).drop(columns='key')
    split_times = times.groupby(["spec", "category"]).sum()['time'].unstack(fill_value=0.0)

    split_times =\
        split_times.rename({'build': 'Mat. creation', 'mul': 'Mat. multiply',
                            'load': 'Mat. reuse', 'search': 'Search'}, axis=1)

    total = split_times.sum(axis=1)
    total.name = "Swizzleflow"

    load_times = load_times[load_times['category'] != 'search'].groupby(['spec'])['time'].sum()
    load_times = pd.DataFrame({"Reloading all": load_times})

    swinv_times_for_join = swinv_info.set_index('spec')['time']
    swinv_times_for_join.name = "Swizzle Inventor"
    for c in SWINV_TIMEOUTS:
        if c not in swinv_times_for_join:
            swinv_times_for_join[c] = np.NAN

    comparison = pd.concat([total, swinv_times_for_join], axis=1, join='inner')

    details = split_times.join(load_times)

    details["Total synthesis"] = total
    return details, comparison

def to_plot_df(df):
    plot_df = df.copy()
    plot_df.index.name = "spec"
    plot_df = extraction.split_spec(plot_df.reset_index())
    plot_df.insert(0, "Level", plot_df["level"].map({1: 'R', 2: 'FC', 3: 'F'}))
    plot_df.insert(0, "Problem", plot_df["problem"].map(utils.PROBLEM_NAMES))
    plot_df.drop(columns=["level", "problem", "spec"], inplace=True)
    return plot_df

PLOT_COLS = 4
def plot(df, use_log, title, filename):
    groups_len = len(df.groupby('Problem'))
    fig, axs = plt.subplots((groups_len + PLOT_COLS - 1) // PLOT_COLS, PLOT_COLS,
                       figsize=(17, 11))
    for idx, group in enumerate(df.groupby('Problem')):
        name, data = group
        ax = axs[idx // PLOT_COLS][idx % PLOT_COLS]
        data = data.set_index('Level')
        # Fix the order of the level groups, remove if it somehow reverses again
        new_index = [x for x in ['F', 'FC', 'R'] if x in data.index]
        data = data.reindex(new_index)
        data = data.fillna(35 * 60) # Plot timeouts somehow
        # The order of the bars in backwards from how I'd like it, fix that
        data = data[data.columns.to_list()[::-1]]
        data.plot(kind='barh', ax=ax, legend=False, logx=use_log)
        ax.set_ylabel('')
        ax.set_title(name)
    for ax in axs[-1]:
        ax.set_xlabel("Time (s)" + (" (log scale)" if use_log else ""))
    handles, labels = axs[-1][0].get_legend_handles_labels()
    handles.reverse()
    labels.reverse()
    fig.suptitle(title)
    fig.legend(handles, labels, loc='upper right')
    fig.tight_layout()
    # Magic formula for providing space for the legend
    fig.subplots_adjust(top=0.875, right=0.925)
    fig.savefig(filename)
    plt.close(fig)

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: [swizzleflow times] [swizzleflow times loads only] [swizzle inventor times] [prefix]")
        sys.exit(1)
    _, swflow_file, swflow_loads_file, swinv_file, prefix = sys.argv
    swflow_data = parsing.parse_file(swflow_file)
    swflow_load_data = parsing.parse_file(swflow_loads_file)
    swinv_data = parsing.parse_swizzle_inventor_file(swinv_file)

    details, comparison = process_data(swflow_data, swflow_load_data, swinv_data)

    details = to_plot_df(details)
    comparison = to_plot_df(comparison)

    details.to_csv(f"{prefix}-details.csv", index=False)
    comparison.to_csv(f"{prefix}-comparison.csv", index=False)

    plot(details, False, "Synthesis time breakdown", f"{prefix}-details.pdf")
    plot(details, True, "Synthesis time breakdown (log)", f"{prefix}-details-log.pdf")
    plot(comparison, False, "Synthesis time comparison", f"{prefix}-comparison.pdf")
    plot(comparison, True, "Synthesis time comparison (log)", f"{prefix}-comparison-log.pdf")
