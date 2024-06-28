import argparse
import pathlib
import sqlite3

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("database", help="Path to the database with the results.")
parser.add_argument(
    "--out", help="If set path where the plots should be stored", type=pathlib.Path
)
args = parser.parse_args()


PREFIX = pathlib.Path(args.database).parent.stem
con = sqlite3.connect(args.database)
data = pd.read_sql(
    """
    SELECT * FROM query_result
    JOIN experiment ON query_result.experiment_id == experiment.id
""",
    con,
)
# Validation results
val = pd.read_sql(
    """
    SELECT embedding_model, mean_absolute_error, mean_squared_error, R2, iteration, sampler FROM validation_summary
    JOIN experiment on validation_summary.experiment_id == experiment.id
    WHERE sampler != "random"
""",
    con,
)

### Average affinity over all iterations
barplot = sns.barplot(
    data.groupby(["embedding_model", "sampler"])["affinity"].mean().reset_index(),
    y="affinity",
    x="embedding_model",
    hue="sampler",
)

if args.out is not None:
    barplot.get_figure().savefig(args.out / f"{PREFIX}_avg_affinity.png")
plt.clf()
### Affinity progression of the iterations
lineplot = sns.lineplot(
    data[data["iteration"] >= 0],
    y="affinity",
    x="iteration",
    hue="embedding_model",
    style="sampler",
)
lineplot.set_xticks(
    ticks=range(data["iteration"].max()), labels=range(1, data["iteration"].max() + 1)
)

if args.out is not None:
    lineplot.get_figure().savefig(args.out / f"{PREFIX}_progression.png")
plt.clf()
### Percentage of top 100 molecules found
pool = pd.read_csv(data["data_path"].unique()[0])
top100 = pool.loc[pool["target"].nsmallest(100).index]
top100percentage = (
    data.groupby(["embedding_model", "sampler", "iteration"])
    .apply(lambda x: x["molecule"].isin(top100["smiles"]).sum())
    .rename("Percentage of Top 100 scores found")
    .groupby(level=[0, 1])
    .cumsum()
)
mols_screened = (
    data.groupby(["embedding_model", "sampler", "iteration"])
    .apply(len)
    .groupby(level=[0, 1])
    .cumsum()
    .rename("Molecules screened")
)
combined = pd.concat([top100percentage, mols_screened], axis=1)
top100plot = sns.lineplot(
    combined,
    x="Molecules screened",
    y="Percentage of Top 100 scores found",
    hue="embedding_model",
    style="sampler",
    markers=True,
)
if args.out is not None:
    top100plot.get_figure().savefig(args.out / f"{PREFIX}_top100.png")
plt.clf()

### Validation results
val_long = pd.melt(
    val,
    id_vars=["embedding_model", "sampler", "iteration"],
    value_vars=["mean_absolute_error", "mean_squared_error", "R2"],
    var_name="metric",
)
val_plot = sns.relplot(
    val_long,
    x="iteration",
    y="value",
    hue="embedding_model",
    col="metric",
    style="sampler",
    markers=True,
    kind="line",
)


if args.out is not None:
    val_plot.figure.savefig(args.out / f"{PREFIX}_validation.png")
plt.clf()
