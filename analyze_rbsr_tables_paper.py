#!/usr/bin/env python3
"""
analyze_rbsr_tables_paper.py

Generate paper-aligned tables from the Negentropy/AELMDB benchmark CSV.
- Scenario-family parameters use set-range notation in LaTeX columns.
- Absolute results default to the paper metrics:
    T_prep, T_rec, S_used, and RSS_after when available.
- Relative results default to ratios versus AELMDBSlice, including RSS when
  available.
- Family names are normalized to the paper style, e.g. base_dense_i.
- Backend order matches the paper:
    Vector, BTreeLMDB, NoWndAELMDB, AELMDB.

Usage:
    python analyze_rbsr_tables_paper.py results.csv
    python analyze_rbsr_tables_paper.py results.csv --outdir out_tables
    python analyze_rbsr_tables_paper.py results.csv --no-rss
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


BACKEND_ORDER = [
    "Vector",
    "BTreeLMDB",
    "AELMDBSlice",
    "NoAggWindowSliceAELMDB",
]

FAMILY_ORDER = [
    "baseline_dense",
    "baseline_sparse",
    "scale_dense",
    "scale_sparse",
    "stress_bigdiff",
    "stress_bigdiff_dyn",
]

FAMILY_LATEX = {
    "baseline_dense": r"\texttt{base\_dense\(_i\)}",
    "baseline_sparse": r"\texttt{base\_sparse\(_i\)}",
    "scale_dense": r"\texttt{scale\_dense\(_i\)}",
    "scale_sparse": r"\texttt{scale\_sparse\(_i\)}",
    "stress_bigdiff": r"\texttt{stress\(_i\)}",
    "stress_bigdiff_dyn": r"\texttt{stress\_dyn\(_i\)}",
}

BACKEND_LATEX = {
    "Vector": r"\texttt{Vector}",
    "BTreeLMDB": r"\texttt{BTreeLMDB}",
    "AELMDBSlice": r"\texttt{AELMDB}",
    "NoAggWindowSliceAELMDB": r"\texttt{NoWndAELMDB}",
}


# ----------------------------- formatting helpers -----------------------------


def latex_escape(x: object) -> str:
    s = str(x)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for a, b in repl.items():
        s = s.replace(a, b)
    return s


def fmt_float(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x):.{digits}f}"


def fmt_ratio(x: float, digits: int = 2) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x):.{digits}f}x"


def render_text_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    rows = [[str(v) for v in row] for row in df.itertuples(index=False, name=None)]
    widths = [len(str(c)) for c in cols]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(items: Iterable[str]) -> str:
        return " | ".join(str(item).ljust(widths[i]) for i, item in enumerate(items))

    sep = "-+-".join("-" * w for w in widths)
    out = [fmt_row(cols), sep]
    out.extend(fmt_row(r) for r in rows)
    return "\n".join(out)


def render_latex_table(df: pd.DataFrame, caption: str, label: str, colspec: str, size_cmd: str = r"\small") -> str:
    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    if size_cmd:
        lines.append(size_cmd)
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\hline")
    lines.append(" & ".join(str(c) for c in df.columns) + r" \\")
    lines.append(r"\hline")
    for row in df.itertuples(index=False, name=None):
        lines.append(" & ".join(str(v) for v in row) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def blank_repeated_first_col(df: pd.DataFrame, first_col: str) -> pd.DataFrame:
    out = df.copy()
    last = None
    vals = []
    for v in out[first_col]:
        if v == last:
            vals.append("")
        else:
            vals.append(v)
            last = v
    out[first_col] = vals
    return out


# ----------------------------- math helpers ----------------------------------


def geometric_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if pd.notna(v) and float(v) > 0.0]
    if not vals:
        return float("nan")
    return float(math.exp(sum(math.log(v) for v in vals) / len(vals)))


def describe_instances(indices: List[int], magnitude: int) -> str:
    idx = sorted(set(int(i) for i in indices))
    if idx == [0]:
        return "1"
    if idx == list(range(1, magnitude + 1)):
        return r"$i=1,\ldots,%d$" % magnitude
    if idx == list(range(2, magnitude + 1, 2)):
        return r"even $i=2,\ldots,%d$" % magnitude
    if idx == list(range(1, max(idx) + 1)):
        return rf"$i=1,\ldots,{max(idx)}$"
    if idx == list(range(2, max(idx) + 1, 2)):
        return rf"even $i=2,\ldots,{max(idx)}$"
    if len(idx) <= 6:
        return "$\\{" + ",".join(str(i) for i in idx) + r"\\}$"
    return str(len(idx))


def infer_formula(values: Iterable[int], indices: Iterable[int], magnitude: int) -> str:
    vals = [int(v) for v in values]
    idxs = [int(i) for i in indices]
    uniq = sorted(set(vals))

    if len(uniq) == 1:
        v = uniq[0]
        if magnitude > 0 and v % magnitude == 0 and v != 0:
            return rf"${v // magnitude}M$"
        return rf"${v}$"

    pos_pairs = [(i, v) for i, v in zip(idxs, vals) if i > 0]

    if pos_pairs and all(v % i == 0 for i, v in pos_pairs):
        ks = {v // i for i, v in pos_pairs}
        if len(ks) == 1:
            return rf"${next(iter(ks))}i$"

    if pos_pairs and all(v % (i * i) == 0 for i, v in pos_pairs):
        ks = {v // (i * i) for i, v in pos_pairs}
        if len(ks) == 1:
            return rf"${next(iter(ks))}i^2$"

    if len(uniq) <= 4:
        return "$\\{" + ",".join(str(v) for v in uniq) + r"\\}$"

    return rf"${uniq[0]}\ldots{uniq[-1]}$"


# ----------------------------- input validation ------------------------------


def has_rss_columns(df: pd.DataFrame) -> bool:
    return "rss_kb_after" in df.columns


def check_required_columns(df: pd.DataFrame, require_rss: bool = False) -> None:
    required = {
        "scenario",
        "scenario_family",
        "scenario_i",
        "magnitude",
        "backend",
        "step_in_slice",
        "n_common_in_slice",
        "n_a_only_in_slice",
        "n_b_only_in_slice",
        "n_common_outside_before",
        "n_common_outside_after",
        "n_a_only_outside_before",
        "n_a_only_outside_after",
        "n_b_only_outside_before",
        "n_b_only_outside_after",
        "prep_total_ms",
        "build_ms",
        "reconcile_ms",
        "A_used_bytes_est",
    }
    if require_rss:
        required.add("rss_kb_after")
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")


def unique_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    shape_cols = [
        "scenario",
        "scenario_family",
        "scenario_i",
        "magnitude",
        "step_in_slice",
        "n_common_in_slice",
        "n_a_only_in_slice",
        "n_b_only_in_slice",
        "n_common_outside_before",
        "n_common_outside_after",
        "n_a_only_outside_before",
        "n_a_only_outside_after",
        "n_b_only_outside_before",
        "n_b_only_outside_after",
    ]
    out = df[shape_cols].drop_duplicates().copy()
    out = out.sort_values(["scenario_family", "scenario_i", "scenario"])
    return out


# ----------------------------- naming helpers --------------------------------


def base_family_name(family: str) -> str:
    if family in FAMILY_ORDER:
        return family
    for base in sorted(FAMILY_ORDER, key=len, reverse=True):
        if family.startswith(base + "_"):
            return base
    return family

def family_latex_name(family: str) -> str:
    return FAMILY_LATEX.get(base_family_name(family), rf"\texttt{{{latex_escape(family)}}}")


def backend_latex_name(backend: str) -> str:
    return BACKEND_LATEX.get(backend, rf"\texttt{{{latex_escape(backend)}}}")


# ----------------------------- table builders --------------------------------


def make_scenario_table(df: pd.DataFrame) -> pd.DataFrame:
    sc = unique_scenarios(df).copy()
    sc["base_family"] = sc["scenario_family"].map(base_family_name)

    if not (sc["n_a_only_in_slice"] == sc["n_b_only_in_slice"]).all():
        raise ValueError("CSV is not symmetric in-slice: n_a_only_in_slice != n_b_only_in_slice")
    if not (
        (sc["n_a_only_outside_before"] == sc["n_b_only_outside_before"])
        & (sc["n_a_only_outside_after"] == sc["n_b_only_outside_after"])
    ).all():
        raise ValueError("CSV is not symmetric outside-slice between A-only and B-only counts")

    rows = []
    for fam, g in sc.groupby("base_family", sort=False):
        magnitude = int(g["magnitude"].iloc[0])
        idxs = g["scenario_i"].tolist()
        common_out = (g["n_common_outside_before"] + g["n_common_outside_after"]).tolist()
        uniq_out = (g["n_a_only_outside_before"] + g["n_a_only_outside_after"]).tolist()
        rows.append(
            {
                "Family": family_latex_name(fam),
                "Instances": describe_instances(idxs, magnitude),
                "step": infer_formula(g["step_in_slice"], idxs, magnitude),
                r"$|X_{\mathrm{in}}\cap Y_{\mathrm{in}}|$": infer_formula(g["n_common_in_slice"], idxs, magnitude),
                r"$|X_{\mathrm{in}}\setminus Y_{\mathrm{in}}|$": infer_formula(g["n_a_only_in_slice"], idxs, magnitude),
                r"$|X_{\mathrm{out}}\cap Y_{\mathrm{out}}|$": infer_formula(common_out, idxs, magnitude),
                r"$|X_{\mathrm{out}}\setminus Y_{\mathrm{out}}|$": infer_formula(uniq_out, idxs, magnitude),
                "_sort_key": FAMILY_ORDER.index(fam) if fam in FAMILY_ORDER else 999,
            }
        )
    out = pd.DataFrame(rows).sort_values("_sort_key").drop(columns=["_sort_key"]).reset_index(drop=True)
    return out


def make_absolute_results_table(df: pd.DataFrame, include_rss: bool = True) -> pd.DataFrame:
    work = df.copy()
    work["base_family"] = work["scenario_family"].map(base_family_name)

    agg = {
        "prep_ms": ("prep_total_ms", "mean"),
        "reconcile_ms": ("reconcile_ms", "mean"),
        "db_mib": ("A_used_bytes_est", lambda s: float(np.mean(s)) / (1024.0 * 1024.0)),
    }
    if include_rss and has_rss_columns(work):
        agg["rss_after_mib"] = ("rss_kb_after", lambda s: float(np.mean(s)) / 1024.0)

    g = work.groupby(["base_family", "backend"], as_index=False).agg(**agg)
    g["base_family"] = pd.Categorical(g["base_family"], categories=FAMILY_ORDER, ordered=True)
    g["backend"] = pd.Categorical(g["backend"], categories=BACKEND_ORDER, ordered=True)
    g = g.sort_values(["base_family", "backend"]).reset_index(drop=True)

    data = {
        "Family": g["base_family"].map(family_latex_name),
        "Backend": g["backend"].map(backend_latex_name),
        r"$T_{\mathrm{prep}}$ (ms)": g["prep_ms"].map(fmt_float),
        r"$T_{\mathrm{rec}}$ (ms)": g["reconcile_ms"].map(fmt_float),
        r"$S_{\mathrm{used}}$ (MiB)": g["db_mib"].map(fmt_float),
    }
    if include_rss and "rss_after_mib" in g.columns:
        data[r"$\mathrm{RSS}_{\mathrm{after}}$ (MiB)"] = g["rss_after_mib"].map(fmt_float)

    out = pd.DataFrame(data)
    return blank_repeated_first_col(out, "Family")


def make_relative_results_table(df: pd.DataFrame, include_rss: bool = True) -> pd.DataFrame:
    cols = [
        "scenario_family",
        "scenario",
        "backend",
        "prep_total_ms",
        "reconcile_ms",
        "A_used_bytes_est",
    ]
    if include_rss and has_rss_columns(df):
        cols.append("rss_kb_after")

    base = df[cols].drop_duplicates().copy()
    base["base_family"] = base["scenario_family"].map(base_family_name)

    value_cols = ["prep_total_ms", "reconcile_ms", "A_used_bytes_est"]
    if include_rss and has_rss_columns(df):
        value_cols.append("rss_kb_after")

    pivot = base.pivot_table(
        index=["base_family", "scenario"],
        columns="backend",
        values=value_cols,
        aggfunc="mean",
    )

    rows = []
    families_present = [f for f in FAMILY_ORDER if f in base["base_family"].unique()]
    for fam in families_present:
        fam_slice = pivot.loc[fam]

        def ratio_gmean(metric: str, backend: str, ref: str = "AELMDBSlice") -> float:
            num = fam_slice[(metric, backend)]
            den = fam_slice[(metric, ref)]
            return geometric_mean(num / den)

        row = {
            "Family": family_latex_name(fam),
            r"\texttt{BTree}/\texttt{AELMDB} rec": fmt_ratio(ratio_gmean("reconcile_ms", "BTreeLMDB")),
            r"\texttt{NoAgg}/\texttt{AELMDB} rec": fmt_ratio(ratio_gmean("reconcile_ms", "NoAggWindowSliceAELMDB")),
            r"\texttt{Vector}/\texttt{AELMDB} rec": fmt_ratio(ratio_gmean("reconcile_ms", "Vector")),
            r"\texttt{BTree}/\texttt{AELMDB} prep": fmt_ratio(ratio_gmean("prep_total_ms", "BTreeLMDB")),
            r"\texttt{BTree}/\texttt{AELMDB} size": fmt_ratio(ratio_gmean("A_used_bytes_est", "BTreeLMDB")),
        }
        if include_rss and has_rss_columns(df):
            row[r"\texttt{BTree}/\texttt{AELMDB} RSS"] = fmt_ratio(ratio_gmean("rss_kb_after", "BTreeLMDB"))
        rows.append(row)

    return pd.DataFrame(rows)


def make_build_importance_table(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["base_family"] = work["scenario_family"].map(base_family_name)
    g = (
        work.groupby(["base_family", "backend"], as_index=False)
        .agg(build_ms=("build_ms", "mean"), reconcile_ms=("reconcile_ms", "mean"))
        .copy()
    )
    g["build_over_reconcile_pct"] = 100.0 * g["build_ms"] / g["reconcile_ms"]
    g["base_family"] = pd.Categorical(g["base_family"], categories=FAMILY_ORDER, ordered=True)
    g["backend"] = pd.Categorical(g["backend"], categories=BACKEND_ORDER, ordered=True)
    g = g.sort_values(["base_family", "backend"]).reset_index(drop=True)

    out = pd.DataFrame(
        {
            "Family": g["base_family"].map(family_latex_name),
            "Backend": g["backend"].map(backend_latex_name),
            "Build/Reconcile (\\%)": g["build_over_reconcile_pct"].map(lambda x: fmt_float(x, 1)),
        }
    )
    return blank_repeated_first_col(out, "Family")


# ----------------------------- captions/labels -------------------------------


def scenario_caption() -> str:
    return (
        "Scenario-family parameters for the benchmark suite. The slice interval is "
        r"$\range{\ell}{u}$. The last two columns aggregate the outside-of-slice "
        "population over both sides of the slice."
    )


def absolute_caption(include_rss: bool) -> str:
    if include_rss:
        return (
            "Absolute results on the benchmark suite. Values are arithmetic means over the scenarios "
            "in each family. Here $T_{\\mathrm{prep}}$ is the one-time preparation cost, "
            "$T_{\\mathrm{rec}}$ is the full reconciliation cost including HAVE/NEED materialization, "
            "$S_{\\mathrm{used}}$ is the estimated used size, and "
            "$\\mathrm{RSS}_{\\mathrm{after}}$ is the resident set size sampled after the isolated bench run."
        )
    return (
        "Absolute results on the benchmark suite. Values are arithmetic means over the scenarios in each family. "
        "Here $T_{\\mathrm{prep}}$ is the one-time preparation cost, $T_{\\mathrm{rec}}$ is the full reconciliation cost "
        "including HAVE/NEED materialization, and $S_{\\mathrm{used}}$ is the estimated used size."
    )


def relative_caption(include_rss: bool) -> str:
    end = (
        " Values larger than $1$ indicate that the numerator backend is slower or larger than "
        "\texttt{AELMDBSlice}; values smaller than $1$ indicate the opposite."
    )
    if include_rss:
        return (
            "Relative results versus \texttt{AELMDBSlice}. Each entry is the geometric mean, over the scenarios "
            "of the family, of the corresponding per-scenario ratio." + end
        )
    return (
        "Relative results versus \texttt{AELMDBSlice}. Each entry is the geometric mean, over the scenarios "
        "of the family, of the corresponding per-scenario ratio." + end
    )


# ----------------------------- main ------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path, help="Path to benchmark CSV")
    ap.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Optional directory where .txt and .tex files will be written",
    )
    ap.add_argument(
        "--no-rss",
        action="store_true",
        help="Do not include RSS in the main absolute/relative tables, even if available.",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    include_rss = (not args.no_rss) and has_rss_columns(df)
    check_required_columns(df, require_rss=False)

    scenario_tbl = make_scenario_table(df)
    absolute_tbl = make_absolute_results_table(df, include_rss=include_rss)
    relative_tbl = make_relative_results_table(df, include_rss=include_rss)
    build_tbl = make_build_importance_table(df)

    latex_scenario = render_latex_table(
        scenario_tbl,
        caption=scenario_caption(),
        label="tab:scenario-families",
        colspec="|l|c|c|c|c|c|c|",
        size_cmd=r"\small",
    )
    latex_absolute = render_latex_table(
        absolute_tbl,
        caption=absolute_caption(include_rss),
        label="tab:absolute-results",
        colspec="|l|l|r|r|r|" + ("r|" if include_rss else ""),
        size_cmd=r"\scriptsize",
    )
    latex_relative = render_latex_table(
        relative_tbl,
        caption=relative_caption(include_rss),
        label="tab:relative-results",
        colspec="|l|r|r|r|r|r|" + ("r|" if include_rss else ""),
        size_cmd=r"\small",
    )
    latex_build = render_latex_table(
        build_tbl,
        caption="Build-time importance relative to reconciliation.",
        label="tab:build-importance",
        colspec="|l|l|r|",
        size_cmd=r"\small",
    )

    print("\n=== Scenario family parameters (text) ===\n")
    print(render_text_table(scenario_tbl))

    print("\n=== Absolute results (text) ===\n")
    print(render_text_table(absolute_tbl))

    print("\n=== Relative results vs AELMDBSlice (text) ===\n")
    print(render_text_table(relative_tbl))

    print("\n=== Build importance (text) ===\n")
    print(render_text_table(build_tbl))

    print("\n=== Scenario family parameters (LaTeX) ===\n")
    print(latex_scenario)

    print("\n=== Absolute results (LaTeX) ===\n")
    print(latex_absolute)

    print("\n=== Relative results vs AELMDBSlice (LaTeX) ===\n")
    print(latex_relative)

    print("\n=== Build importance (LaTeX) ===\n")
    print(latex_build)

    if args.outdir is not None:
        args.outdir.mkdir(parents=True, exist_ok=True)
        (args.outdir / "scenario_family_parameters.txt").write_text(render_text_table(scenario_tbl) + "\n", encoding="utf-8")
        (args.outdir / "scenario_family_parameters.tex").write_text(latex_scenario + "\n", encoding="utf-8")
        (args.outdir / "results_absolute.txt").write_text(render_text_table(absolute_tbl) + "\n", encoding="utf-8")
        (args.outdir / "results_absolute.tex").write_text(latex_absolute + "\n", encoding="utf-8")
        (args.outdir / "results_relative.txt").write_text(render_text_table(relative_tbl) + "\n", encoding="utf-8")
        (args.outdir / "results_relative.tex").write_text(latex_relative + "\n", encoding="utf-8")
        (args.outdir / "build_importance.txt").write_text(render_text_table(build_tbl) + "\n", encoding="utf-8")
        (args.outdir / "build_importance.tex").write_text(latex_build + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
