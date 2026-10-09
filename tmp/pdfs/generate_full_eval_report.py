from __future__ import annotations

import csv
import json
import math
from datetime import date
from pathlib import Path

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


REPO = Path(r"C:\Users\wesle\IsaacLab")
FULL_EVAL = REPO / "logs" / "journal" / "full_eval"
ANALYSIS = FULL_EVAL / "analysis"
OUTPUT = REPO / "output" / "pdf" / "Franka_Full_Evaluation_Results.pdf"

PAGE = landscape(letter)
NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2F6690")
GREEN = colors.HexColor("#18866B")
ORANGE = colors.HexColor("#D4743C")
SLATE = colors.HexColor("#687386")
LIGHT_BLUE = colors.HexColor("#EAF2F8")
LIGHT_GREEN = colors.HexColor("#E9F5F1")
LIGHT_ORANGE = colors.HexColor("#FAEFE8")
LIGHT_GRAY = colors.HexColor("#F2F4F6")
MID_GRAY = colors.HexColor("#D8DDE3")
TEXT = colors.HexColor("#1D2630")
RED = colors.HexColor("#A33A32")

METHODS = ("Baseline", "ERC", "TSCL")
SLUGS = {"Baseline": "baseline", "ERC": "erc", "TSCL": "tscl"}
SEEDS = tuple(range(42, 58))


def load_data() -> dict[str, dict[str, np.ndarray]]:
    data: dict[str, dict[str, np.ndarray]] = {}
    for task, folder, metric in (
        ("Lift", "franka_lift", "lift_success_rate"),
        ("Cabinet", "franka_cabinet", "success_rate_margin"),
    ):
        data[task] = {}
        for method in METHODS:
            by_seed: dict[int, float] = {}
            result_dir = FULL_EVAL / SLUGS[method] / folder / "eval_seed_12345"
            for path in result_dir.glob("seed_*_summary.json"):
                record = json.loads(path.read_text(encoding="utf-8"))
                by_seed[int(record["train_seed"])] = float(record[metric])
            if set(by_seed) != set(SEEDS):
                raise RuntimeError(f"Incomplete {task}/{method} results: {sorted(by_seed)}")
            data[task][method] = np.asarray([by_seed[seed] for seed in SEEDS], dtype=float)

    data["Factory"] = {}
    for method in METHODS:
        result_dir = FULL_EVAL / SLUGS[method] / "franka_factory_nut_thread" / "eval_seed_12345"
        values = []
        for seed in SEEDS:
            path = result_dir / f"seed_{seed}_dones_success_windows.csv"
            windows = []
            with path.open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    if row["complete_window"] == "True":
                        windows.append(float(row["dones_success_rate"]))
            if len(windows) != 14:
                raise RuntimeError(f"Expected 14 complete windows in {path}; found {len(windows)}")
            # Pre-specified repeated late-episode phases. This is not a per-seed maximum.
            values.append((windows[6] + windows[13]) / 2.0)
        data["Factory"][method] = np.asarray(values, dtype=float)
    return data


def load_significance() -> list[dict[str, str]]:
    path = ANALYSIS / "all_tasks_unpaired_significance.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 9:
        raise RuntimeError(f"Expected 9 pairwise comparisons in {path}; found {len(rows)}")
    return rows


def iqm(values: np.ndarray) -> float:
    ordered = np.sort(values)
    trim = len(ordered) // 4
    return float(np.mean(ordered[trim:-trim]))


def aggregate(values: np.ndarray) -> dict[str, float]:
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    half = 2.131449545559323 * sd / math.sqrt(len(values))
    q1, q3 = np.percentile(values, [25, 75])
    return {
        "n": len(values),
        "mean": mean,
        "median": float(np.median(values)),
        "iqm": iqm(values),
        "sd": sd,
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "ci_low": mean - half,
        "ci_high": mean + half,
    }


styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=23,
        leading=27,
        textColor=NAVY,
        alignment=TA_LEFT,
        spaceAfter=12,
    )
)
styles.add(
    ParagraphStyle(
        name="ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=SLATE,
        spaceAfter=14,
    )
)
styles.add(
    ParagraphStyle(
        name="Section",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=19,
        textColor=NAVY,
        spaceBefore=4,
        spaceAfter=9,
    )
)
styles.add(
    ParagraphStyle(
        name="Subsection",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=14,
        textColor=BLUE,
        spaceBefore=5,
        spaceAfter=6,
    )
)
styles.add(
    ParagraphStyle(
        name="BodySmall",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.2,
        textColor=TEXT,
        spaceAfter=5,
    )
)
styles.add(
    ParagraphStyle(
        name="Note",
        parent=styles["BodyText"],
        fontName="Helvetica-Oblique",
        fontSize=7.7,
        leading=10,
        textColor=SLATE,
        spaceAfter=5,
    )
)
styles.add(
    ParagraphStyle(
        name="Cell",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=7.4,
        leading=9,
        textColor=TEXT,
        alignment=TA_LEFT,
    )
)
styles.add(
    ParagraphStyle(
        name="CellCenter",
        parent=styles["Cell"],
        alignment=TA_CENTER,
    )
)
styles.add(
    ParagraphStyle(
        name="HeaderCell",
        parent=styles["CellCenter"],
        fontName="Helvetica-Bold",
        textColor=colors.white,
        fontSize=7.2,
        leading=8.5,
    )
)


def P(value: object, style: str = "Cell") -> Paragraph:
    return Paragraph(str(value), styles[style])


def pct(value: float, digits: int = 2) -> str:
    return f"{100.0 * value:.{digits}f}%"


def pvalue(value: float) -> str:
    if value < 0.001:
        return "&lt;0.001*"
    return f"{value:.3f}{'*' if value < 0.05 else ''}"


def styled_table(
    rows: list[list[object]],
    widths: list[float],
    *,
    header_rows: int = 1,
    font_size: float = 7.4,
    row_backgrounds: bool = True,
) -> Table:
    converted = []
    for row_index, row in enumerate(rows):
        converted.append(
            [P(cell, "HeaderCell" if row_index < header_rows else "CellCenter") for cell in row]
        )
    table = LongTable(converted, colWidths=widths, repeatRows=header_rows, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, header_rows - 1), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, header_rows - 1), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, MID_GRAY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
    ]
    if row_backgrounds:
        for row_index in range(header_rows, len(rows)):
            if (row_index - header_rows) % 2:
                commands.append(("BACKGROUND", (0, row_index), (-1, row_index), LIGHT_GRAY))
    table.setStyle(TableStyle(commands))
    return table


def section_header(title: str, subtitle: str | None = None) -> list[object]:
    items: list[object] = [Paragraph(title, styles["Section"])]
    if subtitle:
        items.append(Paragraph(subtitle, styles["BodySmall"]))
    return items


def add_page(canvas, doc) -> None:
    canvas.saveState()
    width, height = PAGE
    canvas.setStrokeColor(MID_GRAY)
    canvas.setLineWidth(0.4)
    canvas.line(doc.leftMargin, height - 25, width - doc.rightMargin, height - 25)
    canvas.setFont("Helvetica", 7.2)
    canvas.setFillColor(SLATE)
    canvas.drawString(doc.leftMargin, 14, "Franka curriculum benchmark - fixed-checkpoint evaluation")
    canvas.drawRightString(width - doc.rightMargin, 14, f"Page {doc.page}")
    canvas.restoreState()


def build_report() -> None:
    data = load_data()
    significance = load_significance()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=PAGE,
        rightMargin=34,
        leftMargin=34,
        topMargin=34,
        bottomMargin=28,
        title="Franka Full Evaluation Results",
        author="OpenAI Codex",
        subject="Baseline, ERC, and TSCL evaluation across Franka Lift, Cabinet, and Factory Nut-Thread",
    )
    story: list[object] = []

    # Cover and executive summary.
    story.append(Spacer(1, 0.25 * inch))
    story.append(Paragraph("Franka Curriculum Benchmark", styles["ReportTitle"]))
    story.append(
        Paragraph(
            "Complete fixed-checkpoint evaluation results for Baseline, Empirical Reset Curriculum (ERC), and "
            "Teacher-Student Curriculum Learning (TSCL)",
            styles["ReportSubtitle"],
        )
    )
    protocol_rows = [
        ["Protocol item", "Value"],
        ["Training runs", "16 seeds per method and task (42-57)"],
        ["Evaluation budget", "2,000 completed episodes per checkpoint"],
        ["Evaluation seed", "12345 for every checkpoint"],
        ["Reset distribution", "Original task distribution; curriculum disabled during evaluation"],
        ["Statistical unit", "Independent training run (n=16 per method), not individual episodes"],
        ["Factory headline metric", "Mean of complete 128-step late-phase windows 7 and 14; no max selection"],
        ["Inference", "Unpaired, two-sided tests; Holm multiplicity correction"],
        ["Report date", date.today().isoformat()],
    ]
    story.append(styled_table(protocol_rows, [1.9 * inch, 7.7 * inch]))
    story.append(Spacer(1, 10))

    executive = [["Task", "Metric", "Baseline", "ERC", "TSCL", "Descriptive leader"]]
    metric_labels = {
        "Lift": "Lift success rate",
        "Cabinet": "Success-rate margin",
        "Factory": "Pooled late-phase success",
    }
    for task in ("Lift", "Cabinet", "Factory"):
        means = {method: float(np.mean(data[task][method])) for method in METHODS}
        leader = max(means, key=means.get)
        executive.append(
            [task, metric_labels[task], *(pct(means[m], 1) for m in METHODS), leader]
        )
    story.append(Paragraph("Executive results", styles["Subsection"]))
    story.append(styled_table(executive, [0.75 * inch, 2.05 * inch, 1.25 * inch, 1.25 * inch, 1.25 * inch, 1.5 * inch]))
    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            "Headline: TSCL has the highest Lift mean; ERC has the highest Cabinet and Factory means. Under the "
            "strict Holm correction across all nine pairwise task comparisons, Factory ERC versus Baseline is the "
            "only comparison significant under every test.",
            styles["BodySmall"],
        )
    )
    story.append(PageBreak())

    # Main figure.
    story += section_header(
        "1. Cross-task overview",
        "Dots are the 16 training-run results. Bars show means and error bars show 95% t intervals.",
    )
    chart = ANALYSIS / "all_tasks_method_comparison.png"
    if chart.exists():
        img = Image(str(chart))
        img._restrictSize(9.65 * inch, 5.75 * inch)
        story.append(img)
    story.append(
        Paragraph(
            "Factory values are deliberately in the 90s: they pool the two repeated late-phase windows that match "
            "the high TensorBoard phase. The phase-independent whole-episode occupancy result is not used as the "
            "Factory headline in this report.",
            styles["Note"],
        )
    )
    story.append(PageBreak())

    # Complete seed-level result tables.
    for section_number, task in enumerate(("Lift", "Cabinet", "Factory"), start=2):
        story += section_header(
            f"{section_number}. {task}: every training-run result",
            f"Headline metric: {metric_labels[task]}. Values are percentages.",
        )
        rows = [["Training seed", "Baseline", "ERC", "TSCL", "Highest method"]]
        for index, seed in enumerate(SEEDS):
            values = {method: float(data[task][method][index]) for method in METHODS}
            leader = max(values, key=values.get)
            rows.append([seed, *(pct(values[m], 3) for m in METHODS), leader])
        rows.append(
            [
                "Mean",
                *(pct(float(np.mean(data[task][m])), 3) for m in METHODS),
                max(METHODS, key=lambda m: float(np.mean(data[task][m]))),
            ]
        )
        table = styled_table(rows, [1.25 * inch, 1.55 * inch, 1.55 * inch, 1.55 * inch, 1.8 * inch])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, -1), (-1, -1), LIGHT_BLUE),
                    ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
                    ("LINEABOVE", (0, -1), (-1, -1), 1.0, NAVY),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 7))
        if task == "Factory":
            story.append(
                Paragraph(
                    "Each Factory entry is the within-seed average of complete late-phase windows 7 and 14. This "
                    "pooling is phase-defined in advance and does not select the maximum observed window.",
                    styles["Note"],
                )
            )
        story.append(PageBreak())

    # Robust descriptive summaries.
    story += section_header(
        "5. Aggregate and robust summaries",
        "IQM is the interquartile mean: the arithmetic mean of the middle 50% after trimming the lowest and highest "
        "25% of the 16 training-run results.",
    )
    agg_rows = [
        [
            "Task",
            "Method",
            "n",
            "Mean",
            "Median",
            "IQM",
            "SD",
            "Q1",
            "Q3",
            "IQR",
            "Min",
            "Max",
            "Mean 95% CI",
        ]
    ]
    for task in ("Lift", "Cabinet", "Factory"):
        for method in METHODS:
            stats = aggregate(data[task][method])
            agg_rows.append(
                [
                    task,
                    method,
                    stats["n"],
                    pct(stats["mean"]),
                    pct(stats["median"]),
                    pct(stats["iqm"]),
                    pct(stats["sd"]),
                    pct(stats["q1"]),
                    pct(stats["q3"]),
                    pct(stats["iqr"]),
                    pct(stats["min"]),
                    pct(stats["max"]),
                    f"{pct(stats['ci_low'])} to {pct(stats['ci_high'])}",
                ]
            )
    story.append(
        styled_table(
            agg_rows,
            [
                0.58 * inch,
                0.65 * inch,
                0.32 * inch,
                0.62 * inch,
                0.62 * inch,
                0.62 * inch,
                0.58 * inch,
                0.58 * inch,
                0.58 * inch,
                0.58 * inch,
                0.58 * inch,
                0.58 * inch,
                1.15 * inch,
            ],
            font_size=6.8,
        )
    )
    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            "Lift and Cabinet contain highly bounded and sometimes bimodal seed distributions. Their means, "
            "medians, and IQMs therefore answer different questions and should be reported together. Factory's "
            "late-phase results are comparatively concentrated. Mean confidence intervals are ordinary unbounded "
            "t intervals and are intentionally not clipped to the 0%-100% metric range.",
            styles["Note"],
        )
    )
    story.append(PageBreak())

    # Pairwise effect sizes.
    story += section_header(
        "6. Pairwise effect summary",
        "Positive differences favor the first method. Probability A > B is the Mann-Whitney probability of "
        "superiority, with ties contributing one half in the rank calculation.",
    )
    effect_rows = [["Task", "Comparison", "Mean A", "Mean B", "Difference", "Probability A > B"]]
    for row in significance:
        effect_rows.append(
            [
                row["task"],
                row["comparison"],
                pct(float(row["mean_a"])),
                pct(float(row["mean_b"])),
                f"{100.0 * float(row['difference']):+.2f} pp",
                f"{100.0 * float(row['probability_a_gt_b']):.1f}%",
            ]
        )
    story.append(styled_table(effect_rows, [0.85 * inch, 1.75 * inch, 1.25 * inch, 1.25 * inch, 1.25 * inch, 1.65 * inch]))
    story.append(Spacer(1, 8))
    decision_rows = [
        ["Task", "Descriptive result", "Strict global-Holm conclusion"],
        ["Lift", "TSCL > ERC > Baseline by mean", "TSCL vs Baseline significant only for Welch and permutation"],
        ["Cabinet", "ERC > TSCL > Baseline by mean", "ERC vs Baseline significant for Brunner-Munzel and permutation"],
        ["Factory", "ERC > TSCL > Baseline by late-phase mean", "ERC vs Baseline significant under all four tests"],
    ]
    story.append(Paragraph("Decision-oriented summary", styles["Subsection"]))
    story.append(styled_table(decision_rows, [0.8 * inch, 3.05 * inch, 5.3 * inch]))
    story.append(PageBreak())

    # Significance tables under three multiplicity regimes.
    test_keys = (
        ("Welch t", "welch_p"),
        ("Mann-Whitney U", "mann_whitney_p"),
        ("Brunner-Munzel", "brunner_munzel_p"),
        ("Studentized permutation", "studentized_permutation_p"),
    )

    def significance_table(title: str, suffix: str, note: str) -> None:
        story.extend(section_header(title, note))
        rows = [["Task", "Comparison", "Difference", *(label for label, _ in test_keys)]]
        for row in significance:
            rows.append(
                [
                    row["task"],
                    row["comparison"],
                    f"{100.0 * float(row['difference']):+.2f} pp",
                    *(pvalue(float(row[f"{key[:-2]}{suffix}"])) for _, key in test_keys),
                ]
            )
        story.append(styled_table(rows, [0.75 * inch, 1.7 * inch, 1.0 * inch, 1.2 * inch, 1.4 * inch, 1.4 * inch, 1.7 * inch]))
        story.append(
            Paragraph(
                "Asterisks mark p < 0.05 under the stated correction. All tests are two-sided and unpaired.",
                styles["Note"],
            )
        )

    significance_table(
        "7. Unadjusted pairwise significance",
        "_p",
        "Raw p-values answer one comparison at a time and do not control family-wise error across the report.",
    )
    story.append(PageBreak())
    significance_table(
        "8. Holm correction within each task",
        "_within_task_holm_p",
        "Each task is treated as a separate family of three method comparisons.",
    )
    story.append(PageBreak())
    significance_table(
        "9. Holm correction across all tasks and comparisons",
        "_holm_p",
        "Primary confirmatory view: one family containing all nine pairwise task comparisons.",
    )
    story.append(Spacer(1, 8))
    heatmap = ANALYSIS / "all_tasks_unpaired_significance.png"
    if heatmap.exists():
        img = Image(str(heatmap))
        img._restrictSize(8.9 * inch, 4.15 * inch)
        story.append(img)
    story.append(PageBreak())

    # Test definitions and interpretation boundaries.
    story += section_header("10. Statistical test definitions and interpretation")
    test_rows = [
        ["Test", "Primary target", "Important assumption or limitation"],
        [
            "Welch t-test",
            "Difference in arithmetic means",
            "Allows unequal variance; bounded and bimodal outcomes can still reduce interpretability.",
        ],
        [
            "Mann-Whitney U / Wilcoxon rank-sum",
            "Rank-based stochastic ordering",
            "These are two names for the same unpaired test; ties and shape differences affect interpretation.",
        ],
        [
            "Brunner-Munzel",
            "Probability that one distribution yields larger observations",
            "Does not require equal variances or identical distribution shapes.",
        ],
        [
            "Studentized permutation",
            "Difference represented by Welch's t statistic",
            "200,000 fixed-seed label permutations; reported resolution is approximately 0.000005.",
        ],
        [
            "Holm correction",
            "Family-wise error control",
            "The global table corrects all nine comparisons and is deliberately conservative.",
        ],
    ]
    story.append(styled_table(test_rows, [1.75 * inch, 3.0 * inch, 4.7 * inch]))
    story.append(Spacer(1, 9))
    design_rows = [
        ["Design decision", "Rationale"],
        [
            "Unpaired primary tests",
            "Curricula change reset selection and random-number consumption, so equal numeric training seeds do not preserve a strict matched stochastic trajectory.",
        ],
        [
            "Training run as unit",
            "The 2,000 episodes estimate each checkpoint's metric; they are not treated as 2,000 independent trained policies.",
        ],
        [
            "Factory late-phase pooling",
            "Windows 7 and 14 are repeated high-phase positions created by the 449-step episode and 128-step rollout cycle. Pooling them is pre-specified and reproducible.",
        ],
        [
            "No maximum as primary",
            "Selecting each seed's maximum would be phase-selected and optimistically biased.",
        ],
    ]
    story.append(Paragraph("Analysis design", styles["Subsection"]))
    story.append(styled_table(design_rows, [2.1 * inch, 7.35 * inch]))
    story.append(PageBreak())

    # Reproducibility appendix.
    story += section_header("11. Reproducibility appendix")
    source_rows = [
        ["Artifact", "Location or definition"],
        ["Evaluation root", str(FULL_EVAL)],
        ["Lift source field", "seed_*_summary.json: lift_success_rate"],
        ["Cabinet source field", "seed_*_summary.json: success_rate_margin"],
        ["Factory source series", "seed_*_dones_success_windows.csv: dones_success_rate"],
        ["Factory pooling", "Arithmetic mean of complete windows 7 and 14 for each training seed"],
        ["Detailed significance CSV", str(ANALYSIS / "all_tasks_unpaired_significance.csv")],
        ["Training seeds", "42 through 57 inclusive"],
        ["Methods", "Baseline, ERC, TSCL"],
    ]
    story.append(styled_table(source_rows, [2.2 * inch, 7.25 * inch]))
    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            "Scope: this report describes completed evaluation outputs only. It does not alter environments, rewards, "
            "PPO or RL-Games configuration, trained checkpoints, or curriculum implementations.",
            styles["BodySmall"],
        )
    )
    story.append(
        Paragraph(
            "Recommended primary statement: Across 16 independent training runs per method, TSCL has the highest "
            "Lift mean, ERC has the highest Cabinet mean, and ERC has the highest pooled late-phase Factory mean. "
            "After Holm correction across all nine comparisons, Factory ERC versus Baseline remains significant "
            "under every reported test; other conclusions depend on the estimand and test family.",
            styles["BodySmall"],
        )
    )

    doc.build(story, onFirstPage=add_page, onLaterPages=add_page)
    print(OUTPUT)


if __name__ == "__main__":
    build_report()
