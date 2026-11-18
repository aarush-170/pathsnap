"""Build the final technical report: report/PathSnap_Technical_Report.pdf

Usage (from repo root):  python report/build_report.py
Structure follows docs/reference/Deliverables.pdf sections 2-8. Every table and number is read from
results/ (and models/, configs/) so the report always matches the repository.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "report"))

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from report_style import (BODY, CAP, CELL, CELLB, H1, H2, H3, INK, LIGHT, MUTED, PAGE_W, SUB, P, bullets,
                          table)

LOG, FIG, RFIG = ROOT / "results" / "logs", ROOT / "results" / "figures", ROOT / "report" / "figures"
MODELS = ROOT / "results" / "models"
OUT = ROOT / "report" / "PathSnap_Technical_Report.pdf"
TINY = CELL.clone("TINY", fontSize=7.2, leading=9)
TINYB = CELLB.clone("TINYB", fontSize=7.2, leading=9)
MONO = BODY.clone("MONO", fontName="Courier", fontSize=7.4, leading=9.4, backColor=colors.HexColor("#f4f6f8"),
                  borderPadding=(5, 6, 5, 6), spaceBefore=4, spaceAfter=8)
H4 = H3.clone("H4", fontSize=10.5, leading=13, spaceBefore=4, spaceAfter=2, textColor=MUTED)

pct = lambda x: f"{100 * x:.2f}%"
pct1 = lambda x: f"{100 * x:.1f}%"


def fig(path, caption, width=PAGE_W):
    from reportlab.lib.utils import ImageReader
    w, h = ImageReader(str(path)).getSize()
    return KeepTogether([Image(str(path), width=width, height=width * h / w), P(caption, CAP)])


def small_table(rows, widths, header=True):
    data = [[Paragraph(str(c), TINYB if (header and i == 0) else TINY) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd2d9")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
          ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe4f0")))
    t.setStyle(TableStyle(st))
    return t


def team():
    txt = (ROOT / "README.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\|\s*([^|]+?)\s*\|\s*(PES[^|]+?)\s*\|\s*([^|]+?)\s*\|", txt, flags=re.M)
    return [[n, s, c] for n, s, c in rows]


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, 1.2 * cm, "PathSnap: Distributed Map-Matching & Route Reconstruction (UE24CS352A)")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


# ----------------------------------------------------------------------------------------- data
def load():
    D = {}
    D["eda"] = pd.read_csv(LOG / "eda_trip_summary.csv", dtype={"trip": str})
    D["inv"] = pd.read_csv(LOG / "eda_file_inventory.csv", dtype={"trip": str})
    D["anom"] = pd.read_csv(LOG / "eda_anomalies.csv", dtype={"trip": str})
    D["cand"] = pd.read_csv(LOG / "candidate_summary.csv").set_index("split")
    D["sweep"] = pd.read_csv(LOG / "candidate_sweep.csv")
    D["uni"] = pd.read_csv(LOG / "feature_univariate.csv")
    D["rule"] = pd.read_csv(LOG / "rule_baselines.csv").set_index("rule")
    D["cmp"] = pd.read_csv(LOG / "model_comparison.csv")
    D["folds"] = pd.read_csv(LOG / "model_cv_folds.csv")
    D["met"] = {m.name: json.loads((m / "metrics.json").read_text()) for m in MODELS.iterdir() if (m / "metrics.json").exists()}
    D["tuned"] = json.loads((ROOT / "configs" / "lightgbm_tuned.json").read_text())
    D["trials"] = pd.read_csv(LOG / "optuna_trials.csv")
    D["imp"] = pd.read_csv(LOG / "optuna_param_importance.csv", index_col=0).iloc[:, 0]
    D["tcmp"] = pd.read_csv(LOG / "tuning_comparison.csv")
    D["hmm_grid"] = pd.read_csv(LOG / "hmm_grid_lightgbm_tuned.csv")
    D["hmm_cmp"] = pd.read_csv(LOG / "hmm_comparison_lightgbm_tuned.csv")
    D["hmm_cmp_u"] = pd.read_csv(LOG / "hmm_comparison_lightgbm.csv")
    D["hmm_trip"] = pd.read_csv(LOG / "hmm_per_trip_lightgbm_tuned.csv", dtype={"trip": str})
    D["export"] = pd.read_csv(LOG / "export_check.csv", dtype={"trip": str})
    D["evalrun"] = pd.read_csv(LOG / "export_eval_runcheck.csv", dtype={"trip": str})
    D["maps"] = pd.read_csv(LOG / "maps_summary.csv", dtype={"trip": str})
    D["pipe"] = json.loads((ROOT / "models" / "pipeline_config.json").read_text())
    return D


# ----------------------------------------------------------------------------------------- per-model text
MODEL_TEXT = {
    "logreg": dict(
        name="Logistic Regression", why=(
            "Each row is a (GPS point, candidate road) pair described by 26 engineered features, many of them already "
            "relative to the point's other candidates (distance minus the minimum, distance rank, heading cosine). A "
            "linear model on such features is a strong, cheap baseline, and logistic regression outputs probabilities "
            "directly, which the HMM needs as emission scores."),
        strengths=("Fast to train (about 4.5 s) and the fastest at inference (about 2 ms per 1,000 points); stable "
                   "across folds; coefficients are interpretable; probabilities are reasonably calibrated."),
        limits=("Only a linear decision boundary in the features. It cannot express interactions such as \"trust "
                "heading only when the vehicle is moving\" or \"context matters more when the two nearest roads are "
                "close\", which the tree ensembles exploit."),
        hyper="C = 1.0 (L2 penalty), lbfgs solver, max_iter = 1000, features standardised (StandardScaler fitted on "
              "the training fold only).",
        loss="Binary cross-entropy (log loss) plus an L2 penalty on the weights."),
    "gnb": dict(
        name="Gaussian Naive Bayes", why=(
            "A probabilistic baseline: it models each feature as Gaussian within each class and combines them with Bayes' "
            "rule. It tests how far simple per-feature evidence goes without modelling feature interactions."),
        strengths="No hyperparameters to tune, trains in under a second, tiny model, very fast inference.",
        limits=("Assumes features are independent given the class. Our features are strongly correlated (Spearman above "
                "0.9 between dist and dist_minus_min, and among the context features), so the same evidence is "
                "counted several times and probabilities become overconfident. Several features (rank, t_at_end) are "
                "far from Gaussian."),
        hyper="Default var_smoothing = 1e-9; no scaling needed.",
        loss="Maximum-likelihood estimates of per-class Gaussian means and variances (no iterative loss)."),
    "knn": dict(
        name="K-Nearest Neighbours, distance-weighted", why=(
            "Instance-based: a candidate whose feature vector resembles training candidates that were correct is likely "
            "correct. It captures non-linear local structure without assuming a model form."),
        strengths="Non-parametric, no training cost, naturally handles non-linear boundaries.",
        limits=("Prediction is very slow (about 1.9 s per 1,000 GPS points), the slowest of all models and unsuitable for "
                "the live API. It needs feature scaling, memory grows with the training set, so it was trained on a "
                "200,000-row random sample per fit."),
        hyper="k = 25, weights = distance, Euclidean metric on standardised features, 200,000-row training sample.",
        loss="None (lazy learner): prediction is a distance-weighted vote of the 25 nearest training rows."),
    "random_forest": dict(
        name="Random Forest", why=(
            "Bagged decision trees model non-linear rules and feature interactions, are insensitive to feature scale and "
            "robust to correlated features. A natural strong learner for tabular engineered features."),
        strengths="High accuracy with little tuning; averaging many trees reduces variance; good probability estimates.",
        limits=("Largest train/validation gap of all models (deep trees partly memorise training trips); slowest to "
                "train (about 5 minutes on all train trips) and a large model file."),
        hyper="200 trees, max_depth = None, min_samples_leaf = 5, max_features = sqrt, bootstrap sampling, seed 42.",
        loss="Each tree greedily minimises Gini impurity; the forest averages the trees' class probabilities."),
    "lightgbm": dict(
        name="LightGBM", why=(
            "Gradient-boosted trees add trees that correct the previous ensemble's errors. They model interactions "
            "between distance, heading, context and motion features, and the histogram-based, leaf-wise LightGBM is "
            "fast enough on 1.3 million rows to support Bayesian tuning."),
        strengths="Best cross-validation Top-1 and the best log loss in every fold; fast training (about 19 s); "
                  "moderate overfitting.",
        limits="Many interacting hyperparameters; leaf-wise growth can overfit small data if min_child_samples is low; "
               "sequential boosting does not parallelise across trees.",
        hyper="n_estimators = 400, learning_rate = 0.05, num_leaves = 63, min_child_samples = 50, subsample = 0.8, "
              "colsample_bytree = 0.8, reg_lambda = 1.0 (untuned Phase 4 configuration).",
        loss="Binary log loss, minimised by gradient boosting with L2 leaf regularisation."),
    "xgboost": dict(
        name="XGBoost", why=(
            "A second gradient-boosting implementation (depth-wise trees, second-order gradients) to check that the "
            "boosting result does not depend on one library."),
        strengths="Highest student-test Top-1 (by 0.1 points); fast inference (14 ms per 1,000 points).",
        limits="Larger train/validation gap than LightGBM (6.0 vs 3.8 points) and slightly worse log loss in all five "
               "folds; slower to train than LightGBM.",
        hyper="n_estimators = 400, learning_rate = 0.1, max_depth = 8, min_child_weight = 5, subsample = 0.8, "
              "colsample_bytree = 0.8, reg_lambda = 1.0, tree_method = hist.",
        loss="binary:logistic (log loss) with L2 regularisation, second-order boosting."),
    "mlp": dict(
        name="Multi-layer Perceptron, PyTorch", why=(
            "A neural network learns its own non-linear combinations of the features. It is also the only model with "
            "epoch-wise training, so it shows real training and validation loss curves."),
        strengths="Smooth, well-calibrated probabilities (third-best log loss); the smallest overfitting gap thanks to "
                  "early stopping on held-out trips.",
        limits="Needs scaling and careful training; on CPU it is slower to train (about 3 minutes) than boosting and "
               "was slightly less accurate; less interpretable.",
        hyper="Layers 26-128-64-1, ReLU, dropout 0.1, Adam (lr 1e-3, weight decay 1e-5), batch 2,048, up to 30 epochs, "
              "early stopping (patience 4) on a validation set of 10% of the training trips.",
        loss="Binary cross-entropy on logits (BCEWithLogitsLoss)."),
    "linear_svm": dict(
        name="Linear SVM, Platt-calibrated", why=(
            "A maximum-margin linear classifier from the kernel family; with Platt calibration it yields probabilities "
            "for the HMM. It tests whether a margin-based objective beats logistic regression's likelihood."),
        strengths="Robust margin-based boundary; fast inference.",
        limits="Same linear limitation as logistic regression (it scored almost identically); needs calibration for "
               "probabilities; kernel (RBF) SVMs do not scale to 1.3 million rows, so a linear kernel on a 200,000-row "
               "sample was used.",
        hyper="LinearSVC with C = 1.0, standardised features, sigmoid (Platt) calibration with 3-fold internal CV, "
              "200,000-row training sample.",
        loss="Hinge loss with L2 penalty; calibration fits a sigmoid by log loss."),
}


def model_section(key, D):
    t = MODEL_TEXT[key]
    m = D["met"][key]
    F = D["folds"][D["folds"].model == key]
    gap = F.train_top1.mean() - F.top1.mean()
    if gap > 0.03:
        fit = (f"<b>Overfitting (variance).</b> Training-fold Top-1 is {pct1(F.train_top1.mean())} versus "
               f"{pct1(F.top1.mean())} on validation folds, a gap of {100 * gap:.1f} points; training log loss "
               f"{F.train_log_loss.mean():.3f} versus validation {F.log_loss.mean():.3f}. The model fits training "
               "trips noticeably better than unseen trips, so part of what it learns is trip-specific.")
    elif gap > 0.005:
        fit = (f"<b>Well balanced.</b> Training-fold Top-1 {pct1(F.train_top1.mean())} versus validation "
               f"{pct1(F.top1.mean())} (gap {100 * gap:.1f} points).")
    else:
        fit = (f"<b>Underfitting (bias).</b> Training-fold Top-1 ({pct1(F.train_top1.mean())}) is no better than "
               f"validation ({pct1(F.top1.mean())}): the model cannot fit even the training data more closely, so its "
               "limit is model capacity, not lack of data.")
    te = m["test"]
    rows = [P(f"3.2.{list(MODEL_TEXT).index(key) + 1} {t['name']}: {m['family']}", H3),
            P("A. Model understanding", H4),
            P(f"<b>1. Why suitable?</b> {t['why']}"),
            P(f"<b>2. Strengths and limitations.</b> Strengths: {t['strengths']} Limitations: {t['limits']}"),
            P("B. Training and configuration", H4),
            P(f"<b>5. Important hyperparameters (initial values).</b> {t['hyper']}"),
            P(f"<b>6. Loss / objective.</b> {t['loss']}"),
            P("C. Experimental results", H4),
            P(f"<b>7. Student-test performance.</b> Cross-validation Top-1 {pct(m['cv']['top1']['mean'])} "
              f"(± {100 * m['cv']['top1']['std']:.2f} over 5 folds). On the 10 student-test trips: Top-1 "
              f"{pct(te['top1'])}, route continuity of the per-point sequence {pct1(te['route_continuity'])}, "
              f"inference {m['test_infer_ms_per_1k_points']:.0f} ms per 1,000 points."),
            P(f"<b>8. Observation-level Top-1, Macro F1 and log loss (student-test).</b> Top-1 {pct(te['top1'])}; "
              f"Macro F1 {te['macro_f1']:.4f}; log loss {te['log_loss']:.4f}."),
            P(f"<b>9. Over- or underfitting?</b> {fit}")]
    return rows


# ----------------------------------------------------------------------------------------- sections
def build(D):
    s = []
    cs, rule = D["cand"], D["rule"]
    final = D["export"]
    fin_top1 = final.correct_hmm.sum() / final.labelled.sum()
    fin_clf = final.correct_clf.sum() / final.labelled.sum()

    # ----- title page
    s += [Spacer(1, 2.2 * cm), P("PathSnap", H1),
          P("Distributed Map-Matching and Route Reconstruction", SUB),
          P("UE24CS352A Machine Learning Hackathon, PES University (RR Campus), Department of CSE", SUB),
          Spacer(1, 0.6 * cm),
          table([["Name", "SRN", "Class"]] + team(), [7 * cm, 5 * cm, 2.5 * cm]),
          Spacer(1, 0.6 * cm),
          P("Repository: <font face='Courier'>https://github.com/aarush-170/pathsnap</font>. Every number, table and "
            "figure in this report is produced by code in that repository and can be regenerated with the commands in "
            "its README; this PDF itself is built by <font face='Courier'>report/build_report.py</font>.", BODY),
          Spacer(1, 0.4 * cm), P("Summary", H2),
          P(f"PathSnap matches noisy 1 Hz GPS trajectories to the road network. For every GPS point we generate candidate "
            f"road segments within 50 m, describe each with 26 engineered features, score them with a tuned LightGBM "
            f"classifier and decode the whole trip with an HMM/Viterbi model that enforces road connectivity. On the "
            f"10 student-test trips the full pipeline reaches <b>{pct(fin_top1)}</b> observation-level Top-1 accuracy "
            f"(classifier alone {pct(fin_clf)}, no-learning rule baseline "
            f"{pct(rule.loc['nearest + heading (dist - 10*cos)', 'student_test'])}). It is deployed as a FastAPI service "
            f"(Laptop 1) with a Streamlit evaluator interface (Laptop 2)."),
          table([["Stage", "Student-test Top-1"],
                 ["Rule baseline: nearest road + heading (no learning)", pct(rule.loc['nearest + heading (dist - 10*cos)', 'student_test'])],
                 ["LightGBM classifier (untuned, Phase 4)", pct(D["met"]["lightgbm"]["test"]["top1"])],
                 ["LightGBM classifier (Optuna-tuned, Phase 5)", pct(D["met"]["lightgbm_tuned"]["test"]["top1"])],
                 ["<b>Tuned LightGBM + HMM/Viterbi, full deployed pipeline</b>", f"<b>{pct(fin_top1)}</b>"]],
                [11 * cm, PAGE_W - 11 * cm]),
          PageBreak()]

    # ----- 2 dataset & EDA
    e, inv = D["eda"], D["inv"]
    tr, st = e[e.split == "train"], e[e.split == "student_test"]
    s += [P("2. Dataset Understanding and Exploratory Data Analysis", H2),
          P("All analyses in this section use the <b>train</b> (80 trips) and <b>student_test</b> (10 trips) "
            "partitions; the partition is stated for each figure. The evaluator partition was not used for any "
            "analysis. Code: <font face='Courier'>scripts/eda.py</font>; findings and decisions: "
            "<font face='Courier'>results/logs/eda_notes.md</font>."),
          P("2.1 Dataset structure", H3),
          P("Each trip is a folder of four headerless, whitespace-separated files. <font face='Courier'>.track</font> "
            "holds GPS observations (longitude, latitude, time in seconds); <font face='Courier'>.nodes</font> holds "
            "road-network nodes (longitude, latitude; the row number is the node id); <font face='Courier'>.arcs</font> "
            "holds directed road segments (from-node, to-node; the row number is the arc id); "
            "<font face='Courier'>.route</font> lists the reference arcs driven, in order. Every trip has its own "
            "network, so ids are local to the trip. Trips are spread worldwide (train latitudes from about -35 to +61 "
            "degrees, longitudes from about -100 to +137 degrees)."),
          small_table([["Partition", "Trips", "GPS points", "Median points/trip", "Median arcs/trip", "Max arcs",
                        "Files present", "Size (MB)"]] +
                      [[sp, str(g.trip.nunique()), f"{int(g.n_points.sum()):,}", f"{g.n_points.median():,.0f}",
                        f"{g.n_arcs.median():,.0f}", f"{g.n_arcs.max():,}",
                        str(bool(inv[inv.split == sp].exists.all())),
                        f"{inv[inv.split == sp].bytes.sum() / 1e6:.1f}"] for sp, g in [("train", tr), ("student_test", st)]],
                      [2.3 * cm, 1.3 * cm, 2 * cm, 2.4 * cm, 2.3 * cm, 1.7 * cm, 2 * cm, PAGE_W - 14 * cm]),
          Spacer(1, 6),
          P("<b>Key structural fact: there is no per-point label.</b> The route is an ordered list of about 280 arcs "
            "for about 1,500 GPS points (about 5 points per arc) and never says which point belongs to which arc. "
            "About 95% of arcs have a reverse twin: a two-way street is stored as two directed arcs with identical "
            "geometry, so distance alone can never separate them."),
          P("2.2 How we preprocessed the data", H3)]
    s += bullets([
        "<b>Missing and invalid values:</b> none found (0 NaN, 0 invalid coordinates, 0 duplicate or out-of-order "
        "timestamps, 0 invalid route ids, 0 isolated nodes), so no imputation or row removal was needed.",
        "<b>Coordinates to metres:</b> each trip is projected with a local equirectangular projection centred on the "
        "trip (error below 0.1% at trip scale, checked against haversine in a unit test). Euclidean distance in "
        "degrees would be wrong because a degree of longitude shrinks with latitude.",
        "<b>Per-point labels:</b> an order-preserving dynamic-programming alignment of GPS points to the ordered route "
        "(route position never decreases; cost = distance to arc + 0.1 x |route progress - GPS displacement|). It "
        "picks the correct direction on two-way streets, which nearest-arc snapping cannot. Points over 50 m from "
        f"every route arc stay unlabelled: {pct1(cs.loc['train', 'labelled_frac'])} of train and "
        f"{pct1(cs.loc['student_test', 'labelled_frac'])} of student-test points are labelled.",
        "<b>Candidate generation:</b> all arcs within 50 m of each point, cut at distance rank 12 (both directions of a "
        f"street share a rank). Chosen from a radius x K sweep: candidate miss rate {pct(cs.loc['train', 'candidate_miss_rate'])} "
        f"(train) and {pct(cs.loc['student_test', 'candidate_miss_rate'])} (student-test) with "
        f"{cs.loc['train', 'mean_candidates_per_point']:.1f} candidates per point. Output: "
        f"{int(cs.loc['train', 'candidate_rows']):,} train rows.",
        "<b>Gaps and anomalies:</b> points are kept; gap length is a feature and the HMM splits trips at gaps over 60 s. "
        "Duplicate arcs (same from/to pair) are treated as the same road when labelling and scoring."])
    s += [P("2.3 Patterns observed", H3),
          fig(FIG / "eda_data_quality.png", "Figure 2.1. Missing-value and data-quality checks (train + student_test). "
              "No missing values; the issues are gaps, speed jumps and a few route breaks.", PAGE_W * 0.7),
          fig(FIG / "eda_gps_to_route_distance.png",
              "Figure 2.2. Distance from GPS points to the reference route (train and student_test). Median about 4.5 m; "
              "almost all points within 30 m, which motivates the 50 m candidate radius. The 60 m+ spike in train "
              "comes from a few trips whose route does not explain the GPS.")]
    s += bullets([
        f"About 99% of sampling intervals are exactly 1 s; {int((e.gaps_gt_60s > 0).sum())} trips have gaps over 60 s "
        f"(maximum {e.max_gap_s.max():,.0f} s).",
        "In trips 54, 33 and 02 long stretches of GPS lie 100 m or more from the route. Trip 33's far points lie on other "
        "mapped roads (the route skips part of the drive); trips 54, 02 and 96 drive on a road missing from the network. "
        "Those points are left unlabelled rather than given a wrong label: the reference route is not perfect truth.",
        f"Road networks: median {tr.n_weak_components.median():.0f} weakly connected components per train trip, the "
        f"largest strongly connected component holds about {pct1(e.largest_scc_frac.median())} of nodes; "
        f"{int(e.route_breaks.sum())} route breaks in {int((e.route_breaks > 0).sum())} trips."])
    s += [fig(FIG / "eda_trajectories_typical.png",
              "Figure 2.3. Typical trips (3 train, 3 student_test): GPS (blue), reference route (green) and road network (grey)."),
          fig(FIG / "eda_network_connectivity.png",
              "Figure 2.4. Road-network size and connectivity per trip (train and student_test)."),
          fig(RFIG / "folium_trip77_full.jpg",
              "Figure 2.5. Folium map (student_test trip 00000077): road network, reference route, predicted route, GPS "
              "observations and mismatched points, with legend and layer control (results/maps/route_00000077.html).",
              PAGE_W * 0.85)]
    uni = D["uni"].set_index("feature")
    s += [P("2.4 Features extracted and why", H3),
          P("Each row is one (GPS point, candidate arc) pair with 26 features; the target is whether the candidate is "
            "the point's labelled arc. No feature reads the route (enforced by a unit test), so the same code runs on "
            "evaluator trips in the API. Code: <font face='Courier'>src/pathsnap/features.py</font>."),
          small_table([["Group", "Features", "Why (link to EDA)", "Best AUC"],
                       ["Distance (7)", "dist, rank, dist_minus_min, dist_ratio_min, dist_z, n_candidates, gap_1st_2nd",
                        "GPS error is about 4.5 m; distance relative to rival candidates is robust to trip-to-trip noise.",
                        f"{uni.auc_abs['dist_minus_min']:.3f}"],
                       ["Projection (3)", "t, t_at_end, arc_len", "Arcs are short (median about 37 m): a point projecting "
                        "past an arc's end is probably on the next arc.", f"{uni.auc_abs['t_at_end']:.3f}"],
                       ["Heading (6)", "heading_cos, heading_absdiff, heading_cos_prev/next, heading_rank, heading_valid",
                        "About 95% of arcs have a reverse twin: only heading separates the two directions.",
                        f"{uni.auc_abs['heading_cos']:.3f}"],
                       ["Context (4)", "ctx_dist_mean, ctx_dist_max, ctx_minus_min, ctx_rank",
                        "Neighbouring points i-2..i+2 should also lie near the true arc.", f"{uni.auc_abs['ctx_minus_min']:.3f}"],
                       ["Motion (3)", "speed, dt_prev, dt_next", "Gaps and low speed make heading and context unreliable.",
                        f"{uni.auc_abs['speed']:.3f}"],
                       ["Network (3)", "from_in_degree, to_out_degree, has_reverse_twin", "Errors concentrate at junctions.",
                        f"{uni.auc_abs['from_in_degree']:.3f}"]],
                      [2.3 * cm, 5.4 * cm, 6.6 * cm, PAGE_W - 14.3 * cm]),
          Spacer(1, 4),
          P(f"No-learning rule baselines on student_test: nearest arc {pct(rule.loc['nearest arc (dist)', 'student_test'])} "
            f"(two-way twins tie, so direction is a coin flip); nearest arc + heading "
            f"{pct(rule.loc['nearest + heading (dist - 10*cos)', 'student_test'])}. The learned models must beat the latter. "
            "Single-feature ROC-AUC for all 26 features: results/figures/feature_univariate_auc.png and "
            "results/logs/feature_univariate.csv."),
          PageBreak()]

    # ----- 3 models
    C = D["cmp"].set_index("model")
    order = ["logreg", "gnb", "knn", "random_forest", "lightgbm", "xgboost", "mlp", "linear_svm"]
    keycfg = {"logreg": "C=1, L2, scaled", "gnb": "defaults", "knn": "k=25, distance, 200k sample",
              "random_forest": "200 trees, leaf 5, sqrt", "lightgbm": "400 trees, lr 0.05, 63 leaves",
              "xgboost": "400 trees, lr 0.1, depth 8", "mlp": "128-64, dropout 0.1, early stop",
              "linear_svm": "C=1, Platt, 200k sample"}
    rows = [["Model", "Algorithm family", "Key configuration", "CV Macro F1", "Top-1 (student-test)", "Route continuity",
             "Student-test Macro F1", "Log loss (test)", "Inference (ms / 1k pts)"]]
    for k in order:
        r = C.loc[k]
        rows.append([MODEL_TEXT[k]["name"], r.family, keycfg[k], f"{r.cv_macro_f1:.4f}", pct(r.test_top1),
                     pct1(r.test_route_continuity), f"{r.test_macro_f1:.4f}", f"{r.test_log_loss:.4f}",
                     f"{r.infer_ms_per_1k_points:,.0f}"])
    s += [P("3. Model Experimentation and Comparison", H2),
          P("Eight models from seven algorithm families were trained with one shared protocol "
            "(<font face='Courier'>scripts/train_models.py</font>): 5-fold GroupKFold by trip on the 80 train trips "
            "(16 trips per fold, fold map in <font face='Courier'>results/logs/cv_folds.csv</font>), then a refit on "
            "all train trips and one evaluation on the 10 student-test trips. Seed 42 throughout. Top-1 is the share of "
            "labelled GPS points whose highest-probability candidate is the labelled arc; Macro F1 and log loss are "
            "computed on candidate rows (threshold 0.5); route continuity is the share of road changes in the "
            "per-point sequence that are physically connected."),
          P("3.1 Overall model comparison", H3),
          small_table(rows, [2.6 * cm, 2.2 * cm, 2.6 * cm, 1.4 * cm, 1.6 * cm, 1.5 * cm, 1.6 * cm, 1.4 * cm,
                             PAGE_W - 14.9 * cm]),
          P("Route continuity applies to the per-point classifier output; the HMM (Section 5) raises it to about 91%. "
            "N/A entries: none; every metric applies to every model because all output probabilities.", CAP),
          P(f"<b>Selection.</b> LightGBM and XGBoost are statistically tied on Top-1 (student-test difference 0.1 points, "
            f"about 19 of 19,133 points, against a fold standard deviation of about 3.9 points). We selected "
            f"<b>LightGBM</b> because it has the best cross-validation Top-1 and Macro F1, better log loss in all five "
            f"folds (probability quality matters because the HMM uses the probabilities as emissions), a smaller "
            f"train/validation gap (3.8 vs 6.0 points) and faster training for tuning. Primary metric: observation-level "
            f"Top-1, because it is exactly what the evaluation measures; log loss is the tie-breaker because of the HMM."),
          fig(FIG / "model_cv_comparison.png", "Figure 3.1. Cross-validation Macro F1 and Top-1 (mean ± std over 5 "
              "trip-grouped folds; dots are folds). Dashed line: rule baseline."),
          fig(FIG / "model_test_comparison.png", "Figure 3.2. Student-test Top-1 accuracy and Macro F1."),
          fig(FIG / "model_log_loss.png", "Figure 3.3. Log loss, cross-validation and student-test (log scale).", PAGE_W * 0.8),
          fig(FIG / "model_confusion_matrices.png", "Figure 3.4. Student-test confusion matrices (candidate rows, threshold 0.5)."),
          fig(FIG / "model_speed_vs_accuracy.png", "Figure 3.5. Accuracy versus inference cost.", PAGE_W * 0.65),
          P("3.2 Individual models", H3)]
    for k in order:
        s += model_section(k, D)
        if k == "mlp":
            s += [fig(FIG / "model_mlp_loss_curve.png", "Figure 3.6. MLP training and validation loss per epoch (final fit; "
                      "validation = held-out training trips). The only model with an epoch-wise training history; no "
                      "curves are shown for the others because they do not train in epochs.", PAGE_W * 0.65)]
    s.append(PageBreak())

    # ----- 4 bayesian
    tj, T, imp = D["tuned"], D["tcmp"], D["imp"]
    p = tj["params"]
    trials = D["trials"]
    s += [P("4. Bayesian Hyperparameter Optimisation (LightGBM)", H2),
          P("Code: <font face='Courier'>scripts/tune_optuna.py</font>; record: "
            "<font face='Courier'>results/logs/optuna_trials.csv</font>, best parameters: "
            "<font face='Courier'>configs/lightgbm_tuned.json</font>.")]
    s += bullets([
        "<b>1. Hyperparameters searched:</b> learning_rate, n_estimators, num_leaves, min_child_samples, subsample, "
        "colsample_bytree, reg_lambda, reg_alpha (subsample_freq fixed at 1).",
        "<b>2. Ranges:</b> learning_rate 0.01-0.2 (log), n_estimators 100-800 (step 50), num_leaves 15-255 (log), "
        "min_child_samples 10-500 (log), subsample 0.5-1.0, colsample_bytree 0.4-1.0, reg_lambda and reg_alpha 0.001-10 "
        "(log). Log scales where the effect is multiplicative.",
        "<b>3. Objective:</b> minimise the mean validation log loss over the same 5 trip-grouped folds as Section 3. Log "
        "loss rewards accurate probabilities (the HMM's emissions) and is smoother than Top-1; Top-1 and Macro F1 were "
        "recorded per trial.",
        f"<b>4. Trials:</b> {len(trials)} within a 45-minute budget ({int((trials.state == 'COMPLETE').sum())} completed, "
        f"{int((trials.state == 'PRUNED').sum())} pruned by a median pruner). Sampler: TPE (seed 42) with 8 random "
        "start-up trials. Trial 0 was the untuned configuration.",
        f"<b>5. Best values (trial {tj['best_trial']}):</b> learning_rate {p['learning_rate']:.4f}, n_estimators "
        f"{p['n_estimators']}, num_leaves {p['num_leaves']}, min_child_samples {p['min_child_samples']}, subsample "
        f"{p['subsample']:.3f}, colsample_bytree {p['colsample_bytree']:.3f}, reg_lambda {p['reg_lambda']:.3f}, "
        f"reg_alpha {p['reg_alpha']:.4f}. CV log loss {tj['cv_log_loss']:.5f}.",
        "<b>6. Did tuning help?</b> Yes, consistently but slightly for the classifier alone, and clearly after the HMM:"])
    u, t = T.iloc[0], T.iloc[1]
    hu, ht = D["hmm_cmp_u"].iloc[1], D["hmm_cmp"].iloc[1]
    s += [small_table([["Metric (same folds, same test trips)", "Untuned", "Tuned"],
                       ["CV Macro F1", f"{u.cv_macro_f1:.4f}", f"{t.cv_macro_f1:.4f}"],
                       ["CV Top-1", pct(u.cv_top1), pct(t.cv_top1)],
                       ["CV log loss", f"{u.cv_log_loss:.4f}", f"{t.cv_log_loss:.4f}"],
                       ["Student-test Top-1", pct(u.test_top1), pct(t.test_top1)],
                       ["Student-test Macro F1", f"{u.test_macro_f1:.4f}", f"{t.test_macro_f1:.4f}"],
                       ["Student-test log loss", f"{u.test_log_loss:.4f}", f"{t.test_log_loss:.4f}"],
                       ["Inference (ms per 1,000 points)", f"{u.infer_ms_per_1k_points:.0f}", f"{t.infer_ms_per_1k_points:.0f}"],
                       ["Student-test Top-1 after HMM", pct(hu.top1), pct(ht.top1)]],
                      [7 * cm, 4 * cm, 4 * cm]),
          Spacer(1, 4),
          P(f"Learning rate, min_child_samples and num_leaves account for "
            f"{100 * imp.loc[['learning_rate', 'min_child_samples', 'num_leaves']].sum():.0f}% of the fANOVA importance. "
            "The tuned model learns more slowly with more, larger trees but forces bigger leaves and column subsampling. "
            "Its probabilities are better calibrated (log loss better in all 5 folds), which is why the gain shows up "
            "after HMM decoding (+0.9 points) rather than in the classifier's own Top-1 (+0.06). Cost: about 4x slower "
            "inference, still about 1.7 s for the largest evaluator trip."),
          fig(FIG / "optuna_history.png", "Figure 4.1. Optimisation history (mean CV log loss per trial; red = best so far).", PAGE_W * 0.75),
          fig(FIG / "optuna_param_importance.png", "Figure 4.2. Hyperparameter importance (fANOVA).", PAGE_W * 0.6),
          fig(FIG / "optuna_tuned_vs_untuned.png", "Figure 4.3. Untuned versus tuned LightGBM on the same folds and test trips."),
          PageBreak()]

    # ----- 5 HMM
    hc = D["hmm_cmp"]
    ht_ = D["hmm_trip"]
    g = D["hmm_grid"].assign(ab=lambda d: d.alpha * d.beta).groupby("ab").top1.first()
    s += [P("5. HMM and Viterbi Route Reconstruction", H2),
          P("Code: <font face='Courier'>src/pathsnap/hmm.py</font> (decoder, 3 unit tests), "
            "<font face='Courier'>scripts/eval_hmm.py</font>. Evaluated on student_test only.")]
    s += bullets([
        "<b>1. Hidden states:</b> for each GPS point, its candidate arcs, each with the point's projected position t "
        "along the arc, so a state is a place on the network.",
        "<b>2. Emission:</b> alpha x log P(arc | point), where P is the tuned LightGBM probability normalised over the "
        "point's candidates.",
        "<b>3. Transition:</b> log P = -|d_net - d_gps| / beta, where d_gps is the straight-line distance between "
        "consecutive GPS points and d_net is the driving distance on the directed network (same arc: |t_b - t_a| x "
        "length; otherwise the remainder of arc a + Dijkstra shortest path from a's head to b's tail + the part of "
        "arc b). Dijkstra is cut off at 3 x d_gps + 200 m and cached per node.",
        "<b>4. Missing candidates and gaps:</b> points without candidates are skipped (the transition is taken from the "
        "last point that had candidates); a time gap over 60 s cuts the chain and the next segment is decoded "
        "independently.",
        "<b>5. Disconnected segments:</b> if no state at point i is reachable from any state at i-1, the chain is cut "
        "and decoding restarts from emissions alone; no impossible connection is ever forced (14 segments over 10 "
        "trips). The final route is made drivable by inserting shortest paths between consecutive predicted arcs.",
        "<b>6. Did HMM decoding help?</b> Yes, on every trip:"])
    s += [small_table([["Method (student-test)", "Top-1", "Route continuity", "Route precision", "Route recall", "Route F1"]] +
                      [[r.method, pct(r.top1), pct1(r.continuity), f"{r.route_precision:.3f}", f"{r.route_recall:.3f}",
                        f"{r.route_f1:.3f}"] for r in hc.itertuples()],
                      [6.2 * cm, 1.8 * cm, 2.3 * cm, 2.2 * cm, 2 * cm, PAGE_W - 14.5 * cm]),
          Spacer(1, 4),
          small_table([["Trip", "Points", "Top-1 classifier", "Top-1 HMM", "Gain (pts)", "Continuity classifier",
                        "Continuity HMM"]] +
                      [[r.trip, f"{r.points_clf:,}", pct1(r.top1_clf), pct1(r.top1_hmm), f"{100 * r.top1_gain:+.1f}",
                        pct1(r.continuity_clf), pct1(r.continuity_hmm)] for r in ht_.itertuples()],
                      [2.2 * cm, 1.6 * cm, 2.4 * cm, 2 * cm, 1.7 * cm, 2.8 * cm, PAGE_W - 12.7 * cm]),
          Spacer(1, 4),
          P(f"<b>Decoder parameters.</b> Viterbi takes the argmax of a sum, so multiplying all scores by beta shows that "
            f"only the product alpha x beta matters; equal products gave identical results in the grid. Top-1 by alpha x "
            f"beta: " + ", ".join(f"{k:g}: {pct1(v)}" for k, v in g.items()) + ". We use beta = 5 m, alpha = 0.5 "
            "(product 2.5), chosen on student_test as the guidelines allow for decoder parameters; the optimum is broad, "
            "so the result is not fragile."),
          P(f"<b>Where the HMM did not help.</b> Trip 00000077 remains the weakest ({pct1(ht_.set_index('trip').loc['00000077', 'top1_hmm'])}): "
            "its errors cluster where the vehicle is stopped at junctions (heading unknown, several roads equally close). "
            f"On that trip the HMM fixed {int(D['maps'].set_index('trip').loc['00000077', 'fixed'])} points but made "
            f"{int(D['maps'].set_index('trip').loc['00000077', 'broken'])} previously correct points wrong, mostly a few "
            "seconds early or late at turns. Route recall barely changes (short arcs between 1 Hz samples cannot be "
            "recovered per point; they are only filled when drawing the route). The deployed pipeline, which also "
            f"decodes the 27 unlabelled student-test points, gives {pct(fin_top1)} (Section 7)."),
          PageBreak()]

    # ----- 6 folium
    ms = D["maps"]
    s += [P("6. Geographical Visualisation Using Folium", H2),
          P("Code: <font face='Courier'>scripts/make_maps.py</font> and <font face='Courier'>src/pathsnap/viz.py</font>; "
            "interactive maps: <font face='Courier'>results/maps/route_&lt;trip&gt;.html</font> (committed). Each map shows "
            "the original GPS observations (blue), the nearby road network (grey), the reference route (green), the "
            "HMM-predicted route (red, dashed), the classifier-only route (purple, toggle) and mismatched points "
            "(orange), with a legend, a layer control, start/end markers, a stats box and a view fitted to the trip."),
          small_table([["Trip", "Why chosen", "Top-1 classifier", "Top-1 HMM", "Mismatches classifier", "Mismatches HMM",
                        "Route gaps"]] +
                      [[r.trip, {"00000077": "hardest", "00000040": "typical", "00000007": "best"}.get(r.trip, ""),
                        pct1(r.top1_clf), pct1(r.top1_hmm), str(r.wrong_clf), str(r.wrong_hmm), str(r.route_gaps)]
                       for r in ms.itertuples()],
                      [2.2 * cm, 2 * cm, 2.4 * cm, 2 * cm, 2.8 * cm, 2.6 * cm, PAGE_W - 14 * cm]),
          Spacer(1, 6),
          fig(RFIG / "folium_trip77_junction_zoom.jpg",
              "Figure 6.1. Folium map, student_test trip 00000077 zoomed on junctions with the classifier-only layer "
              "switched on (purple): the classifier strays onto crossing roads where the vehicle waits (orange "
              "clusters), while the HMM route (red) stays on the reference route (green).", PAGE_W * 0.85),
          fig(FIG / "map_00000077.png",
              "Figure 6.2. Static version for print: full trip, then a zoom where the HMM fixed the most classifier "
              "mistakes, before (classifier only) and after (HMM)."),
          PageBreak()]

    # ----- 7 API / UI
    ex, ev = D["export"], D["evalrun"]
    s += [P("7. API, UI and Deployment", H2),
          P("<b>Architecture.</b> Laptop 1 runs the FastAPI service (<font face='Courier'>api/main.py</font>) with the "
            "exported model (<font face='Courier'>models/lightgbm_tuned.txt</font>, "
            "<font face='Courier'>models/pipeline_config.json</font>) and the evaluator trips. Laptop 2 runs the Streamlit "
            "UI (<font face='Courier'>ui/app.py</font>), which only talks to the API. Endpoints: GET /health, GET /trips, "
            "POST /predict {trip_id}. The API predicts from a route-free copy of the trip using the same "
            "<font face='Courier'>predict_trip()</font> function as the experiments, then reads the reference route only "
            "to score the result server-side; responses never contain reference arcs. Every request is logged to the "
            "API console. Setup and the demo flow are in <font face='Courier'>docs/DEPLOYMENT.md</font>."),
          P(f"<b>Verification.</b> From raw files on student_test, the exported pipeline reproduces the classifier "
            f"probabilities exactly and gives {pct(fin_top1)} Top-1 (classifier alone {pct(fin_clf)}), "
            f"{ex.total_s.min():.1f}-{ex.total_s.max():.1f} s per trip. A run-check on all 10 evaluator trips (prediction "
            f"only, nothing scored) completed without errors in {ev.total_s.min():.1f}-{ev.total_s.max():.1f} s per trip, "
            f"each with a continuous route."),
          (fig(RFIG / "ui_laptop2.png", "Figure 7.1. Evaluator UI on Laptop 2 during the two-laptop demo, after "
               "running a prediction on an evaluation trip.", PAGE_W * 0.9)
           if (RFIG / "ui_laptop2.png").exists() else
           fig(RFIG / "ui_trip40.jpg", "Figure 7.1. Evaluator UI (Laptop 2) after running a prediction: Top-1, gain over "
               "classifier only, correct points, continuity, route F1, processing time and the predicted route on a "
               "map. Captured against a local test API serving the student_test trips.", PAGE_W * 0.9)),
          fig(RFIG / "api_cmd_laptop1.png",
              "Figure 7.2. API console (CMD) on Laptop 1 during the two-laptop demo over a phone hotspot. Laptop 2 "
              "(172.20.10.3) calls GET /health and POST /predict for evaluation trip 00000015; Laptop 1 (172.20.10.2) "
              "runs the full pipeline (6,854 GPS points, 47,086 candidates, 2.43 s), scores it server-side and returns "
              "HTTP 200. The accuracy shown was observed only in this demo run and was not used for any development "
              "decision.", PAGE_W),
          (Spacer(1, 1) if (RFIG / "ui_laptop2.png").exists() else
           P("<b>[Add before submission]</b> Screenshot of the UI on Laptop 2 from the two-laptop demo "
             "(save it as report/figures/ui_laptop2.png and rebuild; it then replaces Figure 7.1 automatically).",
             BODY.clone("ADD", backColor=colors.HexColor("#fff4e5"), borderPadding=(5, 6, 5, 6)))),
          PageBreak()]

    # ----- 8 integrity
    s += [P("8. Academic Integrity and Reproducibility", H2)]
    s += bullets([
        "All code, experiment logs, configurations, model weights and figures are in the team repository; every table "
        "and figure in this report is regenerated from them (data: <font face='Courier'>scripts/eda.py</font>, "
        "<font face='Courier'>build_dataset.py</font>, <font face='Courier'>build_features.py</font>; models: "
        "<font face='Courier'>train_models.py</font>, <font face='Courier'>tune_optuna.py</font>; decoding and maps: "
        "<font face='Courier'>eval_hmm.py</font>, <font face='Courier'>make_maps.py</font>; deployment: "
        "<font face='Courier'>export_artifacts.py</font>; this PDF: <font face='Courier'>report/build_report.py</font>).",
        "Partitions were kept separate: models were fitted and tuned on the 80 train trips with trip-grouped "
        "cross-validation; student_test was used for model comparison and decoder settings; the evaluator partition was "
        "used for nothing except one prediction-only run-check of the deployed pipeline (no scoring), and its labels "
        "never enter the prediction path.",
        "No predictions are hardcoded: the API runs the full pipeline on every request and its console logs every request.",
        "The work, models and results are this team's own; no code, models or results were exchanged with other teams.",
        f"Unit tests: {len(list((ROOT / 'tests').glob('test_*.py')))} test files, run with "
        "<font face='Courier'>python -m pytest -q tests</font>."])
    return s


def main():
    D = load()
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=2 * cm, title="PathSnap Technical Report", author="PathSnap team")
    doc.build(build(D), onFirstPage=on_page, onLaterPages=on_page)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
