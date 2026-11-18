"""Build report/PathSnap_Data_Preprocessing_Features.pdf: a plain-language explainer of the
dataset, preprocessing and feature engineering, with figures from results/.

Usage (from repo root):  python report/build_explainer_pdf.py
Key numbers are read from results/logs so the PDF always matches the latest run.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parents[1]
LOG, FIG = ROOT / "results" / "logs", ROOT / "results" / "figures"
OUT = ROOT / "report" / "PathSnap_Data_Preprocessing_Features.pdf"

INK, MUTED, ACCENT, LIGHT = colors.HexColor("#1f2933"), colors.HexColor("#52606d"), \
    colors.HexColor("#2463eb"), colors.HexColor("#eef2f7")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Title"], fontSize=22, leading=27, textColor=INK, spaceAfter=4)
SUB = ParagraphStyle("SUB", parent=ss["Normal"], fontSize=11, textColor=MUTED, alignment=TA_CENTER, spaceAfter=14)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=15, leading=19, textColor=ACCENT, spaceBefore=8, spaceAfter=6)
H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontSize=11.5, leading=15, textColor=INK, spaceBefore=6, spaceAfter=3)
BODY = ParagraphStyle("BODY", parent=ss["Normal"], fontSize=10, leading=14, textColor=INK, spaceAfter=5)
CELL = ParagraphStyle("CELL", parent=BODY, fontSize=8.8, leading=11.5, spaceAfter=0)
CELLB = ParagraphStyle("CELLB", parent=CELL, fontName="Helvetica-Bold")
CAP = ParagraphStyle("CAP", parent=BODY, fontSize=8.5, leading=11, textColor=MUTED, alignment=TA_CENTER, spaceAfter=8)
SAY = ParagraphStyle("SAY", parent=BODY, backColor=LIGHT, borderPadding=(6, 8, 6, 8), leftIndent=8, rightIndent=8,
                     spaceBefore=6, spaceAfter=10)
BUL = ParagraphStyle("BUL", parent=BODY, leftIndent=14, bulletIndent=4, spaceAfter=3)

PAGE_W = A4[0] - 4 * cm


def P(text, style=BODY):
    return Paragraph(text, style)


def say(text):
    return P(f"<b>How to say it:</b> <i>{text}</i>", SAY)


def bullets(items):
    return [Paragraph(t, BUL, bulletText="•") for t in items]


def table(rows, widths, header=True, zebra=True):
    data = [[P(c, CELLB if (header and i == 0) else CELL) if isinstance(c, str) else c for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd2d9")),
          ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe4f0")))
    if zebra:
        for r in range(1 if header else 0, len(rows)):
            if r % 2 == 0:
                st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f7f9fb")))
    t.setStyle(TableStyle(st))
    return t


def figure(name, caption, width=PAGE_W):
    from reportlab.lib.utils import ImageReader
    p = FIG / name
    w, h = ImageReader(str(p)).getSize()
    img = Image(str(p), width=width, height=width * h / w)
    return KeepTogether([img, P(caption, CAP)])


def pipeline_diagram():
    steps = ["Raw files\n(4 per trip)", "Convert to\nmetres", "Label each\nGPS dot", "Shortlist\nnearby roads",
             "Describe each\nroad (26 features)", "Table for\nthe model"]
    n, gap = len(steps), 10
    bw = (PAGE_W - gap * (n - 1)) / n
    bh, d = 44, Drawing(PAGE_W, 60)
    for i, s in enumerate(steps):
        x = i * (bw + gap)
        d.add(Rect(x, 8, bw, bh, rx=6, ry=6, fillColor=LIGHT if i not in (2, 3, 4) else colors.HexColor("#dbe8ff"),
                   strokeColor=ACCENT, strokeWidth=0.8))
        lines = s.split("\n")
        for j, ln in enumerate(lines):
            d.add(String(x + bw / 2, 8 + bh / 2 + 5 - 11 * j, ln, fontName="Helvetica", fontSize=7.6,
                         fillColor=INK, textAnchor="middle"))
        if i < n - 1:
            ax = x + bw + 1
            d.add(Line(ax, 8 + bh / 2, ax + gap - 3, 8 + bh / 2, strokeColor=MUTED, strokeWidth=1))
            d.add(Polygon([ax + gap - 1, 8 + bh / 2, ax + gap - 5, 8 + bh / 2 + 3, ax + gap - 5, 8 + bh / 2 - 3],
                          fillColor=MUTED, strokeColor=MUTED))
    return d


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, 1.2 * cm, "PathSnap: dataset, preprocessing and features")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def main():
    cs = pd.read_csv(LOG / "candidate_summary.csv").set_index("split")
    rb = pd.read_csv(LOG / "rule_baselines.csv").set_index("rule")
    ex = pd.read_csv(LOG / "feature_example.csv")
    tr, st = cs.loc["train"], cs.loc["student_test"]
    pct = lambda x: f"{100 * x:.1f}%"
    near_st, nh_st = rb.loc["nearest arc (dist)", "student_test"], rb.loc["nearest + heading (dist - 10*cos)", "student_test"]
    near_tr, nh_tr = rb.loc["nearest arc (dist)", "train"], rb.loc["nearest + heading (dist - 10*cos)", "train"]

    s = []
    # ---------------------------------------------------------------- page 1: the problem
    s += [P("PathSnap", H1), P("Dataset, preprocessing and feature engineering, explained simply", SUB)]
    s += [P("The problem in one paragraph", H2),
          P("A car drives through a city and its GPS records a dot every second. The dots are not exactly on the "
            "road: they are usually about 5 m off, sometimes much more, because buildings and signal loss disturb "
            "the GPS. <b>Our task is to say, for every dot, which road the car was really on.</b> This is called "
            "map-matching."),
          P("We do it in three moves. First we <b>shortlist</b> the roads close to each dot (the candidates). "
            "Then we <b>describe</b> every road on the shortlist with a few numbers (the features), such as how close "
            "it is and whether it points the way the car is moving. Finally a model <b>chooses</b> the right road, "
            "having learned from past trips which descriptions mean \"this is the one\". This document covers the "
            "data and the first two moves."),
          P("The pipeline", H3), pipeline_diagram(),
          P("Highlighted steps are the ones this document explains in detail. Every step is code in "
            "<font face='Courier'>src/pathsnap/</font> and can be re-run from the repo.", CAP)]
    s += [say("GPS dots are noisy, so for each dot we shortlist nearby roads, describe each road with simple "
              "numbers, and a model learns to pick the right one.")]

    # ---------------------------------------------------------------- dataset
    s += [PageBreak(), P("1. The dataset", H2),
          P("Every trip is a folder with four plain-text files: the GPS dots, a map of the area (two files), "
            "and the answer."),
          table([["File", "What is in it", "In simple words"],
                 [".track", "longitude, latitude, time (seconds)", "The GPS dots, one per second"],
                 [".nodes", "longitude, latitude (row number = id)", "Points on the map: junctions and bends"],
                 [".arcs", "from-node, to-node (row number = id)", "Road pieces joining two points, with a direction"],
                 [".route", "arc ids, in order", "The answer: road pieces the car really drove"]],
                [2.2 * cm, 6.3 * cm, PAGE_W - 8.5 * cm]),
          Spacer(1, 8),
          P("Three things to know about this data", H3)]
    s += bullets([
        "<b>The answer does not say which dot goes with which road.</b> A route lists about 280 road pieces for "
        "about 1,500 dots, so roughly 5 dots fall on each piece. We have to work out which dot belongs to which "
        "road ourselves (Section 3, Step 2).",
        "<b>Most streets are stored twice.</b> A two-way street is two road pieces, one per direction, lying "
        "exactly on top of each other (about 95% of road pieces have such a twin). Distance alone can never tell "
        "them apart.",
        "<b>Each trip has its own map,</b> so road ids only mean something inside that trip."])
    s += [Spacer(1, 4),
          table([["Group", "Trips", "GPS dots", "Used for"],
                 ["train", "80", f"{int(tr.points):,}", "Learning, cross-validation, tuning"],
                 ["student_test", "10", f"{int(st.points):,}", "Comparing models (our own test)"],
                 ["evaluator", "10", "never opened", "Organisers' final demo, only through the API"]],
                [3 * cm, 1.8 * cm, 3 * cm, PAGE_W - 7.8 * cm]),
          Spacer(1, 6),
          figure("eda_trajectories_typical.png",
                 "Typical trips. Blue = GPS dots, green = the true route, grey = the road map of the area.",
                 width=PAGE_W * 0.92),
          say("Each trip has GPS dots, a road map and the true route. The catch is that the route doesn't say "
              "which dot belongs to which road, and two-way streets are stored twice.")]

    # ---------------------------------------------------------------- data quality
    s += [PageBreak(), P("2. What was messy, and what we did", H2),
          P("The data had <b>no missing values, no invalid coordinates and no out-of-order timestamps</b>. "
            "About 99% of dots are exactly one second apart. The real problems were in a few trips."),
          table([["Problem", "What we found", "What we did"],
                 ["GPS wobble", "Dots are typically about 5 m from the true road; 95% within about 15 m",
                  "Expected. It is why we consider every road within 50 m, not just the nearest."],
                 ["Gaps in the GPS", "24 trips have a silence longer than 60 s (the worst: 153,064 s)",
                  "Kept the dots, and gave the model the gap length as a feature."],
                 ["Dots far from the answer", "In trips 54, 33, 02 long stretches sit 100+ m from the route",
                  "Gave those dots no label rather than a wrong one."],
                 ["Two causes of that", "Trip 33: the car was on mapped roads the route skips. Trips 54, 02, 96: "
                  "the road is missing from the map.", "The answer file is a reference, not perfect truth."],
                 ["Duplicate road pieces", "The same road stored more than once (up to 322 in one trip)",
                  "Treated as one road when labelling."]],
                [3.2 * cm, 6.4 * cm, PAGE_W - 9.6 * cm]),
          Spacer(1, 8),
          figure("eda_gps_to_route_distance.png",
                 "How far GPS dots are from the true route. Almost all are within 30 m (dashed lines: 30 m and 50 m). "
                 "The train spike at 60 m+ comes from the few broken trips.", width=PAGE_W),
          say("The data was clean, but in a few trips the answer can't explain the GPS. We leave those dots out "
              "instead of guessing.")]

    # ---------------------------------------------------------------- preprocessing
    s += [PageBreak(), P("3. Preprocessing in three steps", H2),
          P("Preprocessing turns the raw files into one clean table: <b>one row for every (GPS dot, nearby road) "
            "pair</b>, with a column saying whether that road is the correct one."),
          P("Step 1: Convert positions to metres", H3),
          P("GPS gives degrees, but here a degree east is about 72 km while a degree north is about 111 km. "
            "Measuring in degrees would distort every distance and angle. We convert each trip to metres around its "
            "own centre. Over a trip this is accurate to better than 0.1%, far smaller than the GPS error, and a unit "
            "test checks it against the exact (haversine) formula."),
          say("We measure everything in metres, not degrees, because degrees are stretched differently east-west "
              "and north-south."),
          P("Step 2: Give every dot its correct road (labelling)", H3),
          P("The route lists the roads in driving order but not which dot belongs to which road. We walk along the "
            "dots and the route together and match each dot to a route road, with one rule: <b>we only ever move "
            "forward along the route</b>, because the car never drives its route backwards. We also prefer matches "
            "where the distance moved along the route equals the distance the GPS moved."),
          P("Why not just take the nearest route road? On a two-way street both directions are equally near, so "
            "\"nearest\" is a coin flip, and on a street driven twice it can pick the wrong visit. Moving forward "
            "fixes both. On average our labels differ from \"nearest\" for 5.9% of dots, and for 44.6% in a trip "
            "that drives into dead ends and back out."),
          P(f"Result: <b>{pct(tr.labelled_frac)}</b> of train dots and <b>{pct(st.labelled_frac)}</b> of test dots "
            "get a label. Dots more than 50 m from every route road stay unlabelled."),
          say("We matched dots to the route in order, never going backwards, so we get the direction right on "
              "two-way streets.")]
    s += [PageBreak(), P("Step 3: Shortlist the roads near each dot (candidate generation)", H3),
          P("For each dot we draw a <b>50 m circle</b> and list every road inside it, keeping at most the "
            "<b>12 closest</b>. These are the candidates. The model's job is to pick the right road from this "
            f"shortlist (about {tr.mean_candidates_per_point:.0f} roads per dot on average) instead of searching a map "
            "that can have 300,000 road pieces. Both directions of a street are always kept or dropped together."),
          P("We chose 50 m and 12 by testing many sizes. A small circle sometimes leaves the true road out (a "
            "\"miss\"); a big one makes the list long for no benefit."),
          table([["Setting", "Train: true road missing", "Test: true road missing", "Roads per dot (train)"],
                 ["50 m, keep 8", "0.88%", "0.24%", "5.9"],
                 ["<b>50 m, keep 12 (chosen)</b>", f"<b>{100 * tr.candidate_miss_rate:.2f}%</b>",
                  f"<b>{100 * st.candidate_miss_rate:.2f}%</b>", f"<b>{tr.mean_candidates_per_point:.1f}</b>"],
                 ["50 m, keep 16", "0.48%", "0.03%", "8.1"]],
                [4.4 * cm, 3.9 * cm, 3.9 * cm, PAGE_W - 12.2 * cm]),
          Spacer(1, 6),
          figure("candidates_miss_rate.png",
                 "Left: how often the true road is missing, by circle size, for each shortlist length. "
                 "Right: the same against how many roads we keep. Beyond 12 the gain is tiny.", width=PAGE_W),
          P(f"Output: {int(tr.points):,} train dots become <b>{int(tr.candidate_rows):,}</b> (dot, road) rows; "
            f"{int(st.candidate_rows):,} for the test trips."),
          say(f"For each point we shortlist roads within 50 m, at most 12, so the model chooses from about "
              f"{tr.mean_candidates_per_point:.0f} options. The true road is on the list "
              f"{100 * (1 - tr.candidate_miss_rate):.1f}% of the time.")]

    # ---------------------------------------------------------------- features
    s += [PageBreak(), P("4. Features: describing every road on the shortlist", H2),
          P("A <b>feature</b> is a number that answers a question about one road on a dot's shortlist. Each row of "
            "our table is one (dot, road) pair with 26 such numbers, plus the answer column <i>is_true</i>. About "
            "1 row in 7 is the correct road."),
          P("A real example: one dot and its shortlist", H3),
          P(f"Dot {int(ex.point_idx.iloc[0])} of test trip {ex.trip.iloc[0]:08d} has {len(ex)} roads within 50 m:")]
    rows = [["Road id", "Correct?", "Distance (m)", "Rank", "Heading match", "Neighbours' distance (m)",
             "Past road's end?"]]
    for r in ex.itertuples():
        mark = lambda v: f"<b>{v}</b>" if r.is_true else str(v)
        rows.append([mark(r.arc), mark("Yes" if r.is_true else "No"), mark(f"{r.dist:.2f}"), mark(int(r.rank)),
                     mark(f"{r.heading_cos:+.2f}"), mark(f"{r.ctx_dist_mean:.2f}"), mark("yes" if r.t_at_end else "no")])
    s += [table(rows, [1.7 * cm, 2.0 * cm, 2.2 * cm, 1.3 * cm, 2.3 * cm, 3.4 * cm, PAGE_W - 12.9 * cm]),
          Spacer(1, 6),
          P("The first two roads are <b>the same street in opposite directions</b>: identical distance (6.64 m), so "
            "distance cannot choose. <b>Heading breaks the tie</b>: +1 means the road points the way the car is "
            "moving, −1 the opposite way. The correct road is the +1 one. The others are further away and the "
            "dot lies past their ends.")]
    s += [P("The six questions we ask about each road", H3),
          table([["Question", "Features (26 in total)", "Why it helps"],
                 ["How close is it?", "dist, rank, dist_minus_min, dist_ratio_min, dist_z, n_candidates, gap_1st_2nd",
                  "The true road is usually the closest or nearly so. Comparing with the other roads at the same dot "
                  "works better than raw distance, because GPS noise differs between trips."],
                 ["Does it point the way the car is going?",
                  "heading_cos, heading_absdiff, heading_cos_prev, heading_cos_next, heading_rank, heading_valid",
                  "The only way to separate the two directions of a two-way street. The car's direction comes from "
                  "the 2 dots before and after. If the car barely moved it is marked unknown."],
                 ["Do the neighbouring dots agree?", "ctx_dist_mean, ctx_dist_max, ctx_minus_min, ctx_rank",
                  "One dot can wobble; the right road also fits the dots around it."],
                 ["Is the dot past the end of this road?", "t, t_at_end, arc_len",
                  "If the dot lands beyond the road's end, the car is probably on the next road."],
                 ["Is the car moving normally?", "speed, dt_prev, dt_next",
                  "When the car is stopped or the GPS paused, direction and neighbours are less reliable."],
                 ["What is the road like?", "from_in_degree, to_out_degree, has_reverse_twin",
                  "Mistakes happen most at junctions where many roads meet."]],
                [3.5 * cm, 5.6 * cm, PAGE_W - 9.1 * cm]),
          Spacer(1, 6)]
    s += bullets([
        "<b>No feature ever reads the answer file.</b> A unit test checks this, so the same code runs unchanged on "
        "the organisers' hidden trips (no leakage).",
        "<b>Relative features</b> (\"distance minus the closest road's distance\", rank) let the model compare a "
        "road with its rivals, even though it scores rows one at a time.",
        "<b>Heading as a cosine:</b> +1 same direction, −1 opposite, with no jump at 0/360 degrees.",
        "<b>No missing values:</b> when heading is unknown we use a neutral value plus a flag, so every model can "
        "use every row."])
    s += [say("For each nearby road we ask how close it is, whether it points the way the car is going, whether "
              "the neighbouring points agree, and whether the car is moving normally. Direction matters most, because "
              "two-way streets are stored twice.")]

    # ---------------------------------------------------------------- evidence
    s += [PageBreak(), P("5. Proof that the features work", H2),
          P("Before training any model, we tested simple hand-written rules on our own test trips. "
            "<b>Top-1 accuracy</b> = the share of dots where the rule's first choice is the correct road."),
          table([["Rule (no learning)", "Train", "Test trips"],
                 ["Pick the closest road", pct(near_tr), pct(near_st)],
                 ["Pick the closest road that points the way the car is going", pct(nh_tr), pct(nh_st)],
                 ["Best possible (true road is on the shortlist)", f"{100 * (1 - tr.candidate_miss_rate):.1f}%",
                  f"{100 * (1 - st.candidate_miss_rate):.1f}%"]],
                [10 * cm, 2.6 * cm, PAGE_W - 12.6 * cm]),
          Spacer(1, 4),
          P(f"\"Closest road\" gets only about half right because both directions of a street are equally close. "
            f"Adding direction lifts it to {pct(nh_st)}. <b>The models we train next must beat this.</b>"),
          figure("feature_univariate_auc.png",
                 "How well each feature alone separates the correct road from the others (ROC-AUC; 0.5 = no better "
                 "than chance, 1.0 = perfect). Relative distance and context are strongest (about 0.9); heading is "
                 "about 0.83.", width=PAGE_W * 0.52),
          P("Weak-looking features (speed, time gaps, junction size) score near 0.5 alone. We keep them because "
            "their job is to tell the model <i>when</i> to trust the strong features, which a single-feature score "
            "cannot show.")]
    s += [figure("feature_distributions.png",
                 "Blue = the correct road, red = other roads on the shortlist, for the eight strongest features. "
                 "The correct road sits almost always at 0 (closest) on the relative-distance features.",
                 width=PAGE_W),
          say(f"Distance alone gets about half right, because two-way streets are stored twice. Adding direction "
              f"takes it to {pct(nh_st)}, and that is the baseline our models must beat.")]

    # ---------------------------------------------------------------- Q&A + numbers
    s += [PageBreak(), P("6. Viva answers in one or two sentences", H2)]
    qa = [
        ("What's in the dataset?", "GPS dots, a road map (points and road pieces) and the true route for each trip."),
        ("What's the tricky part?", "The route doesn't say which dot belongs to which road, so we work that out."),
        ("Why are roads duplicated?", "A two-way street is stored once per direction; the two copies overlap exactly."),
        ("How did you avoid cheating?", "Train, test and evaluator trips are kept separate, and features never read "
                                        "the answer file."),
        ("Why convert to metres?", "Degrees are not equal east-west and north-south, so distances would be wrong."),
        ("How did you label each dot?", "By walking along the dots and the route together, only moving forward, "
                                        "matching each dot to the best-fitting route road."),
        ("Why not the nearest road?", "On two-way streets both directions are equally near, so it guesses the "
                                      "direction at random."),
        ("What is candidate generation?", "A shortlist of roads within 50 m of each dot, at most 12, so the model "
                                          "picks from a few options instead of the whole map."),
        ("Why 50 m and 12?", "We tested many sizes; this keeps the true road on the list 99.4% of the time with "
                             "about 7 roads per dot."),
        ("What about bad data?", "Dots the route can't explain get no label; gaps are given to the model as a feature."),
        ("What is a feature here?", "A number describing one road on a dot's shortlist, like its distance or "
                                    "whether it points the car's way."),
        ("Most useful feature?", "Distance compared with the other nearby roads; and direction, which separates the "
                                 "two sides of a street."),
        ("Why look at neighbouring dots?", "One dot can wobble; the right road fits the dots around it too."),
        ("Why keep speed and gaps?", "They tell the model when direction and neighbours can be trusted."),
    ]
    s += [table([["#", "Question", "Answer"]] + [[str(i + 1), q, a] for i, (q, a) in enumerate(qa)],
                [0.8 * cm, 4.6 * cm, PAGE_W - 5.4 * cm])]
    s += [Spacer(1, 10), P("7. Numbers to remember", H2),
          table([["Number", "Meaning"],
                 ["80 / 10 / 10", "Trips for training / our testing / the organisers' demo"],
                 ["about 5 m", "Typical GPS error (dot to true road)"],
                 ["50 m, 12 roads", "Shortlist: roads within 50 m, at most 12 per dot"],
                 [f"{100 * (1 - tr.candidate_miss_rate):.1f}%", "How often the true road is on the shortlist (train)"],
                 [f"{pct(tr.labelled_frac)} / {pct(st.labelled_frac)}", "Dots that get a label (train / test)"],
                 [f"{int(tr.candidate_rows):,}", "(dot, road) rows in the training table"],
                 ["26", "Features per row, in 6 question groups"],
                 [f"{pct(near_st)} → {pct(nh_st)}", "Correct road picked by 'closest' vs 'closest + right "
                                                          "direction' (test trips)"]],
                [4 * cm, PAGE_W - 4 * cm]),
          Spacer(1, 8),
          P("All numbers come from <font face='Courier'>results/logs/</font> in the repository and are regenerated "
            "by <font face='Courier'>scripts/eda.py</font>, <font face='Courier'>scripts/build_dataset.py</font> and "
            "<font face='Courier'>scripts/build_features.py</font>.", CAP)]

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=2 * cm, title="PathSnap: dataset, preprocessing and features",
                            author="PathSnap team")
    doc.build(s, onFirstPage=on_page, onLaterPages=on_page)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
