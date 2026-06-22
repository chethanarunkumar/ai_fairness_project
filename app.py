import os, io, json, tempfile
from flask import Flask, render_template, request, jsonify, send_file
from ml_engine import run_audit, run_audit_from_df, DATASET_CFG, BASE
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT

app = Flask(__name__)

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index(): return render_template("index.html")

@app.route("/dashboard")
def dashboard(): return render_template("dashboard.html")

# ── FIX 1 & 2: Get dataset info (columns + real row count) ───────────────────
@app.route("/dataset_info", methods=["POST"])
def dataset_info():
    """Return all column names + real row count for a demo dataset."""
    data    = request.get_json()
    ds_name = data.get("dataset", "Adult Income")
    cfg     = DATASET_CFG.get(ds_name)
    if not cfg:
        return jsonify({"error": "Unknown dataset"}), 400
    try:
        df      = pd.read_csv(os.path.join(BASE, cfg["file"]))
        columns = df.columns.tolist()
        return jsonify({
            "columns":   columns,
            "row_count": len(df),
            "label_col": cfg["label"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── FIX 3-6: Analyse uploaded CSV columns ────────────────────────────────────
@app.route("/upload_info", methods=["POST"])
def upload_info():
    """Parse uploaded CSV, return columns + row count."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    try:
        sep = "\t" if f.filename.endswith(".tsv") else ","
        df  = pd.read_csv(f, sep=sep, nrows=5000)   # preview first 5k for col detection
        # Guess numeric vs categorical
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
        # Save full file to temp for later audit
        f.seek(0)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv",
                                          dir=os.path.join(BASE, "uploads"))
        f.save(tmp.name)
        return jsonify({
            "columns":     df.columns.tolist(),
            "num_cols":    num_cols,
            "cat_cols":    cat_cols,
            "row_count":   "preview (up to 5000)",
            "tmp_path":    os.path.basename(tmp.name),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Main audit endpoint ───────────────────────────────────────────────────────
@app.route("/audit", methods=["POST"])
def audit():
    # Uploaded file audit
    if request.content_type and "multipart" in request.content_type:
        f          = request.files.get("file")
        sensitive  = request.form.get("sensitive",  "")
        label_col  = request.form.get("label_col",  "")
        mitigation = request.form.get("mitigation", "reweighing")
        model_name = request.form.get("model_name", "XGBoost")
        if not f:
            return jsonify({"error": "No file"}), 400
        try:
            sep = "\t" if f.filename.endswith(".tsv") else ","
            df  = pd.read_csv(f, sep=sep)
            result = run_audit_from_df(df, sensitive, label_col, mitigation, model_name)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Demo dataset audit (JSON)
    data       = request.get_json()
    dataset    = data.get("dataset",    "Adult Income")
    sensitive  = data.get("sensitive",  "gender")
    mitigation = data.get("mitigation", "reweighing")
    model_name = data.get("model_name", "XGBoost")
    try:
        result = run_audit(dataset, sensitive, mitigation, model_name)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── FIX 7: Export PDF ─────────────────────────────────────────────────────────
@app.route("/export_pdf", methods=["POST"])
def export_pdf():
    data = request.get_json()
    buf  = io.BytesIO()
    _build_pdf(data, buf)
    buf.seek(0)
    fname = f"fairness_audit_{data.get('dataset','report').replace(' ','_')}.pdf"
    return send_file(buf, mimetype="application/pdf",
                     as_attachment=True, download_name=fname)

def _build_pdf(d, buf):
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm,  bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    NAVY  = colors.HexColor("#1a2a4a")
    TEAL  = colors.HexColor("#028090")
    MINT  = colors.HexColor("#02c39a")
    GREEN = colors.HexColor("#059669")
    RED   = colors.HexColor("#dc2626")
    AMBER = colors.HexColor("#d97706")
    LGRAY = colors.HexColor("#f4f7f9")
    MGRAY = colors.HexColor("#e0e7ef")

    title_style = ParagraphStyle("title", fontSize=22, textColor=NAVY,
                                  fontName="Helvetica-Bold", spaceAfter=12, spaceBefore=6, alignment=TA_CENTER)
    sub_style   = ParagraphStyle("sub",   fontSize=10, textColor=TEAL,
                                  fontName="Helvetica", spaceAfter=12, spaceBefore=4, alignment=TA_CENTER)
    h2_style    = ParagraphStyle("h2",    fontSize=13, textColor=NAVY,
                                  fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6)
    body_style  = ParagraphStyle("body",  fontSize=9,  textColor=colors.HexColor("#334155"),
                                  fontName="Helvetica", spaceAfter=4, leading=14)
    label_style = ParagraphStyle("lbl",   fontSize=8,  textColor=colors.HexColor("#64748b"),
                                  fontName="Helvetica-Bold")

    # Handle both normal audit and Full Dataset Scan
    is_fds = d.get("scan_type") == "Full Dataset Scan"
    fs    = d.get("avg_fairness", d.get("fairness_score", 0)) if is_fds else d.get("fairness_score", 0)
    v_txt = "FAIR" if fs>=75 else "MODERATE BIAS" if fs>=50 else "HIGH BIAS"
    v_col = GREEN if fs>=75 else AMBER if fs>=50 else RED

    story = []

    # ── Header ──
    story.append(Spacer(1, 8))
    story.append(Paragraph("AI FAIRNESS & BIAS AUDIT REPORT", title_style))
    story.append(Spacer(1, 6))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=2, color=TEAL, spaceAfter=12))

    # ── Verdict banner ──
    score_label = "Avg Fairness Score" if is_fds else "Fairness Score"
    scan_label  = f"  |  Full Dataset Scan ({d.get('n_folds',5)}-Fold CV)" if is_fds else ""
    verdict_data = [[Paragraph(f"OVERALL VERDICT: {v_txt}  |  {score_label}: {fs}/100{scan_label}", 
                               ParagraphStyle("verd", fontSize=13, textColor=colors.white,
                                              fontName="Helvetica-Bold", alignment=TA_CENTER))]]
    vt = Table(verdict_data, colWidths=["100%"])
    vt.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,-1), v_col),
                             ("ROUNDEDCORNERS", [6]),
                             ("TOPPADDING",(0,0),(-1,-1),10),
                             ("BOTTOMPADDING",(0,0),(-1,-1),10)]))
    story.append(vt)
    story.append(Spacer(1, 12))

    # ── Audit Info ──
    story.append(Paragraph("01 — AUDIT INFORMATION", h2_style))
    if is_fds:
        info_rows = [
            ["Dataset",       d.get("dataset","—"),                "Scan Type",    "Full Dataset Scan (5-Fold CV)"],
            ["Active Model",  d.get("model_name","—"),             "K-Folds",      str(d.get("n_folds",5))],
            ["Total Samples", str(d.get("n_samples","—")),         "Overall Acc",  f"{d.get('overall_accuracy',0)*100:.2f}% ± {d.get('accuracy_std',0)*100:.2f}%"],
            ["Avg Fairness",  f"{d.get('avg_fairness',0)}/100",    "Generated",    str(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M UTC"))],
        ]
    else:
        info_rows = [
            ["Dataset",          d.get("dataset","—"),        "Protected Attribute", d.get("sensitive_attr","—")],
            ["Active Model",     d.get("model_name","—"),     "AUC-ROC",             str(d.get("auc","—"))],
            ["Total Samples",    str(d.get("n_samples","—")), "Test Samples",        str(d.get("n_test","—"))],
            ["Mitigation",       d.get("mitigation","—"),     "Generated",           str(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M UTC"))],
        ]
    it = Table(info_rows, colWidths=[4*cm, 6*cm, 4*cm, 6*cm])
    it.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (0,-1), LGRAY), ("BACKGROUND", (2,0), (2,-1), LGRAY),
        ("FONTNAME",     (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME",     (2,0), (2,-1), "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 8.5),
        ("TEXTCOLOR",    (0,0), (0,-1), NAVY), ("TEXTCOLOR", (2,0),(2,-1), NAVY),
        ("GRID",         (0,0), (-1,-1), 0.5, MGRAY),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.white, LGRAY]),
        ("TOPPADDING",   (0,0), (-1,-1), 6), ("BOTTOMPADDING",(0,0),(-1,-1),6),
        ("LEFTPADDING",  (0,0), (-1,-1), 8), ("RIGHTPADDING", (0,0),(-1,-1),8),
    ]))
    story.append(it)
    story.append(Spacer(1, 10))

    # ── Fairness Metrics ──
    story.append(Paragraph("02 — FAIRNESS METRICS", h2_style))
    di_ok  = 0.8 <= d.get("disparate_impact",0) <= 1.2
    eo_val = round(1 - abs(d.get("equalized_odds",0)), 3)
    pp_val = round(1 - abs(d.get("predictive_parity",0)), 3)
    acc    = d.get("accuracy", 0)

    def verdict_cell(ok): return ("PASS ✓" if ok else "FAIL ✗")
    def vc(ok): return GREEN if ok else RED

    m_headers = [["Metric","Value","Threshold","Status"]]
    m_rows = [
        ["Disparate Impact",  f"{d.get('disparate_impact',0):.3f}", "0.80 – 1.20", verdict_cell(di_ok)],
        ["Equal Opportunity", f"{eo_val:.3f}",                      "≥ 0.80",       verdict_cell(eo_val>=0.8)],
        ["Predictive Parity", f"{pp_val:.3f}",                      "≥ 0.80",       verdict_cell(pp_val>=0.8)],
        ["Model Accuracy",    f"{acc*100:.1f}%",                    "≥ 75%",        verdict_cell(acc>=0.75)],
        ["Dem. Parity Diff",  f"{d.get('demographic_parity',0):.4f}","< 0.05",      verdict_cell(abs(d.get('demographic_parity',0))<0.05)],
        ["Eq. Odds Diff",     f"{d.get('equalized_odds',0):.4f}",   "< 0.05",       verdict_cell(abs(d.get('equalized_odds',0))<0.05)],
    ]
    mt = Table(m_headers + m_rows, colWidths=[5.5*cm, 3.5*cm, 4*cm, 4*cm])
    mt_style = [
        ("BACKGROUND",   (0,0), (-1,0), NAVY), ("TEXTCOLOR", (0,0),(-1,0), colors.white),
        ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE",(0,0),(-1,-1),8.5),
        ("GRID",         (0,0), (-1,-1), 0.5, MGRAY),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, LGRAY]),
        ("TOPPADDING",   (0,0), (-1,-1), 6), ("BOTTOMPADDING",(0,0),(-1,-1),6),
        ("LEFTPADDING",  (0,0), (-1,-1), 8), ("RIGHTPADDING", (0,0),(-1,-1),8),
        ("ALIGN",        (1,0), (-1,-1), "CENTER"),
    ]
    for i, row in enumerate(m_rows, start=1):
        ok = row[3].startswith("PASS")
        mt_style.append(("TEXTCOLOR", (3,i),(3,i), GREEN if ok else RED))
        mt_style.append(("FONTNAME",  (3,i),(3,i), "Helvetica-Bold"))
    mt.setStyle(TableStyle(mt_style))
    story.append(mt)
    story.append(Spacer(1,10))

    # ── Group Breakdown ──
    story.append(Paragraph("03 — GROUP BREAKDOWN", h2_style))
    g_headers = [["Group","Accuracy","Positive Rate","TPR","FPR","FNR"]]
    g_rows = [[d["group_labels"][i],
               f"{d['group_accuracy'][i]*100:.1f}%",
               f"{d['group_pos_rate'][i]*100:.1f}%",
               f"{d['group_tpr'][i]*100:.1f}%",
               f"{d['group_fpr'][i]*100:.1f}%",
               f"{d['group_fnr'][i]*100:.1f}%"]
              for i in range(len(d.get("group_labels",[])))]
    gt = Table(g_headers + g_rows, colWidths=[4*cm,3*cm,3.5*cm,2.5*cm,2.5*cm,2.5*cm])
    gt.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,0), TEAL), ("TEXTCOLOR",(0,0),(-1,0), colors.white),
        ("FONTNAME",   (0,0),(-1,0), "Helvetica-Bold"), ("FONTSIZE",(0,0),(-1,-1),8.5),
        ("GRID",       (0,0),(-1,-1), 0.5, MGRAY),
        ("FONTNAME",   (0,1),(0,-1), "Helvetica-Bold"), ("TEXTCOLOR",(0,1),(0,-1), NAVY),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, LGRAY]),
        ("TOPPADDING", (0,0),(-1,-1),6), ("BOTTOMPADDING",(0,0),(-1,-1),6),
        ("LEFTPADDING",(0,0),(-1,-1),8), ("RIGHTPADDING", (0,0),(-1,-1),8),
        ("ALIGN",      (1,0),(-1,-1),"CENTER"),
    ]))
    story.append(gt)
    story.append(Spacer(1,10))

    # ── Full Dataset Scan: Per-Attribute Fairness Table ──
    if is_fds and d.get("attr_results"):
        story.append(Paragraph("02 — FAIRNESS SCAN — ALL ATTRIBUTES", h2_style))
        attr_headers = [["Attribute", "Fairness Score", "Disparate Impact", "Group 0", "Group 1", "Bias Level"]]
        attr_rows_pdf = []
        for a in d.get("attr_results", []):
            bl = a.get("bias_level", "—")
            bl_col = GREEN if bl=="FAIR" else AMBER if bl=="MODERATE" else RED
            attr_rows_pdf.append([
                a.get("attribute","—"),
                f"{a.get('fairness_score',0)}/100",
                f"{a.get('disparate_impact',0):.3f}",
                a.get("group_labels",["—","—"])[0],
                a.get("group_labels",["—","—"])[1],
                bl,
            ])
        at = Table(attr_headers + attr_rows_pdf,
                   colWidths=[3.5*cm, 3*cm, 3.5*cm, 3*cm, 3*cm, 3*cm])
        at_style = [
            ("BACKGROUND", (0,0),(-1,0), NAVY), ("TEXTCOLOR",(0,0),(-1,0), colors.white),
            ("FONTNAME",   (0,0),(-1,0), "Helvetica-Bold"), ("FONTSIZE",(0,0),(-1,-1),8.5),
            ("GRID",       (0,0),(-1,-1), 0.5, MGRAY),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, LGRAY]),
            ("TOPPADDING", (0,0),(-1,-1),6), ("BOTTOMPADDING",(0,0),(-1,-1),6),
            ("LEFTPADDING",(0,0),(-1,-1),8), ("RIGHTPADDING", (0,0),(-1,-1),8),
            ("ALIGN",      (1,0),(-1,-1),"CENTER"),
        ]
        for i, row in enumerate(attr_rows_pdf, start=1):
            bl = row[5]
            c = GREEN if bl=="FAIR" else AMBER if bl=="MODERATE" else RED
            at_style.append(("TEXTCOLOR", (5,i),(5,i), c))
            at_style.append(("FONTNAME",  (5,i),(5,i), "Helvetica-Bold"))
        at.setStyle(TableStyle(at_style))
        story.append(at)
        story.append(Spacer(1,10))

    # Skip normal fairness/group sections for FDS
    if is_fds:
        # Add K-Fold accuracy section
        story.append(Paragraph("03 — K-FOLD CROSS VALIDATION RESULTS", h2_style))
        fold_rows = [["Fold", "Accuracy"]] +                     [[f"Fold {i+1}", f"{a*100:.2f}%"] for i,a in enumerate(d.get("fold_accuracies",[]))]
        fold_rows.append(["Overall", f"{d.get('overall_accuracy',0)*100:.2f}% ± {d.get('accuracy_std',0)*100:.2f}%"])
        ft = Table(fold_rows, colWidths=[6*cm, 6*cm])
        ft.setStyle(TableStyle([
            ("BACKGROUND", (0,0),(-1,0), TEAL), ("TEXTCOLOR",(0,0),(-1,0), colors.white),
            ("FONTNAME",   (0,0),(-1,0), "Helvetica-Bold"), ("FONTSIZE",(0,0),(-1,-1),9),
            ("GRID",       (0,0),(-1,-1), 0.5, MGRAY),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, LGRAY]),
            ("BACKGROUND", (0,-1),(-1,-1), LGRAY),
            ("FONTNAME",   (0,-1),(-1,-1), "Helvetica-Bold"),
            ("TOPPADDING", (0,0),(-1,-1),6), ("BOTTOMPADDING",(0,0),(-1,-1),6),
            ("ALIGN",      (1,0),(-1,-1),"CENTER"),
        ]))
        story.append(ft)
        story.append(Spacer(1,10))
        doc.build(story)
        return

    # ── All 4 Models Comparison ──
    story.append(Paragraph("04 — ALL 4 MODELS COMPARISON", h2_style))
    mc_headers = [["Model","Accuracy","AUC-ROC","Fairness Score","Disp. Impact","Active"]]
    mc_rows = [[m["name"], f"{m['accuracy']*100:.1f}%", f"{m.get('auc',m['accuracy']):.3f}",
                f"{m['fairness_score']}/100", f"{m['disparate_impact']:.3f}",
                "✓ YES" if m["active"] else "—"]
               for m in d.get("models_comparison",[])]
    mct = Table(mc_headers + mc_rows, colWidths=[4.5*cm,3*cm,2.5*cm,3.5*cm,3*cm,2.5*cm])
    mct_style = [
        ("BACKGROUND", (0,0),(-1,0), NAVY), ("TEXTCOLOR",(0,0),(-1,0), colors.white),
        ("FONTNAME",   (0,0),(-1,0), "Helvetica-Bold"), ("FONTSIZE",(0,0),(-1,-1),8.5),
        ("GRID",       (0,0),(-1,-1), 0.5, MGRAY),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, LGRAY]),
        ("TOPPADDING", (0,0),(-1,-1),6), ("BOTTOMPADDING",(0,0),(-1,-1),6),
        ("LEFTPADDING",(0,0),(-1,-1),8), ("RIGHTPADDING", (0,0),(-1,-1),8),
        ("ALIGN",      (1,0),(-1,-1),"CENTER"),
    ]
    for i, m in enumerate(d.get("models_comparison",[]), start=1):
        if m["active"]:
            mct_style.append(("BACKGROUND",(0,i),(-1,i), colors.HexColor("#f0fdf4")))
            mct_style.append(("TEXTCOLOR", (5,i),(5,i), GREEN))
            mct_style.append(("FONTNAME",  (5,i),(5,i), "Helvetica-Bold"))
    mct.setStyle(TableStyle(mct_style))
    story.append(mct)
    story.append(Spacer(1,10))

    # ── Mitigation Results ──
    story.append(Paragraph("05 — BIAS MITIGATION RESULTS", h2_style))
    mit_headers = [["Metric","Before","After","Δ Change","Improved?"]]
    mit_data = [
        ["Accuracy",         d.get("accuracy",0),                   d.get("mitigated_accuracy",0),       True ],
        ["Dem. Parity Diff", abs(d.get("demographic_parity",0)),    abs(d.get("mitigated_dem_parity",0)),False],
        ["Eq. Odds Diff",    abs(d.get("equalized_odds",0)),        abs(d.get("mitigated_eq_odds",0)),   False],
        ["Disparate Impact", d.get("disparate_impact",0),           d.get("mitigated_disparate_impact",0),None],
        ["Fairness Score",   d.get("fairness_score",0)/100,        d.get("mitigated_fairness",0)/100,   True ],
    ]
    mit_rows = []
    for lbl,bef,aft,hib in mit_data:
        ok = (abs(1-aft)<abs(1-bef)) if hib is None else (aft>bef if hib else aft<bef)
        delta = (aft-bef)*100
        mit_rows.append([lbl, f"{bef*100:.2f}%", f"{aft*100:.2f}%",
                         f"{'▲' if delta>0 else '▼'} {abs(delta):.2f}%",
                         "✓ Yes" if ok else "✗ No"])
    mitt = Table(mit_headers + mit_rows, colWidths=[4.5*cm,3*cm,3*cm,3.5*cm,3*cm])
    mitt_style = [
        ("BACKGROUND", (0,0),(-1,0), TEAL), ("TEXTCOLOR",(0,0),(-1,0), colors.white),
        ("FONTNAME",   (0,0),(-1,0), "Helvetica-Bold"), ("FONTSIZE",(0,0),(-1,-1),8.5),
        ("GRID",       (0,0),(-1,-1), 0.5, MGRAY),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, LGRAY]),
        ("TOPPADDING", (0,0),(-1,-1),6), ("BOTTOMPADDING",(0,0),(-1,-1),6),
        ("LEFTPADDING",(0,0),(-1,-1),8), ("RIGHTPADDING", (0,0),(-1,-1),8),
        ("ALIGN",      (1,0),(-1,-1),"CENTER"),
    ]
    for i, (lbl,bef,aft,hib) in enumerate(mit_data, start=1):
        ok = (abs(1-aft)<abs(1-bef)) if hib is None else (aft>bef if hib else aft<bef)
        mitt_style.append(("TEXTCOLOR",(4,i),(4,i), GREEN if ok else RED))
        mitt_style.append(("FONTNAME", (4,i),(4,i), "Helvetica-Bold"))
    mitt.setStyle(TableStyle(mitt_style))
    story.append(mitt)
    story.append(Spacer(1,10))

    # ── Top Features ──
    story.append(Paragraph("06 — TOP PREDICTIVE FEATURES", h2_style))
    feat_headers = [["Rank","Feature","Importance Score","Bar"]]
    feat_rows = []
    for i, f in enumerate(d.get("top_features",[])[:8], start=1):
        bar = "█" * int(f["shap"]*100/5) if f["shap"] > 0 else ""
        feat_rows.append([str(i), f["name"], f"{f['shap']*100:.1f}%", bar])
    ft = Table(feat_headers + feat_rows, colWidths=[1.5*cm, 5*cm, 4*cm, 6.5*cm])
    ft.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,0), NAVY), ("TEXTCOLOR",(0,0),(-1,0), colors.white),
        ("FONTNAME",   (0,0),(-1,0), "Helvetica-Bold"), ("FONTSIZE",(0,0),(-1,-1),8.5),
        ("GRID",       (0,0),(-1,-1), 0.5, MGRAY),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, LGRAY]),
        ("TEXTCOLOR",  (3,1),(3,-1), TEAL), ("FONTNAME",(3,1),(3,-1),"Courier"),
        ("TOPPADDING", (0,0),(-1,-1),5), ("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",(0,0),(-1,-1),8), ("RIGHTPADDING", (0,0),(-1,-1),8),
        ("ALIGN",      (0,0),(2,-1),"CENTER"),
    ]))
    story.append(ft)
    story.append(Spacer(1,10))

    # ── Recommendations ──
    story.append(Paragraph("07 — RECOMMENDATIONS", h2_style))
    recs = []
    if fs < 75:
        recs += [
            f"⚠  Bias mitigation is REQUIRED before deploying this model.",
            f"   Detected significant disparity in '{d.get('sensitive_attr')}' attribute.",
            f"   Recommended action: Apply reweighing or adversarial debiasing.",
        ]
    else:
        recs += ["✅  Model meets minimum fairness thresholds for deployment."]
    recs += [
        "•  Schedule quarterly fairness re-audits to monitor drift.",
        "•  Align with EU AI Act fairness requirements.",
        "•  Document bias audit results for regulatory compliance.",
        "•  Cross-validate fairness across multiple sensitive attributes.",
        "•  Align with UN SDG 10 — Reduced Inequalities.",
    ]
    for r in recs:
        story.append(Paragraph(r, body_style))
    story.append(Spacer(1, 8))

    # ── Footer ──


    doc.build(story)

# ── Full Dataset Scan endpoint ───────────────────────────────────────────────
@app.route("/full_scan", methods=["POST"])
def full_scan():
    # Uploaded file full scan
    if request.content_type and "multipart" in request.content_type:
        f          = request.files.get("file")
        label_col  = request.form.get("label_col", "")
        mitigation = request.form.get("mitigation", "reweighing")
        model_name = request.form.get("model_name", "XGBoost")
        if not f:
            return jsonify({"error": "No file"}), 400
        try:
            from sklearn.model_selection import StratifiedKFold
            import numpy as np
            sep = "	" if f.filename.endswith(".tsv") else ","
            df  = pd.read_csv(f, sep=sep).dropna().reset_index(drop=True)
            if label_col not in df.columns:
                return jsonify({"error": f"Label column not found: {label_col}"}), 400

            # Prepare label
            from ml_engine import MODELS, _group_metrics, _fairness_metrics, _fairness_score
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, roc_auc_score

            y_raw = df[label_col]
            uniq  = sorted(y_raw.unique())
            lbl_map = {uniq[0]: 0, uniq[-1]: 1}
            y = y_raw.map(lbl_map).fillna(0).astype(int).values

            feat_cols = [c for c in df.columns if c != label_col]
            X = df[feat_cols].copy()
            for col in X.select_dtypes(exclude=["number"]).columns:
                X[col] = pd.Categorical(X[col]).codes.astype(float)
            X = X.fillna(X.median())

            if model_name not in MODELS:
                model_name = "XGBoost"

            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            fold_accs = []
            for tr_idx, te_idx in kf.split(X, y):
                Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
                ytr, yte = y[tr_idx], y[te_idx]
                sc = StandardScaler()
                m  = MODELS[model_name]()
                m.fit(sc.fit_transform(Xtr), ytr)
                fold_accs.append(accuracy_score(yte, m.predict(sc.transform(Xte))))

            overall_acc = round(float(np.mean(fold_accs)), 4)
            overall_std = round(float(np.std(fold_accs)), 4)

            # Per-column fairness scan
            sc2 = StandardScaler()
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
            m2 = MODELS[model_name]()
            m2.fit(sc2.fit_transform(X_tr), y_tr)
            y_pred2 = m2.predict(sc2.transform(X_te))

            attr_results = []
            for col in feat_cols[:10]:  # scan top 10 columns
                try:
                    col_data = df[col]
                    if col_data.nunique() == 2:
                        uq = sorted(col_data.unique())
                        sens_full = col_data.map({uq[0]:0, uq[1]:1}).fillna(0).astype(int)
                        lbl0, lbl1 = str(uq[0]), str(uq[1])
                    elif col_data.dtype in ["int64","float64", "int32", "float32"]:
                        med = col_data.median()
                        sens_full = (col_data >= med).astype(int)
                        lbl0, lbl1 = f"Below {med:.0f}", f"Above {med:.0f}"
                    else:
                        mf = col_data.value_counts().index[0]
                        sens_full = (col_data == mf).astype(int)
                        lbl0, lbl1 = f"Non-{mf}", str(mf)

                    _, _, _, _, s_tr2, s_te2 = train_test_split(
                        X, y, sens_full, test_size=0.25, random_state=42, stratify=y)
                    gm = _group_metrics(pd.Series(y_te), pd.Series(y_pred2), pd.Series(s_te2.values))
                    fm = _fairness_metrics(gm, 1, 0)
                    fs = _fairness_score(fm["demographic_parity"], fm["equalized_odds"], fm["disparate_impact"])
                    attr_results.append({
                        "attribute": col, "fairness_score": fs,
                        "disparate_impact": round(fm["disparate_impact"], 4),
                        "demographic_parity": round(fm["demographic_parity"], 4),
                        "equalized_odds": round(fm["equalized_odds"], 4),
                        "group_labels": [lbl0, lbl1],
                        "group_accuracy": [round(gm.get(0,{}).get("acc",0),4), round(gm.get(1,{}).get("acc",0),4)],
                        "group_pos_rate": [round(gm.get(0,{}).get("pos_rate",0),4), round(gm.get(1,{}).get("pos_rate",0),4)],
                        "bias_level": "FAIR" if fs>=75 else "MODERATE" if fs>=50 else "HIGH BIAS",
                    })
                except:
                    pass

            avg_fairness = round(float(np.mean([a["fairness_score"] for a in attr_results])), 1) if attr_results else 0

            from ml_engine import _real_shap, _real_lime, _build_xai, _perm_importance, _reweigh, _group_metrics, _fairness_metrics, _fairness_score
            Xtr_sc2 = sc2.fit_transform(X_tr); Xte_sc2 = sc2.transform(X_te)
            # SHAP
            try:
                shap_res,_,shap_bl=_real_shap(m2,Xtr_sc2,Xte_sc2,feat_cols,n_bg=15,n_samples=20)
                top_f=shap_res
            except:
                top_f=_perm_importance(m2,Xte_sc2,y_te,feat_cols); shap_bl=0.0
            # LIME
            try:
                lime_f=_real_lime(m2,Xtr_sc2,Xte_sc2[0],feat_cols,
                    feature_values_original=X_te.iloc[0].values,n_samples=80)
            except: lime_f=None
            # Mitigation
            try:
                best_a=min(attr_results,key=lambda a:a["fairness_score"]) if attr_results else {}
                bc=df[best_a["attribute"]] if best_a.get("attribute") and best_a["attribute"] in df.columns else pd.Series(np.zeros(len(df),dtype=int))
                sf=(bc>=bc.median()).astype(int) if bc.nunique()>2 else bc.map({sorted(bc.unique())[0]:0,sorted(bc.unique())[-1]:1}).fillna(0).astype(int)
                _,_,_,_,str2,ste2=train_test_split(X,y,sf,test_size=0.25,random_state=42,stratify=y)
                s_te_v=ste2.values; s_tr_v=str2.values
                gm_f=_group_metrics(pd.Series(y_te),pd.Series(y_pred2),pd.Series(s_te_v))
                fm_f=_fairness_metrics(gm_f,1,0)
                fs_f=_fairness_score(fm_f["demographic_parity"],fm_f["equalized_odds"],fm_f["disparate_impact"])
                wm=_reweigh(pd.Series(y_tr),pd.Series(s_tr_v))
                mm=MODELS[model_name]()
                try: mm.fit(Xtr_sc2,y_tr,sample_weight=wm)
                except: mm.fit(Xtr_sc2,y_tr)
                ypm=mm.predict(Xte_sc2); acm=round(accuracy_score(y_te,ypm),4)
                gm_m=_group_metrics(pd.Series(y_te),pd.Series(ypm),pd.Series(s_te_v))
                fm_m=_fairness_metrics(gm_m,1,0)
                fsm=_fairness_score(fm_m["demographic_parity"],fm_m["equalized_odds"],fm_m["disparate_impact"])
                g0=gm_f.get(0,{}); g1=gm_f.get(1,{}); g0m=gm_m.get(0,{}); g1m=gm_m.get(1,{})
            except:
                fm_f={"demographic_parity":0,"equalized_odds":0,"disparate_impact":1,"predictive_parity":0}
                fm_m=fm_f; gm_f={}; fs_f=avg_fairness; acm=overall_acc; fsm=avg_fairness
                g0={}; g1={}; g0m={}; g1m={}; best_a={}
            # XAI
            try:
                xai_f=_build_xai(top_f,fm_f,gm_f,feat_cols,
                    best_a.get("attribute",feat_cols[0] if feat_cols else "—"),
                    best_a.get("group_labels",["Group 0","Group 1"])[0],
                    best_a.get("group_labels",["Group 0","Group 1"])[1],
                    f.filename,model_name,overall_acc,fm_f["demographic_parity"],
                    lime_result=lime_f,shap_baseline=shap_bl)
            except: xai_f={}

            return jsonify({
                "dataset": f.filename, "model_name": model_name,
                "scan_type": "Full Dataset Scan",
                "n_samples": len(df), "n_folds": 5,
                "overall_accuracy": overall_acc, "overall_auc": overall_acc,
                "accuracy_std": overall_std,
                "fold_accuracies": [round(a,4) for a in fold_accs],
                "avg_fairness": avg_fairness,
                "overall_verdict": "FAIR" if avg_fairness>=75 else "MODERATE" if avg_fairness>=50 else "HIGH BIAS",
                "attr_results": attr_results, "mitigation": mitigation,
                "top_features": top_f, "n_features": len(feat_cols),
                "accuracy": overall_acc, "auc": overall_acc, "n_test": len(y_te),
                "fairness_score": fs_f, "sensitive_attr": best_a.get("attribute","—"),
                "demographic_parity": round(fm_f["demographic_parity"],4),
                "equalized_odds": round(fm_f["equalized_odds"],4),
                "disparate_impact": round(fm_f["disparate_impact"],4),
                "group_labels": best_a.get("group_labels",["Group 0","Group 1"]),
                "group_accuracy": [round(g0.get("acc",0),4),round(g1.get("acc",0),4)],
                "group_pos_rate": [round(g0.get("pos_rate",0),4),round(g1.get("pos_rate",0),4)],
                "group_tpr": [round(g0.get("tpr",0),4),round(g1.get("tpr",0),4)],
                "group_fpr": [round(g0.get("fpr",0),4),round(g1.get("fpr",0),4)],
                "group_fnr": [round(g0.get("fnr",0),4),round(g1.get("fnr",0),4)],
                "mitigated_accuracy": acm, "mitigated_fairness": fsm,
                "mitigated_dem_parity": round(fm_m["demographic_parity"],4),
                "mitigated_eq_odds": round(fm_m["equalized_odds"],4),
                "mitigated_disparate_impact": round(fm_m["disparate_impact"],4),
                "mitigated_group_accuracy": [round(g0m.get("acc",0),4),round(g1m.get("acc",0),4)],
                "mitigated_group_pos_rate": [round(g0m.get("pos_rate",0),4),round(g1m.get("pos_rate",0),4)],
                "xai": xai_f,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Demo dataset full scan
    data       = request.get_json()
    dataset    = data.get("dataset",    "Adult Income")
    mitigation = data.get("mitigation", "reweighing")
    model_name = data.get("model_name", "XGBoost")
    try:
        from ml_engine import run_full_dataset_scan
        result = run_full_dataset_scan(dataset, mitigation, model_name)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    os.makedirs(os.path.join(BASE, "uploads"), exist_ok=True)
    app.run(debug=True, port=5000)
