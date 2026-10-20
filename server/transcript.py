import io
import html
from datetime import datetime
from reportlab.lib.pagesizes import LETTER
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def escape_txt(val: any) -> str:
    if val is None:
        return "N/A"
    return html.escape(str(val))


def build_trial_pdf(state: dict) -> io.BytesIO:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "CourtTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
        alignment=1,  # Center
    )

    meta_style = ParagraphStyle(
        "MetaStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#64748b"),
        alignment=1,
    )

    heading2 = ParagraphStyle(
        "CourtH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=14,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "CourtBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#1e293b"),
    )

    verdict_guilty = ParagraphStyle(
        "VerdictGuilty",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#b91c1c"),
        alignment=1,
    )

    verdict_not_guilty = ParagraphStyle(
        "VerdictNotGuilty",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#15803d"),
        alignment=1,
    )

    story = []

    # Title & Header
    story.append(Paragraph("COURTROOM AI &bull; OFFICIAL TRIAL TRANSCRIPT", title_style))
    trial_id = state.get("trial_id", "DEMO-RUN")
    date_str = datetime.now().strftime("%B %d, %Y - %H:%M:%S UTC")
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"Trial ID: {escape_txt(trial_id)} &nbsp;|&nbsp; Certified Recorded: {date_str}", meta_style))
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#cbd5e1"), spaceAfter=14))

    # Final Verdict Banner
    final_verdict = state.get("final_verdict", "Undecided")
    verdict_style = verdict_guilty if "guilty" in final_verdict.lower() else verdict_not_guilty
    story.append(Paragraph(f"FINAL RULING: {escape_txt(final_verdict).upper()}", verdict_style))
    vote_count = state.get("vote_count", {})
    story.append(
        Paragraph(
            f"Jury Vote Tally: <b>{vote_count.get('guilty', 0)} Guilty</b> vs <b>{vote_count.get('not_guilty', 0)} Not Guilty</b>",
            meta_style,
        )
    )
    story.append(Spacer(1, 14))

    # Case Overview
    story.append(Paragraph("1. Case Overview & Facts", heading2))
    case_desc = escape_txt(state.get("case_description", "No case description provided.")).replace("\n", "<br/>")
    story.append(Paragraph(case_desc, body_style))
    story.append(Spacer(1, 10))

    # Arguments Section
    story.append(Paragraph("2. Adversarial Legal Arguments", heading2))

    prosecution = state.get("prosecution_argument", {})
    p_text = escape_txt(prosecution.get("argument_text", "No argument recorded.")).replace("\n", "<br/>")
    p_strat = escape_txt(prosecution.get("legal_strategy", "N/A"))
    p_conf = int(prosecution.get("confidence_score", 0) * 100)
    p_sources = ", ".join(escape_txt(s) for s in prosecution.get("cited_sources", [])) or "None cited"

    story.append(Paragraph(f"<b>Prosecution Argument</b> (Strategy: <i>{p_strat}</i> | Confidence: {p_conf}%)", body_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(p_text, body_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"<b>Cited Authorities:</b> {p_sources}", meta_style))
    story.append(Spacer(1, 12))

    defense = state.get("defense_argument", {})
    d_text = escape_txt(defense.get("argument_text", "No argument recorded.")).replace("\n", "<br/>")
    d_strat = escape_txt(defense.get("legal_strategy", "N/A"))
    d_conf = int(defense.get("confidence_score", 0) * 100)
    d_sources = ", ".join(escape_txt(s) for s in defense.get("cited_sources", [])) or "None cited"

    story.append(Paragraph(f"<b>Defense Argument</b> (Strategy: <i>{d_strat}</i> | Confidence: {d_conf}%)", body_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(d_text, body_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"<b>Cited Authorities:</b> {d_sources}", meta_style))
    story.append(Spacer(1, 14))

    # Judge Evaluation
    judge_scores = state.get("judge_scores", {})
    ruling = escape_txt(judge_scores.get("preliminary_ruling", "N/A")).replace("_", " ").title()
    summary = escape_txt(judge_scores.get("reasoning_summary", "No judge evaluation.")).replace("\n", "<br/>")

    story.append(Paragraph("3. Presiding Judge Review & Scores", heading2))
    story.append(Paragraph(f"<b>Preliminary Judicial Finding:</b> {ruling}", body_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(summary, body_style))
    story.append(Spacer(1, 8))

    # Judge Scorecard Table
    p_scores = judge_scores.get("prosecution_scores", {})
    d_scores = judge_scores.get("defense_scores", {})
    categories = sorted(list(set(list(p_scores.keys()) + list(d_scores.keys()))))
    
    if categories:
        table_data = [["Scoring Dimension", "Prosecution (0-10)", "Defense (0-10)"]]
        for cat in categories:
            dim_name = cat.replace("_", " ").title()
            table_data.append([dim_name, str(p_scores.get(cat, "-")), str(d_scores.get(cat, "-"))])
        
        t = Table(table_data, colWidths=[200, 150, 150])
        t.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 8.5),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ])
        )
        story.append(t)
        story.append(Spacer(1, 14))

    # Citation Verification Audit
    flags = state.get("hallucination_flags", [])
    if flags:
        story.append(Paragraph("<b>Citation Verification Audit Warnings:</b>", body_style))
        for flag in flags:
            citation_name = escape_txt(flag.get("citation", "Unknown"))
            reason = escape_txt(flag.get("reason", "Not verified"))
            side = escape_txt(flag.get("side", "General"))
            story.append(Paragraph(f"&bull; <b>[{side.upper()}] {citation_name}:</b> {reason}", meta_style))
        story.append(Spacer(1, 10))

    # Jury Deliberation Panel
    jury_verdicts = state.get("jury_verdicts", [])
    if jury_verdicts:
        story.append(Paragraph("4. Individual Jury Deliberation Roster", heading2))
        j_data = [["#", "Juror Profile", "Verdict", "Conf.", "Reasoning"]]
        for j in jury_verdicts:
            j_id = str(j.get("juror_id", "-"))
            prof = j.get("profile") or {}
            prof_str = f"{prof.get('age', '—')}y, {prof.get('occupation', 'Citizen')} ({prof.get('political_lean', '')})"
            v = escape_txt(j.get("verdict", "N/A"))
            c = f"{int(j.get('confidence', 0.5) * 100)}%"
            r = escape_txt(j.get("reasoning", "No comment."))
            j_data.append([
                Paragraph(j_id, meta_style),
                Paragraph(prof_str, meta_style),
                Paragraph(f"<b>{v}</b>", meta_style),
                Paragraph(c, meta_style),
                Paragraph(r, meta_style),
            ])

        jt = Table(j_data, colWidths=[20, 140, 60, 40, 240])
        jt.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 8.5),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ])
        )
        story.append(jt)
        story.append(Spacer(1, 14))

    # Demographic Analysis
    demo_analysis = escape_txt(state.get("demographic_analysis", "")).replace("\n", "<br/>")
    if demo_analysis and demo_analysis != "N/A":
        story.append(Paragraph("5. Jury Demographic Analysis", heading2))
        story.append(Paragraph(demo_analysis, body_style))
        story.append(Spacer(1, 14))

    doc.build(story)
    buffer.seek(0)
    return buffer
