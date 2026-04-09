import os
import sys
from pathlib import Path

# MANDATORY: Add project root to sys.path before any local imports
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import asyncio
import io
from dotenv import load_dotenv

# Now we can import from our local packages
from graph.courtroom_graph import create_courtroom_graph
from rag.ingestion import ingest_pdf, ingest_text
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
import plotly.graph_objects as go
import json

load_dotenv()

st.set_page_config(page_title="CourtRoom AI", layout="wide")

# Custom CSS for courtroom theme
st.markdown("""
<style>
    .prosecution-box {
        border-left: 5px solid #ff4b4b;
        padding: 15px;
        margin-bottom: 20px;
        background-color: #fff5f5;
        border-radius: 5px;
        color: #1a1a1a;
    }
    .defense-box {
        border-left: 5px solid #0068c9;
        padding: 15px;
        margin-bottom: 20px;
        background-color: #f0f7ff;
        border-radius: 5px;
        color: #1a1a1a;
    }
    .judge-box {
        border-left: 5px solid #ffaa00;
        padding: 15px;
        margin-bottom: 20px;
        background-color: #fffdf0;
        border-radius: 5px;
        color: #1a1a1a;
    }
    .juror-card {
        border: 1px solid #ddd;
        padding: 10px;
        border-radius: 8px;
        background-color: #f9f9f9;
        text-align: center;
        color: #1a1a1a;
    }
    .guilty-badge {
        background-color: #ff4b4b;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .not-guilty-badge {
        background-color: #28a745;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_pdf(state):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=LETTER)
    styles = getSampleStyleSheet()
    
    title_style = styles['Heading1']
    subtitle_style = styles['Heading2']
    normal_style = styles['Normal']
    
    Story = []
    
    Story.append(Paragraph("CourtRoom AI - Trial Transcript", title_style))
    Story.append(Spacer(1, 12))
    
    Story.append(Paragraph(f"<b>Case details:</b> {state.get('case_description', '')}", normal_style))
    Story.append(Spacer(1, 12))
    
    Story.append(Paragraph(f"<b>Final Verdict:</b> {state.get('final_verdict', 'N/A')}", subtitle_style))
    Story.append(Spacer(1, 24))
    
    Story.append(Paragraph("Prosecution's Argument", subtitle_style))
    prosecution_text = state.get('prosecution_argument', {}).get('argument_text', 'N/A')
    Story.append(Paragraph(str(prosecution_text).replace('\n', '<br/>'), normal_style))
    Story.append(Spacer(1, 12))
    
    Story.append(Paragraph("Defense's Argument", subtitle_style))
    defense_text = state.get('defense_argument', {}).get('argument_text', 'N/A')
    Story.append(Paragraph(str(defense_text).replace('\n', '<br/>'), normal_style))
    Story.append(Spacer(1, 12))
    
    judge_scores = state.get('judge_scores', {})
    Story.append(Paragraph("Judge's Summary", subtitle_style))
    Story.append(Paragraph(str(judge_scores.get('reasoning_summary', 'N/A')), normal_style))
    Story.append(Spacer(1, 12))
    
    Story.append(Paragraph("Jury Deliberation Analysis", subtitle_style))
    Story.append(Paragraph(str(state.get('demographic_analysis', 'N/A')), normal_style))
    
    doc.build(Story)
    buffer.seek(0)
    return buffer

async def run_trial(case_desc, max_rounds):
    app = create_courtroom_graph()
    initial_state = {
        "case_description": case_desc,
        "case_documents": [],
        "retrieved_context": "",
        "web_search_results": "",
        "prosecution_argument": {},
        "defense_argument": {},
        "judge_scores": {},
        "hallucination_flags": [],
        "jury_profiles": [],
        "jury_verdicts": [],
        "final_verdict": "",
        "round_number": 0,
        "max_rounds": max_rounds,
        "debate_history": [],
        "vote_count": {"guilty": 0, "not_guilty": 0},
        "demographic_analysis": ""
    }
    
    result = await app.ainvoke(initial_state)
    return result

# Sidebar
with st.sidebar:
    st.title("⚖️ CourtRoom AI")
    
    # Initialize session state for case input if not present
    if "case_data" not in st.session_state:
        st.session_state.case_data = ""

    case_input = st.text_area("Describe the case", 
                              value=st.session_state.case_data,
                              height=200, 
                              placeholder="Enter case details or load sample...")
    
    sample_cases = {
        "Insider Trading": "data/sample_cases/insider_trading.txt",
        "Corporate Fraud (Theranos config)": "data/sample_cases/corporate_fraud.txt",
        "IP Theft (Trade Secrets)": "data/sample_cases/ip_infringement.txt"
    }
    
    selected_sample = st.selectbox("Or choose a sample case:", list(sample_cases.keys()))
    
    if st.button("Load Selected Sample"):
        try:
            with open(sample_cases[selected_sample], "r", encoding="utf-8") as f:
                content = f.read()
                st.session_state.case_data = content
                st.rerun()
        except FileNotFoundError:
            st.error(f"Sample case file not found at {sample_cases[selected_sample]}")

    uploaded_files = st.file_uploader("Upload case documents (PDF)", type=["pdf"], accept_multiple_files=True)
    rounds = st.slider("Argument rounds", min_value=1, max_value=3, value=2)
    
    begin_trial = st.button("🏛️ Begin Trial", use_container_width=True)
    st.divider()
    st.markdown("Powered by **LangGraph + Groq + Tavily**")

# Main Area
if begin_trial:
    if not case_input:
        st.error("Please provide a case description.")
    else:
        with st.spinner("⚖️ Trial in progress..."):
            # Handle PDF ingestion
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    path = os.path.join("data", uploaded_file.name)
                    with open(path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    ingest_pdf(path)
            else:
                ingest_text(case_input)

            # Run graph
            try:
                final_state = asyncio.run(run_trial(case_input, rounds))
                st.session_state.final_state = final_state
            except Exception as e:
                st.error(f"Error during trial execution: {e}")
                
if "final_state" in st.session_state:
    state = st.session_state.final_state

    st.header("📋 Live Trial Feed")
    
    # Check if we have valid outputs
    prosecution = state.get('prosecution_argument', {})
    if isinstance(prosecution, dict) and "argument_text" in prosecution:
        # Prosecution
        st.markdown(f"""
        <div class="prosecution-box">
            <h3>Prosecution Argument</h3>
            <p>{prosecution.get('argument_text', 'N/A')}</p>
            <strong>Cited Sources:</strong> {", ".join(prosecution.get('cited_sources', []))}
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prosecution Confidence", f"{prosecution.get('confidence_score', 0)*100:.0f}%")
        with col2:
            st.caption(f"Strategy: {prosecution.get('legal_strategy', 'N/A')}")
    else:
        st.warning("Prosecution argument missing or malformed.")

    defense = state.get('defense_argument', {})
    if isinstance(defense, dict) and "argument_text" in defense:
        # Defense
        st.markdown(f"""
        <div class="defense-box">
            <h3>Defense Argument</h3>
            <p>{defense.get('argument_text', 'N/A')}</p>
            <strong>Cited Sources:</strong> {", ".join(defense.get('cited_sources', []))}
        </div>
        """, unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        with col3:
            st.metric("Defense Confidence", f"{defense.get('confidence_score', 0)*100:.0f}%")
        with col4:
            st.caption(f"Strategy: {defense.get('legal_strategy', 'N/A')}")
    else:
        st.warning("Defense argument missing or malformed.")

    # Judge
    st.divider()
    st.subheader("👨‍⚖️ Judge Scorecard")
    st.markdown(f"""
    <div class="judge-box">
        <h4>Preliminary Ruling: {state['judge_scores'].get('preliminary_ruling', 'N/A').replace('_', ' ').title()}</h4>
        <p>{state['judge_scores'].get('reasoning_summary', '')}</p>
    </div>
    """, unsafe_allow_html=True)

    j_col1, j_col2 = st.columns(2)
    with j_col1:
        st.write("**Prosecution Scores**")
        scores = state['judge_scores'].get('prosecution_scores', {})
        for k, v in scores.items():
            st.metric(str(k).replace('_', ' ').title(), v)
            
    with j_col2:
        st.write("**Defense Scores**")
        scores = state['judge_scores'].get('defense_scores', {})
        for k, v in scores.items():
            st.metric(str(k).replace('_', ' ').title(), v)

    if state.get('hallucination_flags'):
        for flag in state['hallucination_flags']:
            st.warning(f"⚠️ **Citation Alert ({flag['side']})**: {flag['citation']} - {flag['reason']}")

    # Jury
    st.divider()
    st.subheader("👥 The Jury Box")
    
    # 3 rows of 4 jurors
    for i in range(0, 12, 4):
        cols = st.columns(4)
        for j in range(4):
            juror_idx = i + j
            if juror_idx < len(state['jury_verdicts']):
                juror = state['jury_verdicts'][juror_idx]
                with cols[j]:
                    st.markdown(f"""
                    <div class="juror-card">
                        <b>Juror #{juror['juror_id']}</b><br/>
                        <small>{juror['profile']['age']}y, {juror['profile']['occupation']}</small><br/>
                        <span class="{'guilty-badge' if juror['verdict'] == 'Guilty' else 'not-guilty-badge'}">
                            {juror['verdict']}
                        </span><br/>
                        <small>Confidence: {juror['confidence']*100:.0f}%</small>
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption(juror['reasoning'])

    v_col1, v_col2 = st.columns(2)
    vote_count = state.get('vote_count', {'guilty': 0, 'not_guilty': 0})
    v_col1.metric("Guilty Votes", vote_count.get('guilty', 0))
    v_col2.metric("Not Guilty Votes", vote_count.get('not_guilty', 0))
    
    st.info(f"**Demographic Analysis:** {state.get('demographic_analysis', 'N/A')}")

    # Final Verdict
    st.divider()
    final_verdict = state.get('final_verdict', 'N/A')
    if final_verdict == "Guilty":
        st.error(f"⚖️ FINAL VERDICT: {final_verdict}")
    elif final_verdict == "Not Guilty":
        st.success(f"⚖️ FINAL VERDICT: {final_verdict}")
    else:
        st.warning(f"⚖️ FINAL VERDICT: {final_verdict}")

    # Download
    pdf_file = generate_pdf(state)
    st.download_button(
        label="📄 Download Trial Transcript",
        data=pdf_file,
        file_name="courtroom_transcript.pdf",
        mime="application/pdf"
    )
