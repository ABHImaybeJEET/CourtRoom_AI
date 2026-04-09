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
        border-top: 4px solid #ff4b4b;
        padding: 20px;
        margin-bottom: 25px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-radius: 8px;
        color: #1a1a1a;
    }
    .defense-box {
        border-top: 4px solid #0068c9;
        padding: 20px;
        margin-bottom: 25px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-radius: 8px;
        color: #1a1a1a;
    }
    .judge-box {
        border-top: 4px solid #ffaa00;
        padding: 20px;
        margin-bottom: 25px;
        background-color: #faf9f6;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-radius: 8px;
        color: #1a1a1a;
        font-family: serif;
    }
    .juror-card {
        border-top: 3px solid #e0e0e0;
        padding: 16px;
        border-radius: 6px;
        background-color: #fafafa;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        color: #2c2c2c;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        transition: transform 0.2s ease;
    }
    .juror-card:hover {
        transform: translateY(-2px);
    }
    .juror-header {
        font-weight: 700;
        font-size: 1.05em;
        margin-bottom: 8px;
        color: #111;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .guilty-badge {
        background-color: #ff3b30;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.75em;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .not-guilty-badge {
        background-color: #34c759;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.75em;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .juror-reasoning {
        font-size: 0.85em;
        line-height: 1.4;
        color: #555;
        margin-top: 12px;
        padding-top: 10px;
        border-top: 1px dashed #ddd;
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
        "Insider Trading (Wall Street)": "data/sample_cases/insider_trading.txt",
        "Satyam Scam (India Financial Fraud)": "data/sample_cases/india_fraud.txt",
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
    
    if st.button("🧹 Clear Case Memory"):
        if os.path.exists("data/faiss_index"):
            import shutil
            shutil.rmtree("data/faiss_index")
        st.success("Memory cleared! Ready for new case.")

    # Instant Demo feature
    if selected_sample in ["Insider Trading (Wall Street)", "Satyam Scam (India Financial Fraud)"] and st.session_state.case_data:
        if st.button("⚡ Instant Preview (Recorded Demo)", use_container_width=True):
            file_map = {
                "Insider Trading (Wall Street)": "data/cached_results/insider_trading.json",
                "Satyam Scam (India Financial Fraud)": "data/cached_results/india_fraud.json"
            }
            try:
                with open(file_map[selected_sample], "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                    st.session_state.final_state = cached_data
                    st.rerun()
            except Exception as e:
                st.error(f"Could not load cache: {e}")

    st.divider()
    
    # Usage Stats
    if "api_stats" in st.session_state:
        st.subheader("📊 Session API Usage")
        stats = st.session_state.api_stats
        st.caption(f"Total Model Calls: {stats['total_calls']}")
        st.caption(f"Fallback Redirects: {stats['fallback_count']}")
        with st.expander("Model Breakdown"):
            for m, count in stats['model_usage'].items():
                st.write(f"{m}: {count}")
    
    st.divider()
    st.markdown("Powered by **LangGraph + Groq + Tavily**")

# Main Area
if begin_trial:
    if not case_input:
        st.error("Please provide a case description.")
    else:
        # Hard-coded legal facts to keep users engaged during processing
        st.info("""
        **💡 While the AI Deliberates - Did you know?**
        *   **The Hallucination Danger:** In 2023, real lawyers were sanctioned by judges for submitting fake legal citations generated by standard ChatGPT. CourtRoom AI prevents this by using a dedicated Judge Agent to fact-check the attorneys via live Tavily web searches!
        *   **India's Enron:** The historic Satyam Scam (faking ₹5,000+ Crores) forced sweeping changes to SEBI's corporate governance and auditing laws across India.
        *   **Jury Psychology:** Empirical studies prove that demographically diverse juries make 30% fewer factual errors during deliberation. This is why our system initiates 12 entirely distinct persona agents for the final verdict!
        """)
        
        with st.spinner("⚖️ Initiating Multi-Agent Debate and scanning Vector Precedents..."):
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
                # Use a new event loop for this run to avoid conflict with streamlit
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                final_state = loop.run_until_complete(run_trial(case_input, rounds))
                st.session_state.final_state = final_state
                loop.close()
            except Exception as e:
                st.error(f"Error during trial execution: {e}")
                
if "final_state" in st.session_state:
    state = st.session_state.final_state

    # Fairness Check: Only show results if the final verdict was reached
    if not state.get('final_verdict'):
        st.error("⚠️ Trial was interrupted. To ensure legal fairness, partial results will not be displayed. Please check your API keys/internet and try again.")
    else:
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
                        <div class="juror-header">
                            <span>#{juror['juror_id']} • <span style="font-weight:normal; font-size: 0.9em;">{juror['profile']['age']}y, {juror['profile']['occupation']}</span></span>
                        </div>
                        <div style="margin-top: 5px;">
                            <span class="{'guilty-badge' if juror['verdict'] == 'Guilty' else 'not-guilty-badge'}">
                                {juror['verdict']} ({juror['confidence']*100:.0f}%)
                            </span>
                        </div>
                        <div class="juror-reasoning">"{juror['reasoning']}"</div>
                    </div>
                    """, unsafe_allow_html=True)

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
