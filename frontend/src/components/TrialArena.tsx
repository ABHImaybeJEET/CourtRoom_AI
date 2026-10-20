import React, { useState } from 'react'
import type { TrialState } from '../types'
import {
  Scale,
  Shield,
  Gavel,
  Download,
  AlertTriangle,
  CheckCircle,
  FileText,
  Clock,
  ChevronDown,
  ChevronUp,
} from 'lucide-react'

interface TrialArenaProps {
  state: TrialState
  isStreaming: boolean
  currentStage: string
  onReset: () => void
}

const STAGES = [
  { key: 'INITIALIZATION', label: 'Case Intake' },
  { key: 'EVIDENCE_RETRIEVAL', label: 'Evidence & Search' },
  { key: 'PROSECUTION_ARGUMENT', label: 'Prosecution Argument' },
  { key: 'DEFENSE_ARGUMENT', label: 'Defense Rebuttal' },
  { key: 'JUDICIAL_REVIEW', label: 'Judicial Scoring' },
  { key: 'JURY_DELIBERATION', label: 'Jury Box' },
  { key: 'CONCLUDED', label: 'Final Verdict' },
]

export const TrialArena: React.FC<TrialArenaProps> = ({ state, isStreaming, currentStage, onReset }) => {
  const [expandedJuror, setExpandedJuror] = useState<number | null>(null)
  const [activeTab, setActiveTab] = useState<'arguments' | 'judge' | 'jury' | 'evidence'>('arguments')

  const prosecution = state.prosecution_argument
  const defense = state.defense_argument
  const judgeScores = state.judge_scores
  const juryVerdicts = state.jury_verdicts || []
  const voteCount = state.vote_count || { guilty: 0, not_guilty: 0 }
  const finalVerdict = state.final_verdict
  const hallucinationFlags = state.hallucination_flags || []

  const isGuilty = finalVerdict && finalVerdict.toLowerCase().includes('guilty') && !finalVerdict.toLowerCase().includes('not')

  return (
    <div className="animate-fade" style={{ maxWidth: '1440px', margin: '30px auto', padding: '0 24px' }}>
      {/* Top Banner: Trial Header & Export Actions */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '16px',
        marginBottom: '24px',
        paddingBottom: '20px',
        borderBottom: '1px solid var(--border-subtle)',
      }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '4px' }}>
            <span className="badge badge-gold">
              <Clock size={12} /> {isStreaming ? 'PROCEEDINGS IN PROGRESS' : 'TRIAL RECORDED'}
            </span>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
              Trial ID: {state.trial_id || 'ACTIVE'}
            </span>
          </div>
          <h2 className="font-serif" style={{ fontSize: '1.8rem', fontWeight: 900 }}>
            Courtroom Chamber &bull; Trial Deliberation
          </h2>
        </div>

        {/* Action Controls */}
        <div style={{ display: 'flex', gap: '10px' }}>
          {state.trial_id && (
            <>
              <a
                href={`/api/trials/${state.trial_id}/export/pdf`}
                className="btn btn-primary"
                download
                title="Download certified trial transcript in PDF format"
              >
                <Download size={15} /> Download PDF Transcript
              </a>
              <a
                href={`/api/trials/${state.trial_id}/export/json`}
                className="btn btn-secondary"
                download
                title="Export complete trial state JSON"
              >
                <FileText size={15} /> Export JSON
              </a>
            </>
          )}
          <button className="btn btn-secondary" onClick={onReset}>
            New Trial
          </button>
        </div>
      </div>

      {/* Stepper / Timeline */}
      <div className="glass-panel" style={{ padding: '16px 20px', marginBottom: '28px' }}>
        <div style={{
          display: 'grid',
          gridTemplateColumns: `repeat(${STAGES.length}, 1fr)`,
          gap: '8px',
          alignItems: 'center',
        }}>
          {STAGES.map((s, idx) => {
            const isCurrent = currentStage === s.key
            const isCompleted = STAGES.findIndex(item => item.key === currentStage) > idx || currentStage === 'CONCLUDED'
            return (
              <div key={s.key} style={{ textAlign: 'center' }}>
                <div style={{
                  height: '4px',
                  borderRadius: '2px',
                  background: isCompleted
                    ? 'var(--gold-500)'
                    : isCurrent
                    ? 'linear-gradient(90deg, var(--gold-500), transparent)'
                    : 'rgba(255, 255, 255, 0.1)',
                  marginBottom: '8px',
                }} />
                <span style={{
                  fontSize: '0.72rem',
                  fontWeight: isCurrent ? 700 : 500,
                  color: isCurrent ? '#fbbf24' : isCompleted ? 'var(--text-main)' : 'var(--text-muted)',
                  display: 'block',
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                }}>
                  {s.label}
                </span>
              </div>
            )
          })}
        </div>
      </div>

      {/* Final Verdict Banner (If Concluded) */}
      {finalVerdict && (
        <div
          className="glass-panel animate-fade"
          style={{
            padding: '24px 30px',
            marginBottom: '32px',
            background: isGuilty
              ? 'linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(15, 23, 42, 0.8))'
              : 'linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(15, 23, 42, 0.8))',
            borderColor: isGuilty ? 'rgba(239, 68, 68, 0.4)' : 'rgba(34, 197, 94, 0.4)',
            boxShadow: isGuilty ? '0 0 35px rgba(239, 68, 68, 0.2)' : '0 0 35px rgba(34, 197, 94, 0.2)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: '20px',
          }}
        >
          <div>
            <span style={{ fontSize: '0.8rem', fontWeight: 700, letterSpacing: '0.08em', color: isGuilty ? '#fca5a5' : '#86efac' }}>
              OFFICIAL VERDICT RENDERED BY THE JURY
            </span>
            <h2 className="font-serif" style={{ fontSize: '2.4rem', fontWeight: 900, color: isGuilty ? '#ef4444' : '#22c55e', margin: '4px 0' }}>
              {finalVerdict.toUpperCase()}
            </h2>
            <p style={{ fontSize: '0.92rem', color: 'var(--text-dim)', maxWidth: '650px' }}>
              {state.demographic_analysis || 'Jury panel has completed deliberation across demographic segments.'}
            </p>
          </div>

          <div style={{ display: 'flex', gap: '20px', alignItems: 'center' }}>
            <div style={{ textAlign: 'center', padding: '12px 20px', borderRadius: 'var(--radius-md)', background: 'rgba(239, 68, 68, 0.12)', border: '1px solid rgba(239, 68, 68, 0.3)' }}>
              <span style={{ fontSize: '1.6rem', fontWeight: 900, color: '#f87171' }}>{voteCount.guilty}</span>
              <span style={{ display: 'block', fontSize: '0.75rem', fontWeight: 700, color: '#fca5a5' }}>GUILTY VOTES</span>
            </div>
            <div style={{ textAlign: 'center', padding: '12px 20px', borderRadius: 'var(--radius-md)', background: 'rgba(34, 197, 94, 0.12)', border: '1px solid rgba(34, 197, 94, 0.3)' }}>
              <span style={{ fontSize: '1.6rem', fontWeight: 900, color: '#4ade80' }}>{voteCount.not_guilty}</span>
              <span style={{ display: 'block', fontSize: '0.75rem', fontWeight: 700, color: '#86efac' }}>NOT GUILTY</span>
            </div>
          </div>
        </div>
      )}

      {/* Navigation Tabs */}
      <div style={{ display: 'flex', gap: '8px', marginBottom: '24px', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '12px' }}>
        {[
          { id: 'arguments', label: 'Adversarial Arguments', count: null },
          { id: 'judge', label: 'Judge Scorecard & Citations', count: hallucinationFlags.length ? `${hallucinationFlags.length} Alert` : null },
          { id: 'jury', label: '12-Juror Deliberation Box', count: `${juryVerdicts.length} Jurors` },
          { id: 'evidence', label: 'Case Context & Precedent', count: null },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`btn ${activeTab === tab.id ? 'btn-primary' : 'btn-secondary'}`}
            style={{ fontSize: '0.86rem', padding: '8px 18px' }}
          >
            {tab.label}
            {tab.count && (
              <span style={{
                background: 'rgba(0, 0, 0, 0.3)',
                padding: '1px 6px',
                borderRadius: '999px',
                fontSize: '0.7rem',
                marginLeft: '4px',
              }}>
                {tab.count}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* TAB 1: Adversarial Arguments */}
      {activeTab === 'arguments' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '28px' }}>
          {/* Prosecution Podium */}
          <div className="glass-panel" style={{
            padding: '28px',
            borderTop: '4px solid #ef4444',
            background: 'linear-gradient(180deg, rgba(239, 68, 68, 0.05), rgba(15, 23, 42, 0.7))',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <div style={{ padding: '8px', borderRadius: '8px', background: 'rgba(239, 68, 68, 0.15)' }}>
                  <Shield size={20} color="#ef4444" />
                </div>
                <div>
                  <h3 style={{ fontSize: '1.2rem', fontWeight: 800 }}>Prosecution Argument</h3>
                  <span style={{ fontSize: '0.78rem', color: 'var(--text-dim)' }}>
                    Strategy: <b style={{ textTransform: 'capitalize', color: '#fca5a5' }}>{prosecution?.legal_strategy || 'Formulating...'}</b>
                  </span>
                </div>
              </div>
              {prosecution && (
                <div style={{ textAlign: 'right' }}>
                  <span style={{ fontSize: '1.2rem', fontWeight: 800, color: '#f87171' }}>
                    {Math.round((prosecution.confidence_score || 0) * 100)}%
                  </span>
                  <span style={{ display: 'block', fontSize: '0.7rem', color: 'var(--text-muted)' }}>CONFIDENCE</span>
                </div>
              )}
            </div>

            <div style={{ minHeight: '220px', fontSize: '0.94rem', lineHeight: 1.7, color: 'var(--text-main)', marginBottom: '20px' }}>
              {prosecution?.argument_text ? (
                prosecution.argument_text.split('\n\n').map((para, i) => (
                  <p key={i} style={{ marginBottom: '14px' }}>{para}</p>
                ))
              ) : (
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-muted)', paddingTop: '40px' }}>
                  <Clock className="animate-spin" size={18} /> Preparing prosecution opening statement...
                </div>
              )}
            </div>

            {/* Cited Authorities */}
            {prosecution?.cited_sources && prosecution.cited_sources.length > 0 && (
              <div style={{ borderTop: '1px solid rgba(255, 255, 255, 0.08)', paddingTop: '16px' }}>
                <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-dim)', display: 'block', marginBottom: '8px' }}>
                  CITED AUTHORITIES &amp; STATUTES:
                </span>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                  {prosecution.cited_sources.map((src, i) => (
                    <span key={i} className="badge badge-prosecution" style={{ fontSize: '0.72rem' }}>
                      {src}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Defense Podium */}
          <div className="glass-panel" style={{
            padding: '28px',
            borderTop: '4px solid #3b82f6',
            background: 'linear-gradient(180deg, rgba(59, 130, 246, 0.05), rgba(15, 23, 42, 0.7))',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <div style={{ padding: '8px', borderRadius: '8px', background: 'rgba(59, 130, 246, 0.15)' }}>
                  <Scale size={20} color="#3b82f6" />
                </div>
                <div>
                  <h3 style={{ fontSize: '1.2rem', fontWeight: 800 }}>Defense Rebuttal</h3>
                  <span style={{ fontSize: '0.78rem', color: 'var(--text-dim)' }}>
                    Strategy: <b style={{ textTransform: 'capitalize', color: '#93c5fd' }}>{defense?.legal_strategy || 'Formulating...'}</b>
                  </span>
                </div>
              </div>
              {defense && (
                <div style={{ textAlign: 'right' }}>
                  <span style={{ fontSize: '1.2rem', fontWeight: 800, color: '#60a5fa' }}>
                    {Math.round((defense.confidence_score || 0) * 100)}%
                  </span>
                  <span style={{ display: 'block', fontSize: '0.7rem', color: 'var(--text-muted)' }}>CONFIDENCE</span>
                </div>
              )}
            </div>

            <div style={{ minHeight: '220px', fontSize: '0.94rem', lineHeight: 1.7, color: 'var(--text-main)', marginBottom: '20px' }}>
              {defense?.argument_text ? (
                defense.argument_text.split('\n\n').map((para, i) => (
                  <p key={i} style={{ marginBottom: '14px' }}>{para}</p>
                ))
              ) : (
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-muted)', paddingTop: '40px' }}>
                  <Clock className="animate-spin" size={18} /> Awaiting defense response to prosecution brief...
                </div>
              )}
            </div>

            {/* Cited Authorities */}
            {defense?.cited_sources && defense.cited_sources.length > 0 && (
              <div style={{ borderTop: '1px solid rgba(255, 255, 255, 0.08)', paddingTop: '16px' }}>
                <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-dim)', display: 'block', marginBottom: '8px' }}>
                  CITED AUTHORITIES &amp; SAFE HARBORS:
                </span>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                  {defense.cited_sources.map((src, i) => (
                    <span key={i} className="badge badge-defense" style={{ fontSize: '0.72rem' }}>
                      {src}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* TAB 2: Judge Scorecard */}
      {activeTab === 'judge' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 0.9fr', gap: '28px' }}>
          {/* Ruling & Scorecard */}
          <div className="glass-panel" style={{ padding: '28px', borderTop: '4px solid #f59e0b' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '18px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <Gavel size={22} color="#f59e0b" />
                <h3 className="font-serif" style={{ fontSize: '1.3rem', fontWeight: 800 }}>Presiding Judge Assessment</h3>
              </div>
              {judgeScores?.preliminary_ruling && (
                <span className="badge badge-gold" style={{ fontSize: '0.8rem' }}>
                  Finding: {judgeScores.preliminary_ruling.replace('_', ' ').toUpperCase()}
                </span>
              )}
            </div>

            <p style={{ fontSize: '0.94rem', lineHeight: 1.7, color: 'var(--text-dim)', marginBottom: '24px' }}>
              {judgeScores?.reasoning_summary || 'Judicial evaluation pending presentation of counsel arguments.'}
            </p>

            {/* Scorecard Dimensions */}
            {judgeScores && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                {['legal_soundness', 'evidence_quality', 'logical_coherence', 'precedent_accuracy'].map((dim) => {
                  const pVal = judgeScores.prosecution_scores?.[dim] ?? 5
                  const dVal = judgeScores.defense_scores?.[dim] ?? 5
                  return (
                    <div key={dim} style={{ background: 'rgba(255, 255, 255, 0.02)', padding: '12px 16px', borderRadius: 'var(--radius-sm)' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.84rem', fontWeight: 600, marginBottom: '6px' }}>
                        <span>{dim.replace('_', ' ').toUpperCase()}</span>
                        <div style={{ display: 'flex', gap: '16px' }}>
                          <span style={{ color: '#f87171' }}>Prosecution: {pVal}/10</span>
                          <span style={{ color: '#60a5fa' }}>Defense: {dVal}/10</span>
                        </div>
                      </div>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                        <div style={{ height: '6px', background: 'rgba(239, 68, 68, 0.2)', borderRadius: '3px', overflow: 'hidden' }}>
                          <div style={{ height: '100%', width: `${pVal * 10}%`, background: '#ef4444' }} />
                        </div>
                        <div style={{ height: '6px', background: 'rgba(59, 130, 246, 0.2)', borderRadius: '3px', overflow: 'hidden' }}>
                          <div style={{ height: '100%', width: `${dVal * 10}%`, background: '#3b82f6' }} />
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </div>

          {/* Citation Grounding Audit */}
          <div className="glass-panel" style={{ padding: '28px' }}>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 800, marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Shield size={18} color="#38bdf8" /> Citation Verification Audit
            </h3>
            <p style={{ fontSize: '0.84rem', color: 'var(--text-muted)', marginBottom: '18px' }}>
              Authorities cited by each side are independently checked against external legal search indexes to identify unsupported precedents.
            </p>

            {hallucinationFlags.length === 0 ? (
              <div style={{
                padding: '20px',
                borderRadius: 'var(--radius-md)',
                background: 'rgba(34, 197, 94, 0.08)',
                border: '1px solid rgba(34, 197, 94, 0.25)',
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
              }}>
                <CheckCircle size={24} color="#22c55e" />
                <div>
                  <h4 style={{ fontSize: '0.9rem', color: '#4ade80' }}>Zero Hallucination Flags</h4>
                  <p style={{ fontSize: '0.78rem', color: 'var(--text-dim)', margin: 0 }}>
                    All cited case authorities and statutes verified against precedent databases.
                  </p>
                </div>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                {hallucinationFlags.map((flag, idx) => (
                  <div key={idx} style={{
                    padding: '14px',
                    borderRadius: 'var(--radius-sm)',
                    background: 'rgba(239, 68, 68, 0.08)',
                    border: '1px solid rgba(239, 68, 68, 0.25)',
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
                      <AlertTriangle size={15} color="#ef4444" />
                      <span style={{ fontSize: '0.85rem', fontWeight: 700, color: '#f87171' }}>
                        [{flag.side.toUpperCase()}] {flag.citation}
                      </span>
                    </div>
                    <p style={{ fontSize: '0.78rem', color: 'var(--text-dim)', margin: 0 }}>
                      {flag.reason}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* TAB 3: 12-Juror Deliberation Box */}
      {activeTab === 'jury' && (
        <div>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '20px' }}>
            <div>
              <h3 className="font-serif" style={{ fontSize: '1.4rem', fontWeight: 800 }}>Simulated Jury Deliberation Panel</h3>
              <p style={{ fontSize: '0.86rem', color: 'var(--text-dim)' }}>
                Each juror deliberates according to distinct age, career background, empathy quotient, and skepticism metrics.
              </p>
            </div>
            <div style={{ display: 'flex', gap: '12px' }}>
              <span className="badge badge-guilty" style={{ fontSize: '0.8rem', padding: '6px 14px' }}>
                {voteCount.guilty} Guilty
              </span>
              <span className="badge badge-not-guilty" style={{ fontSize: '0.8rem', padding: '6px 14px' }}>
                {voteCount.not_guilty} Not Guilty
              </span>
            </div>
          </div>

          {/* 3x4 Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '18px' }}>
            {juryVerdicts.map((juror) => {
              const isExpanded = expandedJuror === juror.juror_id
              const isJGuilty = juror.verdict === 'Guilty' || juror.verdict === 'Liable'
              const profile = juror.profile || {}
              return (
                <div
                  key={juror.juror_id}
                  className="glass-panel glass-panel-interactive"
                  style={{
                    padding: '18px',
                    borderTop: `3px solid ${isJGuilty ? '#ef4444' : '#22c55e'}`,
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '8px' }}>
                    <div>
                      <span style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--gold-400)' }}>
                        JUROR #{juror.juror_id}
                      </span>
                      <h4 style={{ fontSize: '0.98rem', fontWeight: 700 }}>
                        {profile.occupation || 'Citizen'}
                      </h4>
                      <span style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>
                        {profile.age || '—'} yrs &bull; {profile.political_lean || 'Independent'}
                      </span>
                    </div>

                    <span className={isJGuilty ? 'badge badge-guilty' : 'badge badge-not-guilty'}>
                      {juror.verdict} ({Math.round(juror.confidence * 100)}%)
                    </span>
                  </div>

                  {/* Trait Gauges */}
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', margin: '10px 0', fontSize: '0.74rem' }}>
                    <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: '6px 8px', borderRadius: '4px' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Empathy:</span>{' '}
                      <b>{profile.empathy_score || 5}/10</b>
                    </div>
                    <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: '6px 8px', borderRadius: '4px' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Skepticism:</span>{' '}
                      <b>{profile.skepticism_score || 5}/10</b>
                    </div>
                  </div>

                  {/* Reasoning */}
                  <div style={{ borderTop: '1px dashed rgba(255, 255, 255, 0.1)', paddingTop: '10px', marginTop: '10px' }}>
                    <p style={{
                      fontSize: '0.82rem',
                      color: 'var(--text-dim)',
                      lineHeight: 1.5,
                      fontStyle: 'italic',
                      display: '-webkit-box',
                      WebkitLineClamp: isExpanded ? 'unset' : 3,
                      WebkitBoxOrient: 'vertical',
                      overflow: 'hidden',
                    }}>
                      "{juror.reasoning}"
                    </p>
                    <button
                      onClick={() => setExpandedJuror(isExpanded ? null : juror.juror_id)}
                      style={{
                        background: 'transparent',
                        border: 'none',
                        color: 'var(--gold-400)',
                        fontSize: '0.72rem',
                        fontWeight: 600,
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '4px',
                        marginTop: '6px',
                        padding: 0,
                      }}
                    >
                      {isExpanded ? <><ChevronUp size={12} /> Show Less</> : <><ChevronDown size={12} /> Read Full Deliberation</>}
                    </button>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* TAB 4: Case Context & Evidence */}
      {activeTab === 'evidence' && (
        <div className="glass-panel" style={{ padding: '28px' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 800, marginBottom: '14px' }}>
            Retrieved Case Evidence &amp; Precedent Grounding
          </h3>
          <div style={{ marginBottom: '24px' }}>
            <h4 style={{ fontSize: '0.88rem', color: 'var(--gold-400)', marginBottom: '8px' }}>
              CASE DESCRIPTION &amp; ALLEGATIONS
            </h4>
            <div style={{ background: 'rgba(0, 0, 0, 0.3)', padding: '16px', borderRadius: 'var(--radius-sm)', fontSize: '0.9rem', lineHeight: 1.6 }}>
              {state.case_description || 'None provided.'}
            </div>
          </div>

          <div>
            <h4 style={{ fontSize: '0.88rem', color: '#38bdf8', marginBottom: '8px' }}>
              VECTOR STORE RETRIEVED CONTEXT
            </h4>
            <div style={{ background: 'rgba(0, 0, 0, 0.3)', padding: '16px', borderRadius: 'var(--radius-sm)', fontSize: '0.85rem', lineHeight: 1.6, color: 'var(--text-dim)' }}>
              {state.retrieved_context ? (
                <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'var(--font-mono)' }}>
                  {state.retrieved_context}
                </pre>
              ) : (
                'No external documents ingested.'
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
