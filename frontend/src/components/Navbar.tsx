import React from 'react'
import { Scale, ShieldCheck, Sparkles, RefreshCw } from 'lucide-react'

interface NavbarProps {
  onReset: () => void
  onInstantDemo: (filename: string) => void
  health: { groq: boolean; tavily: boolean; langsmith: boolean } | null
}

export const Navbar: React.FC<NavbarProps> = ({ onReset, onInstantDemo, health }) => {
  return (
    <header style={{
      borderBottom: '1px solid var(--border-subtle)',
      background: 'rgba(8, 12, 20, 0.85)',
      backdropFilter: 'blur(16px)',
      position: 'sticky',
      top: 0,
      zIndex: 50,
      padding: '12px 24px',
    }}>
      <div style={{
        maxWidth: '1440px',
        margin: '0 auto',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '12px',
      }}>
        {/* Brand */}
        <div 
          onClick={onReset} 
          style={{ display: 'flex', alignItems: 'center', gap: '12px', cursor: 'pointer' }}
        >
          <div style={{
            background: 'linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(217, 119, 6, 0.1))',
            border: '1px solid var(--gold-500)',
            borderRadius: '12px',
            width: '42px',
            height: '42px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 0 15px rgba(245, 158, 11, 0.25)',
          }}>
            <Scale size={24} color="#f59e0b" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span className="font-serif" style={{ fontSize: '1.35rem', fontWeight: 900, letterSpacing: '0.04em' }}>
                CourtRoom <span style={{ color: '#f59e0b' }}>AI</span>
              </span>
              <span className="badge badge-gold" style={{ fontSize: '0.65rem', padding: '2px 8px' }}>v2.0 PRO</span>
            </div>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)', margin: 0 }}>
              Autonomous Legal Reasoning &amp; Jury Simulation
            </p>
          </div>
        </div>

        {/* API Health & Quick Demos */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap' }}>
          {health && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.78rem' }}>
              <span style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '5px',
                color: health.groq ? '#4ade80' : '#f87171',
                background: health.groq ? 'rgba(34, 197, 94, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                padding: '4px 10px',
                borderRadius: '6px',
                border: `1px solid ${health.groq ? 'rgba(34, 197, 94, 0.25)' : 'rgba(239, 68, 68, 0.25)'}`,
              }}>
                <ShieldCheck size={13} /> Groq LLM {health.groq ? 'Active' : 'Offline'}
              </span>
              <span style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '5px',
                color: health.tavily ? '#38bdf8' : '#94a3b8',
                background: 'rgba(56, 189, 248, 0.08)',
                padding: '4px 10px',
                borderRadius: '6px',
                border: '1px solid rgba(56, 189, 248, 0.2)',
              }}>
                Tavily Search {health.tavily ? 'Ready' : 'Mock'}
              </span>
            </div>
          )}

          {/* Quick Demos */}
          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              className="btn btn-secondary"
              style={{ fontSize: '0.8rem', padding: '6px 12px' }}
              onClick={() => onInstantDemo('insider_trading.json')}
              title="View instant pre-recorded Wall Street simulation"
            >
              <Sparkles size={14} color="#f59e0b" /> Demo: Insider Trading
            </button>
            <button
              className="btn btn-secondary"
              style={{ fontSize: '0.8rem', padding: '6px 12px' }}
              onClick={() => onInstantDemo('india_fraud.json')}
              title="View instant pre-recorded Satyam Scam simulation"
            >
              <Sparkles size={14} color="#f59e0b" /> Demo: Satyam Fraud
            </button>
            <button
              className="btn btn-outline-gold"
              style={{ fontSize: '0.8rem', padding: '6px 12px' }}
              onClick={onReset}
            >
              <RefreshCw size={13} /> Reset Setup
            </button>
          </div>
        </div>
      </div>
    </header>
  )
}
