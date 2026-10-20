import React, { useState, useEffect } from 'react'
import { Navbar } from './components/Navbar'
import { CaseSetup } from './components/CaseSetup'
import { TrialArena } from './components/TrialArena'
import type { SampleCase, TrialState } from './types'

export const App: React.FC = () => {
  const [view, setView] = useState<'setup' | 'trial'>('setup')
  const [health, setHealth] = useState<{ groq: boolean; tavily: boolean; langsmith: boolean } | null>(null)
  const [samples, setSamples] = useState<SampleCase[]>([])
  const [isStarting, setIsStarting] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
  const [currentStage, setCurrentStage] = useState('INITIALIZATION')
  const [trialState, setTrialState] = useState<TrialState>({})
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  // Fetch initial health and sample cases
  useEffect(() => {
    fetch('/api/health')
      .then(res => res.json())
      .then(data => setHealth(data.keys))
      .catch(err => console.error('Health check failed', err))

    fetch('/api/cases/samples')
      .then(res => res.json())
      .then(data => setSamples(data))
      .catch(err => console.error('Failed to load sample cases', err))
  }, [])

  // Start new trial
  const handleStartTrial = async (
    description: string,
    rounds: number,
    fastMode: boolean,
    files: File[]
  ) => {
    setIsStarting(true)
    setErrorMsg(null)

    try {
      const formData = new FormData()
      formData.append('case_description', description)
      formData.append('max_rounds', rounds.toString())
      formData.append('fast_mode', fastMode.toString())
      files.forEach(f => formData.append('files', f))

      const res = await fetch('/api/trials', {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        throw new Error(`Failed to initialize trial: ${res.statusText}`)
      }

      const data = await res.json()
      const trialId = data.trial_id

      setTrialState({
        trial_id: trialId,
        case_description: description,
        max_rounds: rounds,
        fast_mode: fastMode,
        status: 'pending',
      })
      setView('trial')
      setIsStreaming(true)
      startEventStream(trialId)
    } catch (err: any) {
      setErrorMsg(err.message || 'Error commencing trial.')
    } finally {
      setIsStarting(false)
    }
  }

  // Connect to SSE stream
  const startEventStream = (trialId: string) => {
    const eventSource = new EventSource(`/api/trials/${trialId}/stream`)

    eventSource.addEventListener('stage_change', (e: MessageEvent) => {
      const data = JSON.parse(e.data)
      setCurrentStage(data.stage || 'INITIALIZATION')
    })

    eventSource.addEventListener('context_retrieved', (e: MessageEvent) => {
      const data = JSON.parse(e.data)
      setCurrentStage('EVIDENCE_RETRIEVAL')
      setTrialState((prev: TrialState) => ({ ...prev, retrieved_context: data.context }))
    })

    eventSource.addEventListener('web_search_completed', () => {
      setCurrentStage('EVIDENCE_RETRIEVAL')
    })

    eventSource.addEventListener('prosecution_argument', (e: MessageEvent) => {
      const data = JSON.parse(e.data)
      setCurrentStage('PROSECUTION_ARGUMENT')
      setTrialState((prev: TrialState) => ({
        ...prev,
        prosecution_argument: data.argument,
        round_number: data.round_number - 1,
      }))
    })

    eventSource.addEventListener('defense_argument', (e: MessageEvent) => {
      const data = JSON.parse(e.data)
      setCurrentStage('DEFENSE_ARGUMENT')
      setTrialState((prev: TrialState) => ({
        ...prev,
        defense_argument: data.argument,
        round_number: data.round_number - 1,
      }))
    })

    eventSource.addEventListener('judge_evaluation', (e: MessageEvent) => {
      const data = JSON.parse(e.data)
      setCurrentStage('JUDICIAL_REVIEW')
      setTrialState((prev: TrialState) => ({
        ...prev,
        judge_scores: data.scores,
        hallucination_flags: data.hallucination_flags,
      }))
    })

    eventSource.addEventListener('round_advanced', (e: MessageEvent) => {
      const data = JSON.parse(e.data)
      setTrialState((prev: TrialState) => ({ ...prev, round_number: data.next_round }))
    })

    eventSource.addEventListener('jury_deliberation', (e: MessageEvent) => {
      const data = JSON.parse(e.data)
      setCurrentStage('JURY_DELIBERATION')
      setTrialState((prev: TrialState) => ({
        ...prev,
        jury_verdicts: data.jury_verdicts,
        vote_count: data.vote_count,
        demographic_analysis: data.demographic_analysis,
      }))
    })

    eventSource.addEventListener('trial_completed', (e: MessageEvent) => {
      const data = JSON.parse(e.data)
      setCurrentStage('CONCLUDED')
      setIsStreaming(false)
      if (data.state) {
        setTrialState(data.state)
      } else {
        setTrialState((prev: TrialState) => ({ ...prev, final_verdict: data.final_verdict }))
      }
      eventSource.close()
    })

    eventSource.addEventListener('error', (e: MessageEvent) => {
      console.error('SSE Error Event', e)
      setIsStreaming(false)
      eventSource.close()
    })

    eventSource.onerror = (err) => {
      console.warn('EventSource disconnected or closed', err)
      setIsStreaming(false)
      eventSource.close()
    }
  }

  // Load instant cached demo simulation
  const handleInstantDemo = async (filename: string) => {
    try {
      const res = await fetch(`/api/cases/load-cached/${filename}`, { method: 'POST' })
      if (!res.ok) throw new Error('Failed to load demo cache')
      const data = await res.json()
      setTrialState(data.state)
      setCurrentStage('CONCLUDED')
      setIsStreaming(false)
      setView('trial')
    } catch (err: any) {
      setErrorMsg(err.message || 'Error loading demo.')
    }
  }

  const handleReset = () => {
    setView('setup')
    setTrialState({})
    setIsStreaming(false)
    setCurrentStage('INITIALIZATION')
    setErrorMsg(null)
  }

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Navbar
        onReset={handleReset}
        onInstantDemo={handleInstantDemo}
        health={health}
      />

      {errorMsg && (
        <div style={{
          background: 'rgba(239, 68, 68, 0.15)',
          borderBottom: '1px solid rgba(239, 68, 68, 0.3)',
          color: '#f87171',
          padding: '12px 24px',
          textAlign: 'center',
          fontSize: '0.9rem',
          fontWeight: 600,
        }}>
          ⚠️ {errorMsg}
        </div>
      )}

      <main style={{ flex: 1, paddingBottom: '60px' }}>
        {view === 'setup' ? (
          <CaseSetup
            onStartTrial={handleStartTrial}
            samples={samples}
            isStarting={isStarting}
          />
        ) : (
          <TrialArena
            state={trialState}
            isStreaming={isStreaming}
            currentStage={currentStage}
            onReset={handleReset}
          />
        )}
      </main>

      {/* Footer */}
      <footer style={{
        borderTop: '1px solid var(--border-subtle)',
        padding: '24px',
        textAlign: 'center',
        fontSize: '0.8rem',
        color: 'var(--text-muted)',
        background: 'rgba(8, 12, 20, 0.6)',
      }}>
        <p>
          ⚖️ <b>CourtRoom AI</b> &bull; Autonomous Legal Reasoning &amp; Jury Simulation Platform &bull; Built with FastAPI, LangGraph, &amp; React
        </p>
        <p style={{ marginTop: '4px', fontSize: '0.74rem' }}>
          <i>Designed for legal argument testing, academic research, and precedent review. Not legal advice.</i>
        </p>
      </footer>
    </div>
  )
}

export default App
