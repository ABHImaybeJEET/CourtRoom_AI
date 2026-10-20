import React, { useState } from 'react'
import type { SampleCase } from '../types'
import { Gavel, UploadCloud, BookOpen, Sliders, Zap, FileText, CheckCircle2, X } from 'lucide-react'

interface CaseSetupProps {
  onStartTrial: (description: string, rounds: number, fastMode: boolean, files: File[]) => void
  samples: SampleCase[]
  isStarting: boolean
}

export const CaseSetup: React.FC<CaseSetupProps> = ({ onStartTrial, samples, isStarting }) => {
  const [description, setDescription] = useState('')
  const [rounds, setRounds] = useState(2)
  const [fastMode, setFastMode] = useState(false)
  const [files, setFiles] = useState<File[]>([])
  const [selectedSampleId, setSelectedSampleId] = useState<string | null>(null)

  const handleSelectSample = (sample: SampleCase) => {
    setSelectedSampleId(sample.id)
    setDescription(sample.description)
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFiles(Array.from(e.target.files))
    }
  }

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index))
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!description.trim()) return
    onStartTrial(description, rounds, fastMode, files)
  }

  return (
    <div className="animate-fade" style={{ maxWidth: '1200px', margin: '40px auto', padding: '0 20px' }}>
      {/* Hero Banner */}
      <div style={{ textAlign: 'center', marginBottom: '40px' }}>
        <div style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '8px',
          padding: '6px 16px',
          background: 'rgba(245, 158, 11, 0.1)',
          borderRadius: '9999px',
          border: '1px solid rgba(245, 158, 11, 0.25)',
          marginBottom: '16px',
        }}>
          <Gavel size={15} color="#f59e0b" />
          <span style={{ fontSize: '0.82rem', fontWeight: 600, color: '#f59e0b', letterSpacing: '0.04em' }}>
            AUTONOMOUS ADVERSARIAL TRIAL SIMULATOR
          </span>
        </div>
        <h1 className="font-serif" style={{ fontSize: '2.8rem', fontWeight: 900, marginBottom: '14px', lineHeight: 1.15 }}>
          Simulate Adversarial Legal Proceedings
        </h1>
        <p style={{ fontSize: '1.1rem', color: 'var(--text-dim)', maxWidth: '780px', margin: '0 auto' }}>
          Evaluate case arguments through automated prosecution, defense rebuttal, judicial citation grounding, and an interactive 12-member simulated jury panel.
        </p>
      </div>

      <form onSubmit={handleSubmit} style={{ display: 'grid', gridTemplateColumns: '1.2fr 0.8fr', gap: '28px' }}>
        {/* Left Column: Case Intake & Evidence */}
        <div className="glass-panel" style={{ padding: '28px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px' }}>
            <label style={{ fontWeight: 700, fontSize: '1.05rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <BookOpen size={18} color="#f59e0b" /> Case Brief &amp; Allegations
            </label>
            <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
              {description.length} characters
            </span>
          </div>

          <textarea
            value={description}
            onChange={(e) => {
              setDescription(e.target.value)
              setSelectedSampleId(null)
            }}
            placeholder="Type or paste the factual case background, timeline, accused conduct, relevant contract or statutory clauses, and primary disputes..."
            rows={10}
            required
            style={{
              width: '100%',
              background: 'rgba(7, 10, 18, 0.7)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--radius-md)',
              color: 'var(--text-main)',
              padding: '14px',
              fontFamily: 'var(--font-body)',
              fontSize: '0.95rem',
              resize: 'vertical',
              outline: 'none',
              marginBottom: '20px',
            }}
          />

          {/* Evidence Upload */}
          <div style={{
            border: '2px dashed rgba(255, 255, 255, 0.12)',
            borderRadius: 'var(--radius-md)',
            padding: '20px',
            textAlign: 'center',
            background: 'rgba(255, 255, 255, 0.02)',
            transition: 'border-color 0.2s',
          }}>
            <UploadCloud size={30} color="#94a3b8" style={{ margin: '0 auto 8px' }} />
            <p style={{ fontSize: '0.88rem', fontWeight: 600, marginBottom: '4px' }}>
              Upload Supporting Exhibits / PDF Evidence
            </p>
            <p style={{ fontSize: '0.76rem', color: 'var(--text-muted)', marginBottom: '12px' }}>
              Files are automatically chunked and indexed into an isolated session vector store.
            </p>
            <input
              type="file"
              id="file-upload"
              accept=".pdf"
              multiple
              onChange={handleFileChange}
              style={{ display: 'none' }}
            />
            <label htmlFor="file-upload" className="btn btn-secondary" style={{ fontSize: '0.82rem', padding: '6px 14px' }}>
              Choose PDF Exhibits
            </label>

            {files.length > 0 && (
              <div style={{ marginTop: '14px', display: 'flex', flexWrap: 'wrap', gap: '8px', justifyContent: 'center' }}>
                {files.map((file, idx) => (
                  <span key={idx} className="badge badge-gold" style={{ fontSize: '0.75rem', padding: '4px 8px' }}>
                    <FileText size={12} /> {file.name}
                    <X size={12} style={{ cursor: 'pointer', marginLeft: '4px' }} onClick={() => removeFile(idx)} />
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Pre-loaded Cases & Controls */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          {/* Sample Cases */}
          <div className="glass-panel" style={{ padding: '24px' }}>
            <h3 style={{ fontSize: '0.98rem', fontWeight: 700, marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <BookOpen size={17} color="#38bdf8" /> Pre-Configured Benchmark Cases
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {samples.map((sample) => {
                const isSelected = selectedSampleId === sample.id
                return (
                  <div
                    key={sample.id}
                    onClick={() => handleSelectSample(sample)}
                    style={{
                      padding: '12px 14px',
                      borderRadius: 'var(--radius-sm)',
                      background: isSelected ? 'rgba(245, 158, 11, 0.12)' : 'rgba(255, 255, 255, 0.03)',
                      border: `1px solid ${isSelected ? 'var(--gold-500)' : 'var(--border-subtle)'}`,
                      cursor: 'pointer',
                      transition: 'all 0.2s',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '3px' }}>
                      <span style={{ fontWeight: 700, fontSize: '0.88rem', color: isSelected ? '#fbbf24' : 'var(--text-main)' }}>
                        {sample.title}
                      </span>
                      {isSelected && <CheckCircle2 size={14} color="#f59e0b" />}
                    </div>
                    <p style={{ fontSize: '0.76rem', color: 'var(--text-dim)', margin: 0 }}>
                      {sample.subtitle}
                    </p>
                  </div>
                )
              })}
            </div>
          </div>

          {/* Simulation Controls */}
          <div className="glass-panel" style={{ padding: '24px' }}>
            <h3 style={{ fontSize: '0.98rem', fontWeight: 700, marginBottom: '18px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Sliders size={17} color="#f59e0b" /> Trial Configuration
            </h3>

            {/* Rounds */}
            <div style={{ marginBottom: '18px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>Argument Rounds:</span>
                <span className="badge badge-gold">{rounds} {rounds === 1 ? 'Round' : 'Rounds'}</span>
              </div>
              <input
                type="range"
                min={1}
                max={3}
                value={rounds}
                onChange={(e) => setRounds(parseInt(e.target.value))}
                style={{ width: '100%', accentColor: '#f59e0b' }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                <span>1 (Opening only)</span>
                <span>2 (Opening + Rebuttal)</span>
                <span>3 (Comprehensive)</span>
              </div>
            </div>

            {/* Fast Mode */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '12px',
              borderRadius: 'var(--radius-sm)',
              background: 'rgba(255, 255, 255, 0.03)',
              marginBottom: '24px',
            }}>
              <div>
                <div style={{ fontSize: '0.88rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <Zap size={14} color="#f59e0b" /> Fast Execution Mode
                </div>
                <p style={{ fontSize: '0.74rem', color: 'var(--text-muted)', margin: 0 }}>
                  Uses a 6-juror pool and high-throughput model endpoints.
                </p>
              </div>
              <input
                type="checkbox"
                checked={fastMode}
                onChange={(e) => setFastMode(e.target.checked)}
                style={{ width: '18px', height: '18px', accentColor: '#f59e0b', cursor: 'pointer' }}
              />
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={!description.trim() || isStarting}
              className="btn btn-primary"
              style={{ width: '100%', padding: '14px', fontSize: '1rem', letterSpacing: '0.02em' }}
            >
              <Gavel size={18} />
              {isStarting ? 'Summoning Courtroom Simulation...' : '⚖️ Commence Courtroom Trial'}
            </button>
          </div>
        </div>
      </form>
    </div>
  )
}
