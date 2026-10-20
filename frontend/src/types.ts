export interface SampleCase {
  id: string
  title: string
  subtitle: string
  description: string
}

export interface CachedCase {
  id: string
  name: string
}

export interface JurorProfile {
  age: number
  gender: string
  occupation: string
  political_lean: string
  education_level: string
  prior_jury_experience: boolean
  empathy_score: number
  skepticism_score: number
  media_bias: number
}

export interface JurorVerdict {
  juror_id: number
  profile: JurorProfile
  verdict: 'Guilty' | 'Not Guilty' | 'Liable' | 'Not Liable'
  confidence: number
  reasoning: string
  swayed_by?: 'prosecution' | 'defense' | 'neither'
}

export interface LegalArgument {
  argument_text: string
  cited_sources: string[]
  confidence_score: number
  legal_strategy: string
}

export interface JudgeScores {
  preliminary_ruling: string
  reasoning_summary: string
  prosecution_scores: Record<string, number>
  defense_scores: Record<string, number>
}

export interface HallucinationFlag {
  side: string
  citation: string
  reason: string
}

export interface TrialState {
  trial_id?: string
  case_description?: string
  retrieved_context?: string
  web_search_results?: string
  prosecution_argument?: LegalArgument
  defense_argument?: LegalArgument
  judge_scores?: JudgeScores
  hallucination_flags?: HallucinationFlag[]
  jury_profiles?: JurorProfile[]
  jury_verdicts?: JurorVerdict[]
  final_verdict?: string
  round_number?: number
  max_rounds?: number
  debate_history?: any[]
  vote_count?: { guilty: number; not_guilty: number }
  demographic_analysis?: string
  fast_mode?: boolean
  status?: string
}
