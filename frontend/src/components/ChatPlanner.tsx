import { useState } from 'react'
import { ApiError, PlanTripResponse, planTrip } from '../lib/api'

interface Props {
  token: string
  onComplete: (result: PlanTripResponse) => void
  onAuthError: () => void
}

const HINTS = [
  'e.g. "I want a relaxing beach holiday in Asia for two weeks in September."',
  'e.g. "Looking for an adventure trip with hiking and local culture in South America."',
  'e.g. "Family-friendly destination in Europe, budget-conscious, late July."',
]

export default function ChatPlanner({ token, onComplete, onAuthError }: Props) {
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const hint = HINTS[0]

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    const trimmed = query.trim()
    if (!trimmed || loading) return

    setError(null)
    setLoading(true)

    try {
      const result = await planTrip(token, trimmed)
      onComplete(result)
    } catch (err: unknown) {
      if (err instanceof ApiError && err.status === 401) {
        onAuthError()
        return
      }
      setError(err instanceof Error ? err.message : 'Planning failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card">
      <div className="card-header">
        <span className="card-title">Plan a Trip</span>
      </div>
      <div className="card-body">
        <form className="planner-body" onSubmit={handleSubmit}>
          <div className="form-group">
            <textarea
              className="textarea"
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder={hint}
              disabled={loading}
              rows={4}
            />
          </div>

          {error && <div className="alert alert-error">{error}</div>}

          {loading ? (
            <div className="planner-loading">
              <div className="loading-dots">
                <span /><span /><span />
              </div>
              <span>Agent is planning your trip — this may take up to a minute…</span>
            </div>
          ) : (
            <button
              className="btn btn-primary btn-full"
              type="submit"
              disabled={!query.trim()}
            >
              ✈ Plan my trip
            </button>
          )}
        </form>
      </div>
    </div>
  )
}
