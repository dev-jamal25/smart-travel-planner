import type { AgentRunSummary } from '../lib/api'

interface Props {
  runs: AgentRunSummary[]
  loading: boolean
  selectedRunId: string | null
  onSelect: (id: string) => void
  onRefresh: () => void
}

function fmtDate(iso: string) {
  const d = new Date(iso)
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) +
    ' ' + d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
}

function statusDot(status: string) {
  const s = status.toLowerCase()
  if (s === 'completed') return '🟢'
  if (s === 'failed') return '🔴'
  if (s === 'running') return '🔵'
  return '⚪'
}

export default function RunHistory({ runs, loading, selectedRunId, onSelect, onRefresh }: Props) {
  return (
    <div className="card history-card">
      <div className="card-header history-header">
        <span className="card-title">Recent runs</span>
        <button
          className="btn btn-ghost btn-sm"
          onClick={onRefresh}
          disabled={loading}
          type="button"
          title="Refresh"
        >
          {loading ? '…' : '↻'}
        </button>
      </div>

      <div className="history-list">
        {loading && runs.length === 0 ? (
          <div className="history-empty">
            <div className="history-empty-icon">🌍</div>
            <div className="history-empty-text">Loading…</div>
          </div>
        ) : runs.length === 0 ? (
          <div className="history-empty">
            <div className="history-empty-icon">🗺️</div>
            <div className="history-empty-text">No trips planned yet.</div>
          </div>
        ) : (
          runs.map(run => (
            <div
              key={run.id}
              className={`history-item${selectedRunId === run.id ? ' selected' : ''}`}
              onClick={() => onSelect(run.id)}
            >
              <div className="history-dest">
                {run.recommended_destination ?? 'Unknown destination'}
              </div>
              <div className="history-query">{run.user_query}</div>
              <div className="history-meta">
                <span>{statusDot(run.status)}</span>
                <span>{fmtDate(run.created_at)}</span>
                {run.total_cost_usd !== null && (
                  <span>${run.total_cost_usd.toFixed(4)}</span>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
