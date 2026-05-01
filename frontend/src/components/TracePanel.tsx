import { useState } from 'react'
import type { AgentRunTraceDetail, PlanTripResponse } from '../lib/api'

interface Props {
  planResult: PlanTripResponse | null
  detail: AgentRunTraceDetail | null
  loading: boolean
}

function statusBadge(status: string) {
  const s = status.toLowerCase()
  if (s === 'completed' || s === 'ok' || s === 'success') return 'badge-success'
  if (s === 'failed' || s === 'error') return 'badge-error'
  if (s === 'running') return 'badge-running'
  return 'badge-neutral'
}

function fmtLatency(ms: number | null) {
  if (ms === null) return '—'
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`
}

function fmtCost(usd: number | null) {
  if (usd === null) return '—'
  return `$${usd.toFixed(5)}`
}

function fmtTokens(n: number | null) {
  return n === null ? '—' : n.toLocaleString()
}

export default function TracePanel({ planResult, detail, loading }: Props) {
  const [traceOpen, setTraceOpen] = useState(false)

  const answer = detail?.final_answer ?? planResult?.answer ?? null
  const destination = detail?.recommended_destination ?? planResult?.recommended_destination ?? null
  const webhookDelivered = detail?.webhook_delivered ?? planResult?.webhook_delivered ?? null
  const status = detail?.status ?? (planResult ? 'completed' : null)

  return (
    <div className="card">
      <div className="card-header">
        <span className="card-title">Trip Plan</span>
        {status && (
          <span className={`badge ${statusBadge(status)}`}>{status}</span>
        )}
      </div>
      <div className="card-body">
        {/* Destination */}
        {destination && (
          <div className="result-destination">
            <div className="result-destination-label">Recommended destination</div>
            <div className="result-destination-name">📍 {destination}</div>
          </div>
        )}

        {/* Answer */}
        {answer ? (
          <div className="result-answer">{answer}</div>
        ) : loading ? (
          <div className="planner-loading" style={{ paddingTop: 8 }}>
            <div className="spinner" />
            <span>Loading trace details…</span>
          </div>
        ) : null}

        {/* Meta row */}
        <div className="result-meta">
          {webhookDelivered !== null && (
            <span className={`badge ${webhookDelivered ? 'badge-success' : 'badge-warning'}`}>
              {webhookDelivered ? '✓ Sent to Discord' : '✗ Discord delivery failed'}
            </span>
          )}
          {detail?.total_cost_usd !== undefined && detail.total_cost_usd !== null && (
            <span className="cost-text">Cost: {fmtCost(detail.total_cost_usd)}</span>
          )}
        </div>

        {/* Trace debug section */}
        {detail && (
          <>
            <div className="divider" />
            <div className="trace-toggle">
              <span className="trace-section-title" style={{ margin: 0 }}>Agent trace</span>
              <button
                className="trace-toggle-btn"
                type="button"
                onClick={() => setTraceOpen(o => !o)}
              >
                {traceOpen ? '▲ Hide' : '▼ Show'}
              </button>
            </div>

            {traceOpen && (
              <div className="trace-body" style={{ marginTop: 16 }}>
                {/* Tool calls */}
                <div>
                  <div className="trace-section-title">Tool calls</div>
                  {detail.tool_calls.length === 0 ? (
                    <p className="trace-empty">No tool calls recorded.</p>
                  ) : (
                    <table className="trace-table">
                      <thead>
                        <tr>
                          <th>Tool</th>
                          <th>Status</th>
                          <th>Latency</th>
                          <th>Summary</th>
                        </tr>
                      </thead>
                      <tbody>
                        {detail.tool_calls.map(tc => (
                          <tr key={tc.id}>
                            <td><strong>{tc.tool_name}</strong></td>
                            <td>
                              <span className={`badge ${statusBadge(tc.status)}`}>{tc.status}</span>
                            </td>
                            <td className="cost-text">{fmtLatency(tc.latency_ms)}</td>
                            <td style={{ maxWidth: 280, wordBreak: 'break-word' }}>
                              {tc.output_summary ?? '—'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>

                {/* LLM usage */}
                <div>
                  <div className="trace-section-title">LLM usage</div>
                  {detail.llm_usage.length === 0 ? (
                    <p className="trace-empty">No LLM usage recorded.</p>
                  ) : (
                    <table className="trace-table">
                      <thead>
                        <tr>
                          <th>Step</th>
                          <th>Model</th>
                          <th>In</th>
                          <th>Out</th>
                          <th>Cost</th>
                          <th>Latency</th>
                        </tr>
                      </thead>
                      <tbody>
                        {detail.llm_usage.map(lu => (
                          <tr key={lu.id}>
                            <td>{lu.step_name}</td>
                            <td className="cost-text">{lu.model}</td>
                            <td className="cost-text">{fmtTokens(lu.input_tokens)}</td>
                            <td className="cost-text">{fmtTokens(lu.output_tokens)}</td>
                            <td className="cost-text">{fmtCost(lu.cost_usd)}</td>
                            <td className="cost-text">{fmtLatency(lu.latency_ms)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>

                {/* Timeline events */}
                <div>
                  <div className="trace-section-title">Timeline</div>
                  {detail.trace_events.length === 0 ? (
                    <p className="trace-empty">No events recorded.</p>
                  ) : (
                    <table className="trace-table">
                      <thead>
                        <tr>
                          <th>Type</th>
                          <th>Event</th>
                          <th>Latency</th>
                        </tr>
                      </thead>
                      <tbody>
                        {detail.trace_events.map(ev => (
                          <tr key={ev.id}>
                            <td>
                              <span className="badge badge-neutral">{ev.event_type}</span>
                            </td>
                            <td>{ev.event_name}</td>
                            <td className="cost-text">{fmtLatency(ev.latency_ms)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
