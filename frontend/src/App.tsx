import { useEffect, useState } from 'react'
import type { Session } from '@supabase/supabase-js'
import { supabase } from './lib/supabase'
import type { AgentRunSummary, AgentRunTraceDetail, PlanTripResponse } from './lib/api'
import { getTrace, listTraces } from './lib/api'
import AuthForm from './components/AuthForm'
import ChatPlanner from './components/ChatPlanner'
import RunHistory from './components/RunHistory'
import TracePanel from './components/TracePanel'

export default function App() {
  const [session, setSession] = useState<Session | null>(null)
  const [sessionLoading, setSessionLoading] = useState(true)

  const [planResult, setPlanResult] = useState<PlanTripResponse | null>(null)
  const [runs, setRuns] = useState<AgentRunSummary[]>([])
  const [runsLoading, setRunsLoading] = useState(false)
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null)
  const [traceDetail, setTraceDetail] = useState<AgentRunTraceDetail | null>(null)
  const [traceLoading, setTraceLoading] = useState(false)

  useEffect(() => {
    supabase.auth.getSession().then(({ data }) => {
      setSession(data.session)
      setSessionLoading(false)
    })

    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, s) => {
      setSession(s)
    })

    return () => subscription.unsubscribe()
  }, [])

  useEffect(() => {
    if (session) loadRuns(session.access_token)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session?.access_token])

  function loadRuns(token: string) {
    setRunsLoading(true)
    listTraces(token)
      .then(data => setRuns(data))
      .catch(() => { /* sidebar failure is non-critical */ })
      .finally(() => setRunsLoading(false))
  }

  async function loadTrace(token: string, runId: string) {
    setTraceLoading(true)
    setTraceDetail(null)
    try {
      const detail = await getTrace(token, runId)
      setTraceDetail(detail)
    } catch {
      /* trace load failure is non-critical */
    } finally {
      setTraceLoading(false)
    }
  }

  function handleTripComplete(result: PlanTripResponse) {
    setPlanResult(result)
    setSelectedRunId(result.run_id)
    const token = session?.access_token
    if (!token) return
    loadRuns(token)
    loadTrace(token, result.run_id)
  }

  function handleSelectRun(runId: string) {
    setPlanResult(null)
    setSelectedRunId(runId)
    const token = session?.access_token
    if (token) loadTrace(token, runId)
  }

  function handleRefreshRuns() {
    const token = session?.access_token
    if (token) loadRuns(token)
  }

  async function handleLogout() {
    await supabase.auth.signOut()
    setPlanResult(null)
    setRuns([])
    setTraceDetail(null)
    setSelectedRunId(null)
  }

  function handleAuthError() {
    supabase.auth.signOut()
  }

  if (sessionLoading) {
    return (
      <div className="loading-screen">
        <div className="spinner" />
      </div>
    )
  }

  if (!session) {
    return <AuthForm />
  }

  const token = session.access_token
  const showResult = planResult !== null || traceDetail !== null || traceLoading

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-header-brand">
          <span className="app-header-icon">✈️</span>
          <span className="app-header-title">Smart Travel Planner</span>
        </div>
        <div className="app-header-right">
          <span className="app-header-email">{session.user.email}</span>
          <button className="btn btn-ghost btn-sm" type="button" onClick={handleLogout}>
            Sign out
          </button>
        </div>
      </header>

      <div className="dashboard">
        <aside className="sidebar">
          <RunHistory
            runs={runs}
            loading={runsLoading}
            selectedRunId={selectedRunId}
            onSelect={handleSelectRun}
            onRefresh={handleRefreshRuns}
          />
        </aside>

        <main className="main-content">
          <ChatPlanner
            token={token}
            onComplete={handleTripComplete}
            onAuthError={handleAuthError}
          />

          {showResult && (
            <TracePanel
              planResult={planResult}
              detail={traceDetail}
              loading={traceLoading}
            />
          )}
        </main>
      </div>
    </div>
  )
}
