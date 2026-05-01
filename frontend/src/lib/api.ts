const API_BASE = import.meta.env.VITE_API_BASE_URL as string

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

async function apiFetch<T>(
  path: string,
  token: string,
  options: RequestInit = {},
): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
      ...(options.headers ?? {}),
    },
  })

  if (res.status === 401) throw new ApiError(401, 'Session expired. Please log in again.')
  if (res.status === 422) {
    const body = await res.json().catch(() => ({ detail: 'Invalid input.' }))
    const detail =
      typeof body.detail === 'string' ? body.detail : 'Please check your input and try again.'
    throw new ApiError(422, detail)
  }
  if (!res.ok) throw new ApiError(res.status, 'Planning failed. Please try again.')

  return res.json() as Promise<T>
}

// ---- Types ----

export interface PlanTripResponse {
  run_id: string
  answer: string
  recommended_destination: string | null
  webhook_delivered: boolean | null
}

export interface AgentRunSummary {
  id: string
  status: string
  user_query: string
  recommended_destination: string | null
  total_cost_usd: number | null
  webhook_delivered: boolean | null
  created_at: string
  completed_at: string | null
}

export interface ToolCallLogResponse {
  id: string
  tool_name: string
  status: string
  output_summary: string | null
  latency_ms: number | null
  created_at: string
}

export interface LLMUsageLogResponse {
  id: string
  step_name: string
  model: string
  input_tokens: number | null
  output_tokens: number | null
  cost_usd: number | null
  latency_ms: number | null
  created_at: string
}

export interface AgentTraceEventResponse {
  id: string
  event_type: string
  event_name: string
  latency_ms: number | null
  created_at: string
}

export interface AgentRunTraceDetail {
  id: string
  status: string
  user_query: string
  final_answer: string | null
  recommended_destination: string | null
  total_cost_usd: number | null
  webhook_delivered: boolean | null
  created_at: string
  completed_at: string | null
  tool_calls: ToolCallLogResponse[]
  llm_usage: LLMUsageLogResponse[]
  trace_events: AgentTraceEventResponse[]
}

// ---- API calls ----

export async function planTrip(token: string, message: string): Promise<PlanTripResponse> {
  return apiFetch<PlanTripResponse>('/chat/plan-trip', token, {
    method: 'POST',
    body: JSON.stringify({ message }),
  })
}

export async function listTraces(token: string): Promise<AgentRunSummary[]> {
  return apiFetch<AgentRunSummary[]>('/traces', token)
}

export async function getTrace(token: string, runId: string): Promise<AgentRunTraceDetail> {
  return apiFetch<AgentRunTraceDetail>(`/traces/${runId}`, token)
}
