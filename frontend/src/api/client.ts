const API_BASE: string = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

let _token: string | null = null
let _onUnauthorized: (() => void) | null = null

/** Called by AuthContext to inject the token after a successful login. */
export function setToken(token: string): void {
  _token = token
}

/** Called by AuthContext to clear the token on logout or 401. */
export function clearToken(): void {
  _token = null
}

/** Registered by AuthContext so API calls can trigger a logout on token expiry. */
export function onUnauthorized(cb: () => void): void {
  _onUnauthorized = cb
}

export async function loginRequest(
  username: string,
  password: string,
): Promise<string> {
  const res = await fetch(`${API_BASE}/api/v1/auth/token`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: `username=${encodeURIComponent(username)}&password=${encodeURIComponent(password)}`,
  })
  if (!res.ok) throw new Error('Invalid username or password')
  const data = await res.json()
  return data.access_token as string
}

function authHeaders(): Record<string, string> {
  return {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${_token ?? ''}`,
  }
}

async function apiFetch(url: string, init: RequestInit): Promise<Response> {
  const res = await fetch(url, { ...init, headers: authHeaders() })
  if (res.status === 401) {
    clearToken()
    _onUnauthorized?.()
    throw new Error('Session expired — please log in again')
  }
  return res
}

export interface RAGSource {
  chunk_id: string
  source: string
  page: number | null
  score: number
  text_snippet: string
  image_urls: string[]
  common_name: string
  scientific_name: string
}

export interface ChatResult {
  reply: string
  session_id: string
  model: string
  meta: { latency_ms: number }
}

export interface RagResult {
  answer: string
  sources: RAGSource[]
  meta: { latency_ms: number; hits: number }
}

export interface AgentResult {
  answer: string
  sources: RAGSource[]
  route: 'rag' | 'direct'
  meta: { latency_ms: number; retries: number }
}

export async function sendChat(message: string, sessionId: string): Promise<ChatResult> {
  const res = await apiFetch(`${API_BASE}/api/v1/chat`, {
    method: 'POST',
    body: JSON.stringify({ message, session_id: sessionId }),
  })
  if (!res.ok) throw new Error(`Server error ${res.status}`)
  return res.json()
}

export async function sendRagQuery(query: string): Promise<RagResult> {
  const res = await apiFetch(`${API_BASE}/api/v1/rag-query`, {
    method: 'POST',
    body: JSON.stringify({ query }),
  })
  if (!res.ok) throw new Error(`Server error ${res.status}`)
  return res.json()
}

export async function sendAgentQuery(query: string): Promise<AgentResult> {
  const res = await apiFetch(`${API_BASE}/api/v1/agent`, {
    method: 'POST',
    body: JSON.stringify({ query }),
  })
  if (!res.ok) throw new Error(`Server error ${res.status}`)
  return res.json()
}

export { API_BASE }
