const API_BASE: string = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

// INTENTIONAL TRADEOFF: credentials are hardcoded in this bundle.
// In a production app the frontend would have a login screen — the user would
// supply their password, the server would issue a token, and no secret would
// ever live in client-side code. For this demo we skip the login screen and
// hard-code a single service account so the UI stays simple while the API
// still enforces JWT auth for every other caller (curl, Postman, etc.).
// See README § Authentication for the full explanation.
const AUTH_USERNAME = 'admin'
const AUTH_PASSWORD = 'changeme'

let _token: string | null = null

async function getToken(): Promise<string> {
  if (_token) return _token
  const res = await fetch(`${API_BASE}/api/v1/auth/token`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: `username=${encodeURIComponent(AUTH_USERNAME)}&password=${encodeURIComponent(AUTH_PASSWORD)}`,
  })
  if (!res.ok) throw new Error('Authentication failed — check API credentials')
  _token = (await res.json()).access_token
  return _token!
}

async function authHeaders(): Promise<Record<string, string>> {
  return {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${await getToken()}`,
  }
}

export interface RAGSource {
  chunk_id: string
  source: string
  page: number | null
  score: number
  text_snippet: string
  image_urls: string[]
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

export async function sendChat(message: string, sessionId: string): Promise<ChatResult> {
  const res = await fetch(`${API_BASE}/api/v1/chat`, {
    method: 'POST',
    headers: await authHeaders(),
    body: JSON.stringify({ message, session_id: sessionId }),
  })
  if (!res.ok) throw new Error(`Server error ${res.status}`)
  return res.json()
}

export async function sendRagQuery(query: string): Promise<RagResult> {
  const res = await fetch(`${API_BASE}/api/v1/rag-query`, {
    method: 'POST',
    headers: await authHeaders(),
    body: JSON.stringify({ query }),
  })
  if (!res.ok) throw new Error(`Server error ${res.status}`)
  return res.json()
}

export { API_BASE }
