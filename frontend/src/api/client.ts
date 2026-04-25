const API_BASE: string = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

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

export async function sendChat(
  message: string,
  sessionId: string,
): Promise<ChatResult> {
  const res = await fetch(`${API_BASE}/api/v1/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, session_id: sessionId }),
  })
  if (!res.ok) throw new Error(`Server error ${res.status}`)
  return res.json()
}

export async function sendRagQuery(query: string): Promise<RagResult> {
  const res = await fetch(`${API_BASE}/api/v1/rag-query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  })
  if (!res.ok) throw new Error(`Server error ${res.status}`)
  return res.json()
}

export { API_BASE }
