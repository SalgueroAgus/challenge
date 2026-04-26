import type { RAGSource } from './api/client'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: RAGSource[]
  route?: 'rag' | 'direct'  // agent mode only
  retries?: number           // agent mode only; > 0 means query was rewritten
}
