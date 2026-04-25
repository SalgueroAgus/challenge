import type { RAGSource } from './api/client'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: RAGSource[]
}
