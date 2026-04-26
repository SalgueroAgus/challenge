import { useEffect, useRef, useState } from 'react'
import { sendAgentQuery, sendChat, sendRagQuery } from '../api/client'
import type { Message } from '../types'
import { LoadingBubble } from './LoadingBubble'
import { MessageBubble } from './MessageBubble'

interface Props {
  mode: 'chat' | 'rag' | 'agent'
  sessionId: string
}

const EMPTY_STATE = {
  chat: {
    heading: 'Chat with the AI',
    sub: 'Ask anything — this is a general LLM conversation.',
  },
  rag: {
    heading: 'Ask about Argentine Birds',
    sub: 'Questions are answered from the Listado de las Aves Argentinas documents.',
  },
  agent: {
    heading: 'Agent Q&A',
    sub: 'The agent decides whether to search the documents or answer directly.',
  },
}

const PLACEHOLDER = {
  chat: 'Message the AI…',
  rag: 'Ask a question about Argentine birds…',
  agent: 'Ask anything — the agent will decide how to answer…',
}

export function ConversationPane({ mode, sessionId }: Props) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  useEffect(() => {
    const ta = textareaRef.current
    if (!ta || ta.scrollHeight === 0) return
    ta.style.height = 'auto'
    ta.style.height = `${Math.min(ta.scrollHeight, 120)}px`
  }, [input])

  async function handleSubmit() {
    const text = input.trim()
    if (!text || isLoading) return

    setMessages(prev => [
      ...prev,
      { id: crypto.randomUUID(), role: 'user', content: text },
    ])
    setInput('')
    setIsLoading(true)
    setError(null)

    try {
      if (mode === 'chat') {
        const res = await sendChat(text, sessionId)
        setMessages(prev => [
          ...prev,
          { id: crypto.randomUUID(), role: 'assistant', content: res.reply },
        ])
      } else if (mode === 'rag') {
        const res = await sendRagQuery(text)
        setMessages(prev => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: 'assistant',
            content: res.answer,
            sources: res.sources,
          },
        ])
      } else {
        const res = await sendAgentQuery(text)
        setMessages(prev => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: 'assistant',
            content: res.answer,
            sources: res.sources,
            route: res.route,
            retries: res.meta.retries,
          },
        ])
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong. Is the backend running?')
    } finally {
      setIsLoading(false)
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const isEmpty = messages.length === 0 && !isLoading

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto px-4 sm:px-8 md:px-16 lg:px-32">
        {isEmpty ? (
          <div className="flex flex-col items-center justify-center py-32 gap-2 text-center">
            <p className="text-2xl font-semibold text-gray-700">
              {EMPTY_STATE[mode].heading}
            </p>
            <p className="text-sm text-gray-400 max-w-sm">
              {EMPTY_STATE[mode].sub}
            </p>
          </div>
        ) : (
          <div className="py-4">
            {messages.map(msg => (
              <MessageBubble key={msg.id} message={msg} />
            ))}
            {isLoading && <LoadingBubble mode={mode} />}
            {error && (
              <p className="text-sm text-red-500 text-center py-3">{error}</p>
            )}
            <div ref={bottomRef} />
          </div>
        )}
      </div>

      <div className="border-t border-gray-100 bg-white px-4 sm:px-8 md:px-16 lg:px-32 py-4">
        <div className="flex items-end gap-2 bg-white border border-gray-200 rounded-2xl px-4 py-3 shadow-sm focus-within:border-gray-400 transition-colors">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={PLACEHOLDER[mode]}
            rows={1}
            className="flex-1 resize-none bg-transparent text-sm text-gray-800 placeholder-gray-400 outline-none leading-relaxed min-h-[1.5rem]"
          />
          <button
            onClick={handleSubmit}
            disabled={!input.trim() || isLoading}
            className="shrink-0 w-8 h-8 bg-gray-900 hover:bg-gray-700 disabled:bg-gray-200 rounded-xl flex items-center justify-center transition-colors"
            aria-label="Send"
          >
            <svg
              className="w-3.5 h-3.5 text-white"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 12h14M12 5l7 7-7 7" />
            </svg>
          </button>
        </div>
        <p className="text-xs text-gray-300 text-center mt-2">
          Enter to send · Shift+Enter for newline
        </p>
      </div>
    </div>
  )
}
