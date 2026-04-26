import { useEffect, useState } from 'react'

const MESSAGES: Record<'chat' | 'rag' | 'agent', string[]> = {
  chat: [
    'Thinking...',
    'Generating response...',
    'Crafting an answer...',
    'Almost there...',
  ],
  rag: [
    'Searching the archives...',
    'Retrieving relevant chunks...',
    'Grounding the answer...',
    'Consulting the documents...',
    'Almost there...',
  ],
  agent: [
    'Classifying your question...',
    'Searching the archives...',
    'Grading the results...',
    'Generating answer...',
    'Almost there...',
  ],
}

export function LoadingBubble({ mode }: { mode: 'chat' | 'rag' | 'agent' }) {
  const messages = MESSAGES[mode]
  const [index, setIndex] = useState(0)

  useEffect(() => {
    const id = setInterval(() => setIndex(i => (i + 1) % messages.length), 2000)
    return () => clearInterval(id)
  }, [messages.length])

  return (
    <div className="flex items-start gap-3 py-3">
      <div className="w-8 h-8 rounded-full bg-gray-100 border border-gray-200 flex items-center justify-center text-xs font-semibold text-gray-500 shrink-0">
        AI
      </div>
      <div className="flex flex-col gap-1.5">
        <div className="bg-white border border-gray-100 rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm">
          <div className="flex gap-1 items-center h-4">
            <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.3s]" />
            <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.15s]" />
            <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" />
          </div>
        </div>
        <p className="text-xs text-gray-400 px-1">{messages[index]}</p>
      </div>
    </div>
  )
}
