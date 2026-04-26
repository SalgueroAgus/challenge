import ReactMarkdown from 'react-markdown'
import type { Message } from '../types'
import { SourceCard } from './SourceCard'

function RouteBadge({ route, retries }: { route: 'rag' | 'direct'; retries?: number }) {
  return (
    <div className="flex items-center gap-1.5 flex-wrap">
      {route === 'rag' ? (
        <span className="inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-100">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
          via RAG
        </span>
      ) : (
        <span className="inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full bg-blue-50 text-blue-700 border border-blue-100">
          <span className="w-1.5 h-1.5 rounded-full bg-blue-500" />
          via Direct
        </span>
      )}
      {retries != null && retries > 0 && (
        <span className="text-xs text-gray-400 font-medium">
          · query rewritten
        </span>
      )}
    </div>
  )
}

export function MessageBubble({ message }: { message: Message }) {
  if (message.role === 'user') {
    return (
      <div className="flex justify-end py-2">
        <div className="max-w-[75%] bg-gray-100 rounded-2xl rounded-tr-sm px-4 py-3 text-sm text-gray-800 leading-relaxed whitespace-pre-wrap">
          {message.content}
        </div>
      </div>
    )
  }

  return (
    <div className="flex items-start gap-3 py-2">
      <div className="w-8 h-8 rounded-full bg-gray-100 border border-gray-200 flex items-center justify-center text-xs font-semibold text-gray-500 shrink-0 mt-0.5">
        AI
      </div>
      <div className="flex flex-col gap-2 min-w-0 flex-1">
        <div className="bg-white border border-gray-100 rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm">
          <div className="prose prose-sm prose-gray max-w-none">
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
        </div>
        {message.route && (
          <RouteBadge route={message.route} retries={message.retries} />
        )}
        {message.sources && message.sources.length > 0 && (
          <div className="flex flex-col gap-1.5 pl-0.5">
            <p className="text-xs text-gray-400 font-medium uppercase tracking-wide">
              Sources ({message.sources.length})
            </p>
            {message.sources.map((src, i) => (
              <SourceCard key={src.chunk_id} source={src} index={i} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
