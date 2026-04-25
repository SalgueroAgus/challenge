import ReactMarkdown from 'react-markdown'
import type { Message } from '../types'
import { SourceCard } from './SourceCard'

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
      <div className="flex flex-col gap-3 min-w-0 flex-1">
        <div className="bg-white border border-gray-100 rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm">
          <div className="prose prose-sm prose-gray max-w-none">
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
        </div>
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
