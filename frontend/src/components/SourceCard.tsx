import { useState } from 'react'
import { API_BASE, type RAGSource } from '../api/client'

interface Props {
  source: RAGSource
  index: number
}

export function SourceCard({ source, index }: Props) {
  const [open, setOpen] = useState(false)

  return (
    <div className="border border-gray-200 rounded-xl overflow-hidden text-sm">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-3 py-2 bg-gray-50 hover:bg-gray-100 transition-colors text-left gap-2"
      >
        <span className="flex items-center gap-2 min-w-0">
          <span className="shrink-0 text-xs bg-gray-200 rounded px-1.5 py-0.5 text-gray-500 font-medium">
            {index + 1}
          </span>
          <span className="truncate text-gray-700 font-medium">{source.source}</span>
          {source.page != null && (
            <span className="shrink-0 text-gray-400 text-xs">p.{source.page}</span>
          )}
        </span>
        <span className="flex items-center gap-2 shrink-0">
          <span className="text-xs text-gray-400">{(source.score * 100).toFixed(0)}%</span>
          <svg
            className={`w-3.5 h-3.5 text-gray-400 transition-transform duration-150 ${open ? 'rotate-180' : ''}`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </span>
      </button>

      {open && (
        <div className="px-3 py-2.5 space-y-3 bg-white border-t border-gray-100">
          {source.text_snippet && (
            <p className="text-gray-500 text-xs leading-relaxed">{source.text_snippet}</p>
          )}
          {source.image_urls.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {source.image_urls.map(url => (
                <img
                  key={url}
                  src={`${API_BASE}${url}`}
                  alt="Document image"
                  className="max-h-36 rounded-lg border border-gray-200 object-contain"
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
