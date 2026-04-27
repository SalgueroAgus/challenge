import { useState } from 'react'
import { ConversationPane } from './components/ConversationPane'
import { LoginScreen } from './components/LoginScreen'
import { AuthProvider, useAuth } from './context/AuthContext'

type Tab = 'chat' | 'rag' | 'agent'

const TABS: { id: Tab; label: string }[] = [
  { id: 'chat', label: 'Chat' },
  { id: 'rag', label: 'RAG Q&A' },
  { id: 'agent', label: 'Agent' },
]

function MainApp() {
  const { isAuthenticated, logout } = useAuth()
  const [tab, setTab] = useState<Tab>('chat')
  const [sessionId] = useState(() => crypto.randomUUID())

  if (!isAuthenticated) return <LoginScreen />

  return (
    <div className="h-screen flex flex-col bg-white font-sans antialiased">
      <header className="shrink-0 border-b border-gray-100 px-5 py-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-gray-900 tracking-tight">Aves Argentinas</span>
          <span className="text-xs bg-gray-100 text-gray-500 px-2 py-0.5 rounded-full font-medium">
            GenAI
          </span>
        </div>

        <div className="flex items-center gap-3">
          <nav className="flex gap-1 bg-gray-100 p-1 rounded-xl" role="tablist">
            {TABS.map(t => (
              <button
                key={t.id}
                role="tab"
                aria-selected={tab === t.id}
                onClick={() => setTab(t.id)}
                className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-all duration-150 ${
                  tab === t.id
                    ? 'bg-white text-gray-900 shadow-sm'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                {t.label}
              </button>
            ))}
          </nav>
          <button
            onClick={logout}
            className="text-xs text-gray-400 hover:text-gray-600 transition-colors"
          >
            Sign out
          </button>
        </div>
      </header>

      {/* All three panes stay mounted to preserve history */}
      <main className="flex-1 overflow-hidden">
        <div className={tab === 'chat' ? 'h-full' : 'hidden'}>
          <ConversationPane mode="chat" sessionId={sessionId} />
        </div>
        <div className={tab === 'rag' ? 'h-full' : 'hidden'}>
          <ConversationPane mode="rag" sessionId={sessionId} />
        </div>
        <div className={tab === 'agent' ? 'h-full' : 'hidden'}>
          <ConversationPane mode="agent" sessionId={sessionId} />
        </div>
      </main>
    </div>
  )
}

export default function App() {
  return (
    <AuthProvider>
      <MainApp />
    </AuthProvider>
  )
}
