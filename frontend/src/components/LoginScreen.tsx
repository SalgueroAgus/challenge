import { type FormEvent, useState } from 'react'
import { useAuth } from '../context/AuthContext'

export function LoginScreen() {
  const { login } = useAuth()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    if (!username.trim() || !password) return
    setLoading(true)
    setError(null)
    try {
      await login(username.trim(), password)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="h-screen flex items-center justify-center bg-white font-sans antialiased">
      <div className="w-full max-w-sm px-6">
        <div className="mb-8 text-center">
          <span className="font-semibold text-gray-900 tracking-tight text-lg">Aves Argentinas</span>
          <span className="ml-2 text-xs bg-gray-100 text-gray-500 px-2 py-0.5 rounded-full font-medium">
            GenAI
          </span>
          <p className="mt-2 text-sm text-gray-400">Sign in to continue</p>
        </div>

        <form onSubmit={handleSubmit} className="flex flex-col gap-3">
          <input
            type="text"
            placeholder="Username"
            value={username}
            onChange={e => setUsername(e.target.value)}
            autoComplete="username"
            required
            className="w-full border border-gray-200 rounded-xl px-4 py-2.5 text-sm text-gray-800 placeholder-gray-400 outline-none focus:border-gray-400 transition-colors"
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={e => setPassword(e.target.value)}
            autoComplete="current-password"
            required
            className="w-full border border-gray-200 rounded-xl px-4 py-2.5 text-sm text-gray-800 placeholder-gray-400 outline-none focus:border-gray-400 transition-colors"
          />

          {error && (
            <p className="text-xs text-red-500 text-center">{error}</p>
          )}

          <button
            type="submit"
            disabled={loading || !username.trim() || !password}
            className="w-full bg-gray-900 hover:bg-gray-700 disabled:bg-gray-200 text-white disabled:text-gray-400 rounded-xl py-2.5 text-sm font-medium transition-colors"
          >
            {loading ? 'Signing in…' : 'Sign in'}
          </button>
        </form>
      </div>
    </div>
  )
}
