import { createContext, useCallback, useContext, useEffect, useState } from 'react'
import { clearToken, loginRequest, onUnauthorized, setToken } from '../api/client'

interface AuthContextValue {
  isAuthenticated: boolean
  login: (username: string, password: string) => Promise<void>
  logout: () => void
}

const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  // Token is kept in sessionStorage so it survives page refreshes but not
  // new tabs or browser close — reasonable security tradeoff for a demo.
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(() => {
    const stored = sessionStorage.getItem('auth_token')
    if (!stored) return false
    // Validate the stored token has not expired before trusting it.
    try {
      const payload = JSON.parse(atob(stored.split('.')[1]))
      if (payload.exp && payload.exp * 1000 < Date.now()) {
        sessionStorage.removeItem('auth_token')
        return false
      }
      setToken(stored)
      return true
    } catch {
      sessionStorage.removeItem('auth_token')
      return false
    }
  })

  const logout = useCallback(() => {
    sessionStorage.removeItem('auth_token')
    clearToken()
    setIsAuthenticated(false)
  }, [])

  // Register the 401 callback so any API call can trigger a logout.
  useEffect(() => {
    onUnauthorized(logout)
  }, [logout])

  const login = useCallback(async (username: string, password: string) => {
    const token = await loginRequest(username, password)
    sessionStorage.setItem('auth_token', token)
    setToken(token)
    setIsAuthenticated(true)
  }, [])

  return (
    <AuthContext.Provider value={{ isAuthenticated, login, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used inside <AuthProvider>')
  return ctx
}
