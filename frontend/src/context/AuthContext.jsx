import { createContext, useContext, useEffect, useState } from 'react'
import api from '../api/client'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!localStorage.getItem('token')) { setLoading(false); return }
    api.get('/auth/me').then((r) => setUser(r.data)).catch(() => {}).finally(() => setLoading(false))
  }, [])

  const login = async (email, password) => {
    const r = await api.post('/auth/login', { email, password })
    localStorage.setItem('token', r.data.token)
    setUser(r.data.user)
  }
  const logout = () => { localStorage.removeItem('token'); setUser(null) }
  const can = (...roles) => !!user && roles.includes(user.role)

  return <AuthContext.Provider value={{ user, loading, login, logout, can }}>{children}</AuthContext.Provider>
}

export const useAuth = () => useContext(AuthContext)
