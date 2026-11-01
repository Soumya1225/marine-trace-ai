import { useEffect, useState } from 'react'
import { Link, Navigate, useNavigate } from 'react-router-dom'
import api, { errorText } from '../api/client'
import { ErrorNote } from '../components/Bits.jsx'
import { useAuth } from '../context/AuthContext.jsx'

export default function Login() {
  const { user, login } = useAuth()
  const navigate = useNavigate()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [firstRun, setFirstRun] = useState(null)
  useEffect(() => { api.get('/auth/setup-status').then((r) => setFirstRun(r.data)).catch(() => {}) }, [])
  if (user) return <Navigate to="/" replace />

  const submit = async (e) => {
    e.preventDefault()
    setBusy(true); setError('')
    try { await login(email, password); navigate('/') } catch (err) { setError(errorText(err)) } finally { setBusy(false) }
  }

  return (
    <div className="login-page">
      <div className="login-copy">
        <h1>SpillWatch</h1>
        <p>Find oil slicks in Sentinel-1 radar images, trace them back to the ships that passed by, and predict where the oil goes next.</p>
        <p><Link to="/welcome">← About the system</Link></p>
      </div>
      <form className="login-card" onSubmit={submit}>
        <h2>Sign in</h2>
        <label>Email<input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required autoFocus /></label>
        <label>Password<input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required /></label>
        <ErrorNote text={error} />
        <button className="btn primary" disabled={busy}>{busy ? 'Signing in…' : 'Sign in'}</button>
        {firstRun?.default_password && (
          <p className="warn-note small">First run: sign in as {firstRun.email} with the default password from the README, then change it under Users.</p>
        )}
      </form>
    </div>
  )
}
