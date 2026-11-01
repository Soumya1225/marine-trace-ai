import { useEffect, useState } from 'react'
import api, { errorText } from '../api/client'
import { ErrorNote } from '../components/Bits.jsx'

export default function Users() {
  const [rows, setRows] = useState([])
  const [form, setForm] = useState({ name: '', email: '', password: '', role: 'analyst' })
  const [error, setError] = useState('')
  const load = () => api.get('/users').then((r) => setRows(r.data)).catch((e) => setError(errorText(e)))
  useEffect(() => { load() }, [])

  const create = async (e) => {
    e.preventDefault(); setError('')
    try { await api.post('/users', form); setForm({ name: '', email: '', password: '', role: 'analyst' }); load() } catch (err) { setError(errorText(err)) }
  }
  const update = async (id, body) => { try { await api.patch(`/users/${id}`, body); load() } catch (err) { setError(errorText(err)) } }
  const resetPw = (u) => { const p = window.prompt(`New password for ${u.email} (min 6 characters)`); if (p) update(u.id, { password: p }) }
  const set = (k) => (e) => setForm({ ...form, [k]: e.target.value })

  return (
    <div className="page">
      <header className="page-head"><h1>Users</h1></header>
      <ErrorNote text={error} />
      <table className="table panel">
        <thead><tr><th>Name</th><th>Email</th><th>Role</th><th>Active</th><th /></tr></thead>
        <tbody>{rows.map((u) => (
          <tr key={u.id}>
            <td>{u.name}</td><td>{u.email}</td>
            <td><select value={u.role} onChange={(e) => update(u.id, { role: e.target.value })}><option>admin</option><option>analyst</option><option>viewer</option></select></td>
            <td><input type="checkbox" checked={u.active} onChange={(e) => update(u.id, { active: e.target.checked })} /></td>
            <td><button className="link-btn" onClick={() => resetPw(u)}>Set password</button></td>
          </tr>
        ))}</tbody>
      </table>
      <form className="panel form-grid" onSubmit={create}>
        <h2 className="span-2">Add a user</h2>
        <label>Name<input value={form.name} onChange={set('name')} required /></label>
        <label>Email<input type="email" value={form.email} onChange={set('email')} required /></label>
        <label>Password<input type="password" value={form.password} onChange={set('password')} minLength={6} required /></label>
        <label>Role<select value={form.role} onChange={set('role')}><option>analyst</option><option>viewer</option><option>admin</option></select></label>
        <button className="btn primary">Add user</button>
      </form>
    </div>
  )
}
