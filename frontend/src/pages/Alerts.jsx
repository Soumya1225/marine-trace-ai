import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import api, { errorText } from '../api/client'
import { Empty, ErrorNote, PriorityBadge, fmtTime } from '../components/Bits.jsx'

export default function Alerts() {
  const [rows, setRows] = useState(null)
  const [error, setError] = useState('')
  const load = () => api.get('/alerts').then((r) => setRows(r.data)).catch((e) => setError(errorText(e)))
  useEffect(() => { load() }, [])
  const read = async (id) => { await api.post(`/alerts/${id}/read`); load() }
  const readAll = async () => { await api.post('/alerts/read-all'); load() }

  return (
    <div className="page narrow">
      <header className="page-head"><h1>Alerts</h1>{rows?.some((a) => !a.read) && <button className="btn" onClick={readAll}>Mark all as read</button>}</header>
      <ErrorNote text={error} />
      {rows && rows.length === 0 && <Empty title="No alerts">Alerts are raised for Medium, High and Critical incidents.</Empty>}
      {rows?.map((a) => (
        <article key={a.id} className={a.read ? 'alert-item' : 'alert-item unread'}>
          <div className="alert-top"><PriorityBadge value={a.level} /><span className="muted small">{fmtTime(a.created_at)}{a.emailed ? ', e-mailed' : ''}</span></div>
          <p>{a.message}</p>
          <div className="row-btns">
            {a.incident_id && <Link className="btn small" to={`/incidents/${a.incident_id}`}>Open incident</Link>}
            {!a.read && <button className="link-btn" onClick={() => read(a.id)}>Mark as read</button>}
          </div>
        </article>
      ))}
    </div>
  )
}
