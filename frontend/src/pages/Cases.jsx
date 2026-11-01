import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import api, { errorText } from '../api/client'
import { ErrorNote, PriorityBadge, fmtNum, fmtTime } from '../components/Bits.jsx'
import '../cases.css'

const COLUMNS = [
  ['detected', 'New'],
  ['under_review', 'Investigating'],
  ['confirmed', 'Confirmed'],
  ['closed', 'Closed'],
  ['rejected', 'False alarm'],
]
const FILTERS = [['all', 'All cases'], ['mine', 'My cases'], ['unassigned', 'Unassigned'], ['overdue', 'Overdue']]
const PRI = { Critical: 0, High: 1, Medium: 2, Low: 3 }

function dueText(c) {
  if (!c.due_at) return null
  const h = (new Date(c.due_at) - Date.now()) / 3600000
  if (['closed', 'rejected'].includes(c.status)) return `due ${fmtTime(c.due_at)}`
  return h < 0 ? `overdue by ${fmtNum(-h, 0)} h` : h < 48 ? `due in ${fmtNum(h, 0)} h` : `due ${fmtTime(c.due_at)}`
}

export default function Cases() {
  const [d, setD] = useState(null)
  const [filter, setFilter] = useState('all')
  const [error, setError] = useState('')

  useEffect(() => {
    const load = () => api.get('/cases').then((r) => setD(r.data)).catch((e) => setError(errorText(e)))
    load()
    const t = setInterval(load, 30000)
    return () => clearInterval(t)
  }, [])

  if (error) return <ErrorNote text={error} />
  if (!d) return <div className="center-note">Loading…</div>

  const keep = (c) => filter === 'all' ? true
    : filter === 'mine' ? c.assigned_to?.id === d.me
      : filter === 'unassigned' ? !c.assigned_to && !['closed', 'rejected'].includes(c.status)
        : c.overdue
  const shown = d.cases.filter(keep)

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h1>Case board</h1>
          <p className="muted">Every detected spill is a case: assign it, investigate it, confirm or reject it, close it.
            Deadlines follow the priority (Critical {d.sla_hours.Critical} h, High {d.sla_hours.High} h, Medium {d.sla_hours.Medium} h, Low {d.sla_hours.Low / 24} days).</p>
        </div>
      </header>

      <section className="figures">
        {[['Open cases', d.counts.open], ['Assigned to me', d.counts.mine], ['Unassigned', d.counts.unassigned],
          ['Overdue', d.counts.overdue], ['Closed', d.counts.closed]].map(([k, v]) => (
          <div key={k} className={`figure${k === 'Overdue' && v ? ' figure-warn' : ''}`}><div className="figure-value">{v}</div><div className="figure-label">{k}</div></div>
        ))}
      </section>

      <div className="chips case-filters">
        {FILTERS.map(([k, label]) => (
          <button key={k} className={filter === k ? 'btn small on' : 'btn small'} onClick={() => setFilter(k)}>{label}</button>
        ))}
      </div>

      <div className="case-board">
        {COLUMNS.map(([status, label]) => {
          const items = shown.filter((c) => c.status === status)
            .sort((a, b) => (b.overdue - a.overdue) || (PRI[a.priority] - PRI[b.priority]) || (b.id - a.id))
          return (
            <section key={status} className="case-col">
              <h2>{label} <span className="muted">{items.length}</span></h2>
              {items.length === 0 && <p className="muted small">No cases</p>}
              {items.map((c) => (
                <Link key={c.id} to={`/incidents/${c.id}`} className={`case-card${c.overdue ? ' overdue' : ''}`}>
                  <div className="case-top"><PriorityBadge value={c.priority} /> <span className="muted small">#{c.id}</span></div>
                  <div className="case-title">{c.title}</div>
                  <div className="small muted">{fmtNum(c.area_km2)} km² · {fmtTime(c.detected_at)}</div>
                  {c.top_suspect && <div className="small">Top suspect: {c.top_suspect.vessel_name}</div>}
                  <div className="case-foot small">
                    <span>{c.assigned_to ? `👤 ${c.assigned_to.name}` : 'Unassigned'}</span>
                    {dueText(c) && <span className={c.overdue ? 'due overdue' : 'due'}>{dueText(c)}</span>}
                  </div>
                </Link>
              ))}
            </section>
          )
        })}
      </div>
    </div>
  )
}
