import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import api, { errorText } from '../api/client'
import { Empty, ErrorNote, PriorityBadge, StatusBadge, fmtNum, fmtTime } from '../components/Bits.jsx'

export default function Incidents() {
  const [rows, setRows] = useState(null)
  const [f, setF] = useState({ status: '', priority: '', q: '', date_from: '', date_to: '' })
  const [error, setError] = useState('')

  const load = () => {
    const params = Object.fromEntries(Object.entries(f).filter(([, v]) => v))
    api.get('/incidents', { params }).then((r) => setRows(r.data)).catch((e) => setError(errorText(e)))
  }
  useEffect(load, []) // eslint-disable-line react-hooks/exhaustive-deps
  const set = (k) => (e) => setF({ ...f, [k]: e.target.value })

  return (
    <div className="page">
      <header className="page-head"><h1>Incidents</h1></header>
      <form className="filters" onSubmit={(e) => { e.preventDefault(); load() }}>
        <input placeholder="Search title" value={f.q} onChange={set('q')} />
        <select value={f.status} onChange={set('status')}>
          <option value="">All statuses</option>
          <option value="detected">Detected</option><option value="under_review">Under review</option>
          <option value="confirmed">Confirmed</option><option value="closed">Closed</option><option value="rejected">Rejected</option>
        </select>
        <select value={f.priority} onChange={set('priority')}>
          <option value="">All priorities</option>
          {['Critical', 'High', 'Medium', 'Low'].map((p) => <option key={p}>{p}</option>)}
        </select>
        <label className="inline">From<input type="date" value={f.date_from} onChange={set('date_from')} /></label>
        <label className="inline">To<input type="date" value={f.date_to} onChange={set('date_to')} /></label>
        <button className="btn">Apply filters</button>
      </form>
      <ErrorNote text={error} />
      {rows && rows.length === 0 && <Empty title="No incidents match">Change the filters or run a new analysis.</Empty>}
      {rows && rows.length > 0 && (
        <table className="table panel">
          <thead><tr><th title="Serial number in this list (always 1, 2, 3 …)">S.No.</th><th title="Permanent incident ID">ID</th><th>Title</th><th>Detected (image time)</th><th>Area</th><th>Confidence</th><th>Priority</th><th>Status</th><th>Top suspect</th></tr></thead>
          <tbody>
            {rows.map((i, idx) => (
              <tr key={i.id}>
                <td>{rows.length - idx}</td>
                <td className="muted">#{i.id}</td>
                <td><Link to={`/incidents/${i.id}`}>{i.title}</Link></td>
                <td>{fmtTime(i.detected_at)}</td>
                <td>{fmtNum(i.area_km2)} km²</td>
                <td>{fmtNum(i.confidence)}</td>
                <td><PriorityBadge value={i.priority} /></td>
                <td><StatusBadge value={i.status} /></td>
                <td>{i.top_suspect ? <Link to={`/vessels/${i.top_suspect.mmsi}`}>{i.top_suspect.vessel_name}</Link> : '–'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
