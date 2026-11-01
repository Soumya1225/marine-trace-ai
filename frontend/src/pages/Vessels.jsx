import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import api, { errorText } from '../api/client'
import { Empty, ErrorNote, fmtNum, fmtTime } from '../components/Bits.jsx'

export default function Vessels() {
  const [rows, setRows] = useState(null)
  const [q, setQ] = useState('')
  const [error, setError] = useState('')
  const load = () => api.get('/vessels', { params: q ? { q } : {} }).then((r) => setRows(r.data)).catch((e) => setError(errorText(e)))
  useEffect(() => { load() }, []) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="page">
      <header className="page-head"><h1>Vessels</h1></header>
      <form className="filters" onSubmit={(e) => { e.preventDefault(); load() }}>
        <input placeholder="Search name or MMSI" value={q} onChange={(e) => setQ(e.target.value)} />
        <button className="btn">Search</button>
      </form>
      <ErrorNote text={error} />
      {rows && rows.length === 0 && <Empty title="No vessels yet">Vessels appear here once they are matched to a spill.</Empty>}
      {rows && rows.length > 0 && (
        <table className="table panel">
          <thead><tr><th>Vessel</th><th>MMSI</th><th>Type</th><th>Length</th><th>Incidents</th><th>Highest score</th><th>Confirmed</th><th>Last seen</th></tr></thead>
          <tbody>
            {rows.map((v) => (
              <tr key={v.mmsi}>
                <td><Link to={`/vessels/${v.mmsi}`}>{v.name}</Link></td><td>{v.mmsi}</td><td>{v.vessel_type}</td>
                <td>{v.length_m ? `${v.length_m} m` : '–'}</td><td>{v.incidents}</td><td>{fmtNum(v.max_score)}</td>
                <td>{v.confirmed}</td><td>{fmtTime(v.last_seen)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
