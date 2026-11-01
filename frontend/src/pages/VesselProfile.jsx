import { useEffect, useState } from 'react'
import { Polyline } from 'react-leaflet'
import { Link, useParams } from 'react-router-dom'
import api, { errorText } from '../api/client'
import BaseMap from '../components/BaseMap.jsx'
import { ErrorNote, PriorityBadge, StatusBadge, fmtNum, fmtTime } from '../components/Bits.jsx'

export default function VesselProfile() {
  const { mmsi } = useParams()
  const [v, setV] = useState(null)
  const [error, setError] = useState('')
  useEffect(() => { api.get(`/vessels/${mmsi}`).then((r) => setV(r.data)).catch((e) => setError(errorText(e))) }, [mmsi])
  if (error) return <ErrorNote text={error} />
  if (!v) return <div className="center-note">Loading…</div>
  const track = v.latest_track.map((p) => [p.lat, p.lon])

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <p className="muted small"><Link to="/vessels">Vessels</Link> / {v.mmsi}</p>
          <h1>{v.name}</h1>
          <p className="head-meta"><span>MMSI {v.mmsi}</span>{v.imo && <span>IMO {v.imo}</span>}<span>{v.vessel_type}</span>{v.length_m && <span>{v.length_m} m</span>}<span>Last seen {fmtTime(v.last_seen)}</span></p>
        </div>
      </header>
      <div className="grid-2">
        <section className="panel">
          <h2>Latest track near a spill</h2>
          {track.length > 1 ? (
            <BaseMap bounds={track} height={360}><Polyline positions={track} pathOptions={{ color: '#C8322B', weight: 3 }} /></BaseMap>
          ) : <p className="muted">No track stored.</p>}
          {v.latest_evidence.length > 0 && <ul className="evidence">{v.latest_evidence.map((e, i) => <li key={i}>{e}</li>)}</ul>}
        </section>
        <section className="panel">
          <h2>Incident history</h2>
          <table className="table">
            <thead><tr><th>Incident</th><th>Image time</th><th>Rank</th><th>Score</th><th>Review</th><th>Case</th></tr></thead>
            <tbody>
              {v.history.map((h) => (
                <tr key={h.incident_id}>
                  <td><Link to={`/incidents/${h.incident_id}`}>#{h.incident_id}</Link> <PriorityBadge value={h.priority} /></td>
                  <td>{fmtTime(h.detected_at)}</td><td>{h.rank}</td><td>{fmtNum(h.score)}</td>
                  <td>{h.review_status}</td><td><StatusBadge value={h.incident_status} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      </div>
    </div>
  )
}
