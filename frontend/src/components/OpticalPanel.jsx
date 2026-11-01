import { useEffect, useState } from 'react'
import api, { errorText } from '../api/client'

function Preview({ id, alt }) {
  const [src, setSrc] = useState(null)
  const [failed, setFailed] = useState(false)
  useEffect(() => {
    if (!id) return undefined
    let url = null
    api.get(`/optical/quicklook/${id}`, { responseType: 'blob' })
      .then((r) => { url = URL.createObjectURL(r.data); setSrc(url) })
      .catch(() => setFailed(true))
    return () => { if (url) URL.revokeObjectURL(url) }
  }, [id])
  if (!id || failed) return <div className="optical-noimg">No preview – use the Copernicus Browser link</div>
  if (!src) return <div className="optical-noimg">Loading preview…</div>
  return <img src={src} alt={alt} className="optical-img" />
}

export default function OpticalPanel({ incidentId }) {
  const [d, setD] = useState(null)
  const [days, setDays] = useState(3)
  const [error, setError] = useState('')
  const load = (n) => {
    setD(null); setError('')
    api.get(`/incidents/${incidentId}/optical`, { params: { days: n } }).then((r) => setD(r.data)).catch((e) => setError(errorText(e)))
  }
  useEffect(() => { load(days) }, [incidentId]) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="panel">
      <h3>Optical check (Sentinel-2)</h3>
      <p className="muted small">Radar cannot tell oil from every look-alike. A cloud-free colour image taken close in time can confirm it: look for a sheen or a brown / silver streak at the spill position ({d ? `${d.lat.toFixed(3)}, ${d.lon.toFixed(3)}` : '…'}).</p>
      <div className="optical-controls">
        <label>Search ± <select value={days} onChange={(e) => { const n = Number(e.target.value); setDays(n); load(n) }}>
          {[1, 3, 5, 10].map((n) => <option key={n} value={n}>{n} day{n > 1 ? 's' : ''}</option>)}
        </select> around the radar image</label>
      </div>
      {error && <p className="warn-note">{error}</p>}
      {!d && !error && <p className="muted">Searching the Copernicus catalogue…</p>}
      {d && <p className={d.scenes.some((s) => s.cloud_pct !== null && s.cloud_pct <= 30) ? 'ok-note' : 'warn-note'}>{d.note}</p>}
      {d && d.scenes.length > 0 && (
        <div className="optical-grid">
          {d.scenes.map((s) => (
            <div key={s.name} className="optical-card">
              <Preview id={s.quicklook_id} alt={`Sentinel-2 preview ${s.time}`} />
              <p><b>{new Date(s.time).toUTCString().slice(5, 22)} UTC</b></p>
              <p className="small">{s.hours_from_radar > 0 ? '+' : ''}{s.hours_from_radar} h from radar · {s.cloud_pct ?? '?'}% cloud · tile {s.tile}</p>
              <a className="btn small" href={s.browser_url} target="_blank" rel="noreferrer">Open in Copernicus Browser</a>
            </div>
          ))}
        </div>
      )}
      <p className="muted small">Sentinel-2 sees only in daylight and not through cloud; it passes every 2–5 days. The preview covers a whole 110 km tile, so use the browser link to zoom in on the spill.</p>
    </div>
  )
}
