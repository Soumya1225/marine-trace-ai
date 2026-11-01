import { useEffect, useState } from 'react'
import { Polygon, Rectangle, Tooltip } from 'react-leaflet'
import { Link } from 'react-router-dom'
import api, { errorText } from '../api/client'
import BaseMap from '../components/BaseMap.jsx'
import { ErrorNote, fmtTime } from '../components/Bits.jsx'
import { useAuth } from '../context/AuthContext.jsx'
import '../zones.css'

const STATUS = {
  new: ['New', 'pri-medium'], queued: ['Queued', 'pri-low'], downloading: ['Downloading', 'pri-low'],
  analysing: ['Analysing', 'pri-low'], done: ['Analysed', 'pri-high'], failed: ['Failed', 'pri-critical'], skipped: ['Skipped', ''],
}

export default function Feed() {
  const { can } = useAuth()
  const [d, setD] = useState(null)
  const [zones, setZones] = useState([])
  const [zone, setZone] = useState('')
  const [days, setDays] = useState(3)
  const [busy, setBusy] = useState('')
  const [msg, setMsg] = useState('')
  const [error, setError] = useState('')
  const [hover, setHover] = useState(null)
  const editable = can('admin', 'analyst')

  const load = () => {
    api.get('/feed').then((r) => setD(r.data)).catch((e) => setError(errorText(e)))
    api.get('/zones').then((r) => setZones(r.data)).catch(() => {})
  }
  useEffect(() => {
    load()
    const t = setInterval(load, 15000)
    return () => clearInterval(t)
  }, [])

  const check = async () => {
    setBusy('Asking Copernicus for new Sentinel-1 images…'); setMsg('')
    try {
      const r = await api.post('/feed/check', { zone_id: zone ? Number(zone) : null, days: Number(days) })
      const total = Object.values(r.data.found).reduce((a, b) => a + b, 0)
      const errs = Object.entries(r.data.errors).map(([k, v]) => `${k}: ${v}`)
      setMsg(`${total} new image(s) found in ${Object.keys(r.data.found).length} zone(s).${errs.length ? ' Problems: ' + errs.join('; ') : ''}`)
      load()
    } catch (e) { setError(errorText(e)) } finally { setBusy('') }
  }
  const act = async (p, what) => {
    try { await api.post(`/feed/${p.id}/${what}`); load() } catch (e) { setError(errorText(e)) }
  }
  const toggleAuto = async (z) => { try { await api.patch(`/zones/${z.id}`, { auto_check: !z.auto_check }); load() } catch (e) { setError(errorText(e)) } }

  if (error) return <ErrorNote text={error} />
  if (!d) return <div className="center-note">Loading…</div>
  const shown = d.passes.filter((p) => !zone || p.zone_id === Number(zone))
  const active = zones.filter((z) => z.active)

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h1>Satellite feed</h1>
          <p className="muted">New Sentinel-1 radar images over your watch zones, straight from the Copernicus Data Space catalogue.
            Automatic check every {d.settings.interval_hours} h for zones with auto check on{d.settings.auto_analyse ? '; new images are downloaded and analysed automatically' : ''}.</p>
        </div>
      </header>

      {editable && (
        <section className="panel feed-controls">
          <label>Zone
            <select value={zone} onChange={(e) => setZone(e.target.value)}>
              <option value="">All active zones</option>
              {active.map((z) => <option key={z.id} value={z.id}>{z.name}</option>)}
            </select>
          </label>
          <label>Look back (days)
            <input type="number" min="1" max="30" value={days} onChange={(e) => setDays(e.target.value)} />
          </label>
          <button className="btn primary" onClick={check} disabled={!!busy || active.length === 0}>{busy ? 'Checking…' : 'Check for new images now'}</button>
          {active.length === 0 && <span className="muted small">Create a <Link to="/zones">watch zone</Link> first.</span>}
        </section>
      )}
      {busy && <p className="warn-note">{busy}</p>}
      {msg && <p className="ok-note">{msg}</p>}
      {!d.settings.can_download && <p className="warn-note">Searching works without an account. To download and analyse an image, add your free Copernicus login (CDSE_USER, CDSE_PASSWORD) to the .env file.</p>}

      <div className="grid-2">
        <section className="panel">
          <h2>Image footprints</h2>
          <BaseMap center={[20, 78]} zoom={4} height={420}>
            {active.map((z) => (
              <Rectangle key={z.id} bounds={[[z.bbox[1], z.bbox[0]], [z.bbox[3], z.bbox[2]]]} pathOptions={{ color: '#2C6E91', weight: 2, fillOpacity: 0.08 }}>
                <Tooltip>{z.name}</Tooltip>
              </Rectangle>
            ))}
            {shown.filter((p) => p.footprint).slice(0, 60).map((p) => (
              <Polygon key={p.id} positions={p.footprint.map(([lon, lat]) => [lat, lon])}
                pathOptions={{ color: hover === p.id ? '#C8322B' : '#E8A42A', weight: hover === p.id ? 3 : 1, fillOpacity: hover === p.id ? 0.15 : 0.03 }}>
                <Tooltip>{fmtTime(p.acquired_at)} · {p.zone}</Tooltip>
              </Polygon>
            ))}
          </BaseMap>
        </section>
        <section className="panel">
          <h2>Zones</h2>
          <table className="table">
            <thead><tr><th>Zone</th><th>Images found</th><th>Last checked</th><th>Auto check</th></tr></thead>
            <tbody>
              {active.map((z) => (
                <tr key={z.id}>
                  <td>{z.name}</td>
                  <td>{d.passes.filter((p) => p.zone_id === z.id).length}</td>
                  <td className="small">{z.last_checked_at ? fmtTime(z.last_checked_at) : 'never'}</td>
                  <td>{editable ? <button className={z.auto_check ? 'btn small on' : 'btn small'} onClick={() => toggleAuto(z)}>{z.auto_check ? 'On' : 'Off'}</button> : (z.auto_check ? 'On' : 'Off')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      </div>

      <section className="panel">
        <h2>{shown.length} image{shown.length === 1 ? '' : 's'}</h2>
        {shown.length === 0 ? <p className="muted">No images yet. Click "Check for new images now".</p> : (
          <table className="table">
            <thead><tr><th>Taken (your time)</th><th>Zone</th><th>Image</th><th>Size</th><th>Status</th><th /></tr></thead>
            <tbody>
              {shown.map((p) => {
                const [label, cls] = STATUS[p.status] || [p.status, '']
                return (
                  <tr key={p.id} onMouseEnter={() => setHover(p.id)} onMouseLeave={() => setHover(null)}>
                    <td>{fmtTime(p.acquired_at)}</td>
                    <td>{p.zone}</td>
                    <td className="small" title={p.name}>{p.name.slice(0, 38)}…</td>
                    <td>{p.size_mb ? `${Math.round(p.size_mb)} MB` : '–'}</td>
                    <td><span className={`badge ${cls}`}>{label}</span>{p.message && <div className="small muted">{p.message}</div>}</td>
                    <td className="zone-row-actions">
                      {editable && ['new', 'failed', 'skipped'].includes(p.status) && <button className="link-btn" onClick={() => act(p, 'analyse')}>Download &amp; analyse</button>}
                      {editable && p.status === 'new' && <button className="link-btn" onClick={() => act(p, 'skip')}>Skip</button>}
                      {p.job_id && <Link to="/incidents">See results</Link>}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </section>
    </div>
  )
}
