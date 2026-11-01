import { useEffect, useState } from 'react'
import { CircleMarker, Rectangle, Tooltip, useMapEvents } from 'react-leaflet'
import { Link, useNavigate } from 'react-router-dom'
import api, { errorText } from '../api/client'
import BaseMap, { PRIORITY_COLOR } from '../components/BaseMap.jsx'
import { ErrorNote, fmtTime } from '../components/Bits.jsx'
import { useAuth } from '../context/AuthContext.jsx'
import '../zones.css'

const rect = (b) => [[b[1], b[0]], [b[3], b[2]]]
const r3 = (x) => Math.round(x * 1000) / 1000

function Drawer({ active, onBox }) {
  const [first, setFirst] = useState(null)
  const [hover, setHover] = useState(null)
  useMapEvents({
    click(e) {
      if (!active) return
      if (!first) { setFirst(e.latlng); return }
      const a = first, b = e.latlng
      onBox([r3(Math.min(a.lng, b.lng)), r3(Math.min(a.lat, b.lat)), r3(Math.max(a.lng, b.lng)), r3(Math.max(a.lat, b.lat))])
      setFirst(null); setHover(null)
    },
    mousemove(e) { if (active && first) setHover(e.latlng) },
  })
  if (!active || !first || !hover) return null
  return <Rectangle bounds={[[first.lat, first.lng], [hover.lat, hover.lng]]} pathOptions={{ color: '#E8A42A', dashArray: '6 6' }} />
}

export default function Zones() {
  const { can } = useAuth()
  const navigate = useNavigate()
  const [zones, setZones] = useState(null)
  const [incidents, setIncidents] = useState([])
  const [drawing, setDrawing] = useState(false)
  const [draft, setDraft] = useState(null)
  const [name, setName] = useState('')
  const [notes, setNotes] = useState('')
  const [error, setError] = useState('')
  const [msg, setMsg] = useState('')
  const editable = can('admin', 'analyst')

  const load = () => {
    api.get('/zones').then((r) => setZones(r.data)).catch((e) => setError(errorText(e)))
    api.get('/incidents').then((r) => setIncidents(r.data)).catch(() => {})
  }
  useEffect(load, [])

  const save = async () => {
    try {
      await api.post('/zones', { name, bbox: draft, notes })
      setDraft(null); setName(''); setNotes(''); setDrawing(false); setMsg('Zone saved'); load()
    } catch (e) { setError(errorText(e)) }
  }
  const presets = async () => {
    try { const r = await api.post('/zones/presets'); setMsg(`${r.data.added} preset zones added`); load() } catch (e) { setError(errorText(e)) }
  }
  const toggle = async (z) => { try { await api.patch(`/zones/${z.id}`, { active: !z.active }); load() } catch (e) { setError(errorText(e)) } }
  const remove = async (z) => {
    if (!window.confirm(`Delete watch zone "${z.name}"?`)) return
    try { await api.delete(`/zones/${z.id}`); load() } catch (e) { setError(errorText(e)) }
  }

  if (error) return <ErrorNote text={error} />
  if (!zones) return <div className="center-note">Loading…</div>

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h1>Watch zones</h1>
          <p className="muted">Sea areas to monitor. Spills found inside a zone are tagged with its name and its alerts say which zone is affected.</p>
        </div>
        {editable && (
          <div className="zone-actions">
            <button className={drawing ? 'btn on' : 'btn primary'} onClick={() => { setDrawing(!drawing); setDraft(null) }}>
              {drawing ? 'Cancel drawing' : '+ Draw a new zone'}
            </button>
            <button className="btn" onClick={presets}>Add Indian coast presets</button>
          </div>
        )}
      </header>
      {msg && <p className="ok-note">{msg}</p>}
      {drawing && !draft && <p className="warn-note">Click one corner of the area on the map, then the opposite corner.</p>}

      <section className="panel">
        <BaseMap center={[20, 78]} zoom={4} height={460}>
          <Drawer active={drawing && !draft} onBox={setDraft} />
          {zones.map((z) => (
            <Rectangle key={z.id} bounds={rect(z.bbox)} pathOptions={{ color: z.active ? '#2C6E91' : '#9E9E9E', weight: 2, fillOpacity: 0.08 }}>
              <Tooltip>{z.name}: {z.incidents} spill(s), {z.open_incidents} open</Tooltip>
            </Rectangle>
          ))}
          {draft && <Rectangle bounds={rect(draft)} pathOptions={{ color: '#E8A42A', weight: 3 }} />}
          {incidents.map((i) => (
            <CircleMarker key={i.id} center={[i.centroid_lat, i.centroid_lon]} radius={6}
              pathOptions={{ color: PRIORITY_COLOR[i.priority], fillOpacity: 0.8 }}>
              <Tooltip>#{i.id} {i.title} ({i.priority})</Tooltip>
            </CircleMarker>
          ))}
        </BaseMap>
      </section>

      {draft && (
        <section className="panel">
          <h2>New zone</h2>
          <p className="small muted">Area: {draft.join(', ')} (min lon, min lat, max lon, max lat)</p>
          <div className="form-grid">
            <label>Name<input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Paradip port approach" /></label>
            <label>Notes<input value={notes} onChange={(e) => setNotes(e.target.value)} placeholder="Why this area matters (optional)" /></label>
          </div>
          <p><button className="btn primary" onClick={save} disabled={!name.trim()}>Save zone</button> <button className="btn" onClick={() => setDraft(null)}>Redraw</button></p>
        </section>
      )}

      <section className="panel">
        <h2>{zones.length} zone{zones.length === 1 ? '' : 's'}</h2>
        {zones.length === 0 ? <p className="muted">No zones yet. Draw one on the map or add the Indian coast presets.</p> : (
          <table className="table">
            <thead><tr><th>Zone</th><th>Spills</th><th>Open</th><th>Critical</th><th>Area (lon/lat box)</th><th>Status</th><th /></tr></thead>
            <tbody>
              {zones.map((z) => (
                <tr key={z.id} className={z.active ? '' : 'muted'}>
                  <td><strong>{z.name}</strong>{z.notes && <div className="small muted">{z.notes}</div>}</td>
                  <td>{z.incidents}</td><td>{z.open_incidents}</td><td>{z.critical}</td>
                  <td className="small">{z.bbox.join(', ')}</td>
                  <td>{z.active ? 'Watching' : 'Paused'}{z.last_checked_at && <div className="small muted">checked {fmtTime(z.last_checked_at)}</div>}</td>
                  <td className="zone-row-actions">
                    <button className="link-btn" onClick={() => navigate(`/analyze?bbox=${z.bbox.join(',')}`)}>Analyse this area</button>
                    {editable && <button className="link-btn" onClick={() => toggle(z)}>{z.active ? 'Pause' : 'Resume'}</button>}
                    {editable && <button className="link-btn danger" onClick={() => remove(z)}>Delete</button>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <p className="small muted">Preset boxes are approximate. See spills per zone on the <Link to="/incidents">Incidents</Link> page.</p>
      </section>
    </div>
  )
}
