import { useEffect, useMemo, useState } from 'react'
import { CircleMarker, Polyline, Rectangle, Tooltip, useMap } from 'react-leaflet'
import api, { errorText } from '../api/client'
import BaseMap from '../components/BaseMap.jsx'
import { ErrorNote, fmtNum, fmtTime } from '../components/Bits.jsx'
import { useAuth } from '../context/AuthContext.jsx'
import '../zones.css'
import '../system.css'

const CAT = [
  ['Tanker', '#8E1B14', (t) => t >= 80 && t <= 89],
  ['Cargo', '#2C6E91', (t) => t >= 70 && t <= 79],
  ['Passenger', '#6B2FA0', (t) => t >= 60 && t <= 69],
  ['Fishing', '#3E8E6A', (t) => t === 30],
  ['Tug / service', '#E8A42A', (t) => (t >= 50 && t <= 59) || t === 31 || t === 32],
  ['Other / unknown', '#5F6F7A', () => true],
]
const cat = (t) => CAT.find(([, , f]) => f(Number(t) || 0))

function FitTo({ bbox }) {
  const map = useMap()
  useEffect(() => { if (bbox) map.fitBounds([[bbox[1], bbox[0]], [bbox[3], bbox[2]]]) }, [bbox, map])
  return null
}

export default function LiveShips() {
  const { can } = useAuth()
  const [zones, setZones] = useState([])
  const [zoneId, setZoneId] = useState('')
  const [d, setD] = useState(null)
  const [sel, setSel] = useState(null)
  const [onlyTankers, setOnlyTankers] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => { api.get('/zones').then((r) => setZones(r.data.filter((z) => z.active))).catch(() => {}) }, [])
  useEffect(() => {
    const load = () => api.get('/live/ships', { params: zoneId ? { zone_id: zoneId } : {} }).then((r) => setD(r.data)).catch((e) => setError(errorText(e)))
    load()
    const t = setInterval(load, 10000)
    return () => clearInterval(t)
  }, [zoneId])

  const zone = zones.find((z) => String(z.id) === String(zoneId))
  const ships = useMemo(() => (d?.ships || []).filter((s) => !onlyTankers || cat(s.type)[0] === 'Tanker'), [d, onlyTankers])
  const selected = ships.find((s) => s.mmsi === sel)

  if (error) return <ErrorNote text={error} />
  if (!d) return <div className="center-note">Loading…</div>
  const st = d.status

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h1>Live ships</h1>
          <p className="muted">Real AIS positions from AISStream.io for your watch zones, refreshed every 10 seconds.
            Every position is also recorded, so satellite images can later be analysed with the real ship tracks.</p>
        </div>
        <span className={`health-pill ${st.connected ? 'ok' : 'warning'}`}>{st.connected ? '● Live' : 'Not connected'}</span>
      </header>
      {!st.configured && <p className="warn-note">Add AISSTREAM_API_KEY to the .env file (free key at aisstream.io), then restart the backend.</p>}
      {st.configured && !st.connected && st.error && <p className="warn-note">{st.error}</p>}

      <section className="panel feed-controls">
        <label>Zone
          <select value={zoneId} onChange={(e) => { setZoneId(e.target.value); setSel(null) }}>
            <option value="">All watch zones</option>
            {zones.map((z) => <option key={z.id} value={z.id}>{z.name}</option>)}
          </select>
        </label>
        <label className="check"><input type="checkbox" checked={onlyTankers} onChange={(e) => setOnlyTankers(e.target.checked)} /> Tankers only</label>
        <span className="small muted">{st.messages} messages received{st.last_message_at ? `, last ${fmtTime(st.last_message_at)}` : ''}</span>
        {can('admin') && <button className="btn small" onClick={() => api.post('/live/restart')}>Reconnect</button>}
      </section>

      <section className="figures">
        {[['Ships in view', d.counts.ships], ['Moving', d.counts.moving], ['Stopped', d.counts.stopped], ['Went silent (AIS off?)', d.counts.silent]].map(([k, v]) => (
          <div key={k} className={`figure${k.startsWith('Went') && v ? ' figure-warn' : ''}`}><div className="figure-value">{v}</div><div className="figure-label">{k}</div></div>
        ))}
      </section>

      <div className="grid-2">
        <section className="panel">
          <BaseMap center={[20, 78]} zoom={4} height={520}>
            <FitTo bbox={zone?.bbox} />
            {zones.map((z) => (
              <Rectangle key={z.id} bounds={[[z.bbox[1], z.bbox[0]], [z.bbox[3], z.bbox[2]]]} pathOptions={{ color: '#2C6E91', weight: 1, fillOpacity: 0.03, dashArray: '4 4' }} />
            ))}
            {selected && selected.trail.length > 1 && <Polyline positions={selected.trail} pathOptions={{ color: '#C8322B', weight: 3 }} />}
            {ships.map((s) => {
              const [label, color] = cat(s.type)
              return (
                <CircleMarker key={s.mmsi} center={[s.lat, s.lon]} radius={s.mmsi === sel ? 8 : label === 'Tanker' ? 6 : 4}
                  pathOptions={{ color: s.silent ? '#B3261E' : color, fillColor: color, fillOpacity: s.silent ? 0.2 : 0.85, weight: s.silent ? 3 : 1, dashArray: s.silent ? '3 3' : null }}
                  eventHandlers={{ click: () => setSel(s.mmsi) }}>
                  <Tooltip>{s.name || `MMSI ${s.mmsi}`} · {label} · {fmtNum(s.sog, 1)} kn{s.silent ? ' · SILENT' : ''}</Tooltip>
                </CircleMarker>
              )
            })}
          </BaseMap>
          <div className="legend-row small">
            {CAT.map(([l, c]) => <span key={l}><i style={{ background: c }} /> {l}</span>)}
            <span><i className="silent-dot" /> went silent</span>
          </div>
        </section>

        <section className="panel">
          {selected ? (
            <>
              <h2>{selected.name || `MMSI ${selected.mmsi}`}</h2>
              <table className="table">
                <tbody>
                  {[['MMSI', selected.mmsi], ['Type', `${cat(selected.type)[0]} (${selected.type ?? '–'})`], ['IMO', selected.imo || '–'],
                    ['Length', selected.length ? `${selected.length} m` : '–'], ['Speed', `${fmtNum(selected.sog, 1)} kn`],
                    ['Course', selected.cog != null ? `${fmtNum(selected.cog, 0)}°` : '–'], ['Destination', selected.destination || '–'],
                    ['Zone', selected.zone || '–'], ['Last seen', `${fmtTime(selected.seen)} (${selected.age_min} min ago)`]].map(([k, v]) => (
                    <tr key={k}><td>{k}</td><td>{v}</td></tr>
                  ))}
                </tbody>
              </table>
              {selected.silent && <p className="warn-note">This ship was moving and stopped reporting more than 30 minutes ago. It may have switched off its AIS, or left radio range.</p>}
              <button className="link-btn" onClick={() => setSel(null)}>Back to the list</button>
            </>
          ) : (
            <>
              <h2>{ships.length} ships</h2>
              <div className="ship-list">
                <table className="table">
                  <thead><tr><th>Ship</th><th>Type</th><th>Speed</th><th>Seen</th></tr></thead>
                  <tbody>
                    {[...ships].sort((a, b) => (b.silent - a.silent) || (cat(a.type)[0] === 'Tanker' ? -1 : 1)).slice(0, 200).map((s) => (
                      <tr key={s.mmsi} onClick={() => setSel(s.mmsi)} className="clickable">
                        <td>{s.silent ? '⚠ ' : ''}{s.name || s.mmsi}</td><td>{cat(s.type)[0]}</td>
                        <td>{fmtNum(s.sog, 1)} kn</td><td className="small">{s.age_min} min</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {ships.length === 0 && <p className="muted">No ships yet. Positions arrive within a minute or two after connecting.</p>}
            </>
          )}
        </section>
      </div>
    </div>
  )
}
