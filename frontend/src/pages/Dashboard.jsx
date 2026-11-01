import { useEffect, useState } from 'react'
import { CircleMarker, Popup, Rectangle, Tooltip as MapTip } from 'react-leaflet'
import { Link } from 'react-router-dom'
import { Bar, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import api, { errorText } from '../api/client'
import BaseMap, { PRIORITY_COLOR } from '../components/BaseMap.jsx'
import { Empty, ErrorNote, PriorityBadge, StatusBadge, fmtNum, fmtTime } from '../components/Bits.jsx'
import { useAuth } from '../context/AuthContext.jsx'
import '../hotspots.css'

export default function Dashboard() {
  const [zones, setZones] = useState([])
  useEffect(() => { api.get('/zones').then((r) => setZones(r.data.filter((z) => z.active))).catch(() => {}) }, [])
  const [s, setS] = useState(null)
  const [error, setError] = useState('')
  const [view, setView] = useState('incidents')
  const { can } = useAuth()
  useEffect(() => { api.get('/stats').then((r) => setS(r.data)).catch((e) => setError(errorText(e))) }, [])
  if (error) return <ErrorNote text={error} />
  if (!s) return <div className="center-note">Loading…</div>

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h1>Overview</h1>
          <p className="muted">{s.open_incidents} open incident{s.open_incidents === 1 ? '' : 's'}, {s.critical_open} critical.</p>
        </div>
        {can('admin', 'analyst') && <Link className="btn primary" to="/analyze">Analyse a satellite image</Link>}
      </header>

      <section className="figures">
        <Figure label="Open incidents" value={s.open_incidents} />
        <Figure label="Total spill area detected" value={`${fmtNum(s.area_total_km2)} km²`} />
        <Figure label="Vessels tracked" value={s.vessels_tracked} />
        <Figure label="Confirmed suspects" value={s.confirmed_suspects} />
        <Figure label="Possible non-ship sources" value={s.possible_non_ship ?? 0} />
        <Figure label="Unread alerts" value={s.unread_alerts} />
      </section>

      {s.total_incidents === 0 ? (
        <Empty title="No incidents yet">Run your first analysis to see spills here. Use “Create demo data” on the New analysis page to try the system before the real data is ready.</Empty>
      ) : (
        <div className="grid-2">
          <section className="panel">
            <div className="panel-head">
              <h2>{view === 'hotspots' ? 'Pollution hotspots' : 'Where spills were found'}</h2>
              <div className="seg">
                <button className={view === 'incidents' ? 'btn small on' : 'btn small'} onClick={() => setView('incidents')}>Incidents</button>
                <button className={view === 'hotspots' ? 'btn small on' : 'btn small'} onClick={() => setView('hotspots')}>Hotspots</button>
              </div>
            </div>
            <BaseMap center={[s.map[0].lat, s.map[0].lon]} zoom={9} height={380}>
              {zones.map((z) => (
                <Rectangle key={`z${z.id}`} bounds={[[z.bbox[1], z.bbox[0]], [z.bbox[3], z.bbox[2]]]} pathOptions={{ color: '#2C6E91', weight: 1.5, fillOpacity: 0.04, dashArray: '4 4' }}>
                  <MapTip>Watch zone: {z.name}</MapTip>
                </Rectangle>
              ))}
              {view === 'incidents' && s.map.map((m) => (
                <CircleMarker key={m.id} center={[m.lat, m.lon]} radius={6 + Math.min(m.area_km2, 10)}
                  pathOptions={{ color: PRIORITY_COLOR[m.priority], fillOpacity: m.verdict === 'Likely look-alike' ? 0.15 : 0.6, dashArray: m.verdict === 'Likely look-alike' ? '3 3' : null }}>
                  <Popup><Link to={`/incidents/${m.id}`}>Incident #{m.id}</Link><br />{m.priority}, {fmtNum(m.area_km2)} km²{m.verdict ? <><br />{m.verdict}</> : null}</Popup>
                </CircleMarker>
              ))}
              {view === 'hotspots' && (s.hotspots || []).map((h, k) => {
                const top = s.hotspots[0].score || 1
                const t = Math.max(0.15, h.score / top)
                return (
                  <Rectangle key={k} bounds={[[h.bounds[1], h.bounds[0]], [h.bounds[3], h.bounds[2]]]}
                    pathOptions={{ color: '#8E1B14', weight: 1, fillColor: `rgb(${200 + 30 * t}, ${180 - 150 * t}, ${60 - 40 * t})`, fillOpacity: 0.25 + 0.5 * t }}>
                    <MapTip>#{k + 1} hotspot: {h.count} spill{h.count === 1 ? '' : 's'} ({h.likely_oil} likely oil), {fmtNum(h.area_km2)} km², near {h.near}</MapTip>
                  </Rectangle>
                )
              })}
            </BaseMap>
            {view === 'incidents'
              ? <p className="muted small">Faded dashed circles: the look-alike check says these are probably not oil.</p>
              : <p className="muted small">10 km grid cells ranked by how often and how much oil was found there; darker = worse. Rejected incidents are left out.</p>}
          </section>
          <section className="panel">
            <h2>Trend by satellite image date</h2>
            <ResponsiveContainer width="100%" height={220}>
              <ComposedChart data={s.trend || s.by_image_date}>
                <XAxis dataKey="date" fontSize={11} />
                <YAxis yAxisId="n" allowDecimals={false} fontSize={11} width={28} />
                <YAxis yAxisId="a" orientation="right" fontSize={11} width={40} unit=" km²" />
                <Tooltip />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar yAxisId="n" dataKey="oil" stackId="c" fill="#C8322B" name="Likely oil" />
                <Bar yAxisId="n" dataKey="lookalike" stackId="c" fill="#E8A42A" name="Likely look-alike" />
                <Line yAxisId="a" dataKey="area_km2" stroke="#2C6E91" strokeWidth={2} name="Area (km²)" />
              </ComposedChart>
            </ResponsiveContainer>
            <p className="muted small">Counted by the date the satellite image was taken. Colours come from the look-alike check.</p>
            <h2>By priority</h2>
            <div className="chips">
              {['Critical', 'High', 'Medium', 'Low'].map((p) => (
                <span key={p} className="chip"><PriorityBadge value={p} /> {s.by_priority[p] || 0}</span>
              ))}
            </div>
          </section>
        </div>
      )}

      {(s.hotspots || []).length > 0 && (
        <section className="panel">
          <h2>Pollution hotspots</h2>
          <table className="table">
            <thead><tr><th>#</th><th>Location (lat, lon)</th><th>Near</th><th>Spills</th><th>Likely oil</th><th>Area</th><th>Worst priority</th><th>Incidents</th></tr></thead>
            <tbody>
              {s.hotspots.map((h, k) => (
                <tr key={k}>
                  <td>{k + 1}</td>
                  <td>{h.lat.toFixed(3)}, {h.lon.toFixed(3)}</td>
                  <td>{h.near || '–'}</td>
                  <td>{h.count}</td>
                  <td>{h.likely_oil}</td>
                  <td>{fmtNum(h.area_km2)} km²</td>
                  <td><PriorityBadge value={h.max_priority} /></td>
                  <td>{h.incident_ids.map((id) => <Link key={id} to={`/incidents/${id}`} className="id-link">#{id}</Link>)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {s.recent.length > 0 && (
        <section className="panel">
          <h2>Recent incidents</h2>
          <table className="table">
            <thead><tr><th>Incident</th><th>Detected</th><th>Area</th><th>Priority</th><th>Status</th><th>Top suspect</th></tr></thead>
            <tbody>
              {s.recent.map((i) => (
                <tr key={i.id}>
                  <td><Link to={`/incidents/${i.id}`}>#{i.id} {i.title}</Link></td>
                  <td>{fmtTime(i.detected_at)}</td>
                  <td>{fmtNum(i.area_km2)} km²</td>
                  <td><PriorityBadge value={i.priority} /></td>
                  <td><StatusBadge value={i.status} /></td>
                  <td>{i.top_suspect ? `${i.top_suspect.vessel_name} (${fmtNum(i.top_suspect.score)})` : '–'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </div>
  )
}

function Figure({ label, value }) {
  return <div className="figure"><div className="figure-value">{value}</div><div className="figure-label">{label}</div></div>
}
