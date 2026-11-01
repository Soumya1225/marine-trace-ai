import { useEffect, useState } from 'react'
import { ImageOverlay, LayersControl, Polygon, Tooltip } from 'react-leaflet'
import { Link } from 'react-router-dom'
import api, { errorText } from '../api/client'
import BaseMap from '../components/BaseMap.jsx'
import { Empty, ErrorNote, PriorityBadge, fmtNum, fmtTime } from '../components/Bits.jsx'
import '../compare.css'

const ll = (poly) => (poly || []).map(([lon, lat]) => [lat, lon])

function km(a, b) {
  const R = 6371, r = Math.PI / 180
  const dlat = (b.centroid_lat - a.centroid_lat) * r, dlon = (b.centroid_lon - a.centroid_lon) * r
  const h = Math.sin(dlat / 2) ** 2 + Math.cos(a.centroid_lat * r) * Math.cos(b.centroid_lat * r) * Math.sin(dlon / 2) ** 2
  return 2 * R * Math.asin(Math.sqrt(h))
}

const wind = (inc) => inc?.spill?.wind_speed_ms ?? inc?.environment?.wind_speed_ms

function SpillMap({ inc, earlier, forecast }) {
  const b = inc.scene.bounds
  const bounds = [[b[1], b[0]], [b[3], b[2]]]
  return (
    <BaseMap bounds={bounds} height={420}>
      <LayersControl position="topright">
        <LayersControl.Overlay checked name="Radar image">
          <ImageOverlay url={inc.scene.sar_url} bounds={bounds} opacity={0.85} />
        </LayersControl.Overlay>
        <LayersControl.Overlay checked name="Detection mask">
          <ImageOverlay url={inc.scene.mask_url} bounds={bounds} opacity={0.9} />
        </LayersControl.Overlay>
        <LayersControl.Overlay checked name="Spill outline">
          <Polygon positions={ll(inc.spill?.polygon)} pathOptions={{ color: '#8E1B14', weight: 2, fillOpacity: 0.15 }}>
            <Tooltip>Incident #{inc.id}: {fmtNum(inc.area_km2, 3)} km²</Tooltip>
          </Polygon>
        </LayersControl.Overlay>
        {earlier && (
          <LayersControl.Overlay checked name="Earlier spill (A)">
            <Polygon positions={ll(earlier.spill?.polygon)} pathOptions={{ color: '#5F6F7A', weight: 2, dashArray: '4 4', fillOpacity: 0.05 }}>
              <Tooltip>Earlier spill, incident #{earlier.id}</Tooltip>
            </Polygon>
          </LayersControl.Overlay>
        )}
        {forecast && (
          <LayersControl.Overlay checked name="Forecast from A">
            <Polygon positions={ll(forecast.polygon)} pathOptions={{ color: '#2C6E91', weight: 2, fillOpacity: 0.12 }}>
              <Tooltip>Where incident #{earlier.id} predicted the oil after {forecast.hours} h</Tooltip>
            </Polygon>
          </LayersControl.Overlay>
        )}
      </LayersControl>
    </BaseMap>
  )
}

export default function Compare() {
  const [list, setList] = useState(null)
  const [ids, setIds] = useState(['', ''])
  const [a, setA] = useState(null)
  const [b, setB] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    api.get('/incidents').then((r) => {
      const rows = [...r.data].sort((x, y) => new Date(x.detected_at) - new Date(y.detected_at))
      setList(rows)
      if (rows.length >= 2) setIds([String(rows[rows.length - 2].id), String(rows[rows.length - 1].id)])
    }).catch((e) => setError(errorText(e)))
  }, [])
  useEffect(() => {
    if (ids[0]) api.get(`/incidents/${ids[0]}`).then((r) => setA(r.data)).catch((e) => setError(errorText(e)))
    if (ids[1]) api.get(`/incidents/${ids[1]}`).then((r) => setB(r.data)).catch((e) => setError(errorText(e)))
  }, [ids])

  if (error) return <ErrorNote text={error} />
  if (!list) return <div className="center-note">Loading…</div>
  if (list.length < 2) return <div className="page"><Empty title="Need two incidents">Run at least two analyses (for example the morning and afternoon images of the same area) to compare them.</Empty></div>

  const hoursApart = a && b ? (new Date(b.detected_at) - new Date(a.detected_at)) / 3600000 : null
  const fc = a && hoursApart > 0 && (a.forecast || []).length
    ? a.forecast.reduce((best, f) => (Math.abs(f.hours - hoursApart) < Math.abs(best.hours - hoursApart) ? f : best))
    : null
  const dist = a && b ? km(a, b) : null
  const rows = a && b ? [
    ['Image time', fmtTime(a.detected_at), fmtTime(b.detected_at)],
    ['Spill area', `${fmtNum(a.area_km2, 3)} km²`, `${fmtNum(b.area_km2, 3)} km²`],
    ['Priority', <PriorityBadge key="pa" value={a.priority} />, <PriorityBadge key="pb" value={b.priority} />],
    ['Detection confidence', fmtNum(a.confidence), fmtNum(b.confidence)],
    ['Wind', wind(a) != null ? `${fmtNum(wind(a))} m/s` : '–', wind(b) != null ? `${fmtNum(wind(b))} m/s` : '–'],
    ['Reliability', a.spill?.reliability || '–', b.spill?.reliability || '–'],
    ['Look-alike check', a.spill?.lookalike_check?.verdict || '–', b.spill?.lookalike_check?.verdict || '–'],
    ['Top suspect', a.top_suspect?.vessel_name || '–', b.top_suspect?.vessel_name || '–'],
  ] : []

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h1>Compare two images</h1>
          <p className="muted">Pick an earlier (A) and a later (B) incident. Map B also shows where A&apos;s forecast expected the oil.</p>
        </div>
      </header>

      <div className="compare-pick">
        {['A (earlier)', 'B (later)'].map((label, k) => (
          <label key={label}>{label}
            <select value={ids[k]} onChange={(e) => setIds(k === 0 ? [e.target.value, ids[1]] : [ids[0], e.target.value])}>
              {list.map((i) => <option key={i.id} value={i.id}>#{i.id} {i.title} ({fmtTime(i.detected_at)})</option>)}
            </select>
          </label>
        ))}
      </div>

      {a && b && (
        <>
          <div className="compare-grid">
            <section className="panel">
              <h2>A: <Link to={`/incidents/${a.id}`}>#{a.id} {a.title}</Link></h2>
              <SpillMap inc={a} />
            </section>
            <section className="panel">
              <h2>B: <Link to={`/incidents/${b.id}`}>#{b.id} {b.title}</Link></h2>
              <SpillMap inc={b} earlier={a} forecast={fc} />
            </section>
          </div>

          <section className="panel">
            <h2>Side by side</h2>
            <div className="compare-facts">
              <span><strong>{hoursApart != null ? fmtNum(hoursApart, 1) : '–'} h</strong> between images</span>
              <span><strong>{dist != null ? fmtNum(dist, 2) : '–'} km</strong> between spill centres</span>
              <span>Area change <strong>{a.area_km2 ? `${b.area_km2 >= a.area_km2 ? '+' : ''}${fmtNum(((b.area_km2 - a.area_km2) / a.area_km2) * 100, 0)}%` : '–'}</strong></span>
              {fc && <span>Forecast used: <strong>+{fc.hours} h</strong> from A</span>}
            </div>
            <table className="table">
              <thead><tr><th /><th>A (earlier)</th><th>B (later)</th></tr></thead>
              <tbody>{rows.map(([k, x, y]) => <tr key={k}><td>{k}</td><td>{x}</td><td>{y}</td></tr>)}</tbody>
            </table>
            {hoursApart != null && hoursApart <= 0 && <p className="muted small">B is not later than A, so the forecast check is not shown. Swap A and B.</p>}
          </section>
        </>
      )}
    </div>
  )
}
