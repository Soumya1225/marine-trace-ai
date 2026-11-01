import { Fragment, useEffect, useState } from 'react'
import { CircleMarker, Polygon, Polyline, Tooltip } from 'react-leaflet'
import api, { errorText } from '../api/client'
import BaseMap from './BaseMap.jsx'
import { useAuth } from '../context/AuthContext.jsx'
import SolutionExtra from './SolutionExtra.jsx'

const TIER_COLOR = { 1: '#3E8E6A', 2: '#E8A42A', 3: '#C8322B' }
const STATUS = { pending: 'Pending', in_progress: 'In progress', done: 'Done' }
const ll = (ring) => (ring || []).map(([lon, lat]) => [lat, lon])

function Bar({ value, color }) {
  return (
    <div style={{ background: '#E9EEF0', borderRadius: 4, height: 10, width: '100%' }}>
      <div style={{ width: `${value}%`, background: color, height: 10, borderRadius: 4 }} />
    </div>
  )
}

export default function SolutionPanel({ incidentId }) {
  const { can } = useAuth()
  const [s, setS] = useState(null)
  const [error, setError] = useState('')
  const [open, setOpen] = useState(null)

  useEffect(() => {
    api.get(`/incidents/${incidentId}/solution`).then((r) => setS(r.data)).catch((e) => setError(errorText(e)))
  }, [incidentId])

  const setStatus = (task, status) => {
    api.patch(`/incidents/${incidentId}/solution/tasks/${task}`, { status }).then((r) => setS(r.data)).catch((e) => setError(errorText(e)))
  }

  if (error) return <div className="panel"><p className="warn-note">{error}</p></div>
  if (!s) return <div className="panel"><p className="muted">Working out the response…</p></div>
  const t = s.tier, m = s.method, ic = s.intercept, cl = s.checklist
  const best = m.methods.find((x) => x.key === m.recommended)
  const editable = can('admin', 'analyst')
  const phases = [...new Set(cl.tasks.map((x) => x.phase))]
  const bounds = [[s.slick.lat, s.slick.lon], ...(ic ? [[ic.base.lat, ic.base.lon]] : []), ...s.booms.sites.map((b) => [b.lat, b.lon])]
  const lats = bounds.map((b) => b[0]), lons = bounds.map((b) => b[1])
  const box = [[Math.min(...lats) - 0.02, Math.min(...lons) - 0.02], [Math.max(...lats) + 0.02, Math.max(...lons) + 0.02]]

  return (
    <div className="panel">
      {m.verify_first && (
        <p className="warn-note"><b>Verify first:</b> this detection has low reliability. Confirm it is oil (aircraft or next satellite pass) before sending vessels or chemicals.</p>
      )}

      <h3>1. Spill tier and who leads the response</h3>
      <div style={{ display: 'flex', gap: '1rem', alignItems: 'flex-start', flexWrap: 'wrap' }}>
        <div style={{ border: `2px solid ${TIER_COLOR[t.tier]}`, color: TIER_COLOR[t.tier], borderRadius: 8, padding: '.6rem 1rem', textAlign: 'center', minWidth: 110 }}>
          <div style={{ fontSize: '2rem', fontWeight: 700, lineHeight: 1 }}>Tier {t.tier}</div>
          <div className="small">{t.tonnes[0]}–{t.tonnes[1]} t</div>
        </div>
        <div style={{ flex: 1, minWidth: 260 }}>
          <p style={{ margin: 0 }}><b>{t.meaning}</b></p>
          <p style={{ margin: '.3rem 0' }}>Lead agency: <b>{t.lead_agency}</b></p>
          <p className="small muted" style={{ margin: 0 }}>Inform: {t.inform.join(', ')}</p>
          <ul className="small muted" style={{ margin: '.3rem 0 0', paddingLeft: '1.1rem' }}>{t.reasons.map((r) => <li key={r}>{r}</li>)}</ul>
        </div>
      </div>

      <h3 style={{ marginTop: '1.2rem' }}>2. Best clean-up method</h3>
      <p style={{ margin: '.2rem 0 .6rem' }}><b>{m.summary}</b> <span className="muted small">(wind {m.wind_ms} m/s)</span></p>
      <p className="small" style={{ margin: '0 0 .6rem' }}>{best.what}</p>
      <table className="table">
        <thead><tr><th>Method</th><th style={{ width: '35%' }}>Suitability</th><th /></tr></thead>
        <tbody>{m.methods.map((x) => (
          <Fragment key={x.key}>
            <tr style={x.key === m.recommended ? { background: '#EAF4EE' } : undefined}>
              <td>{x.name}{x.key === m.recommended && <span className="badge pri-low" style={{ marginLeft: '.4rem' }}>Recommended</span>}</td>
              <td><div style={{ display: 'flex', gap: '.5rem', alignItems: 'center' }}><Bar value={x.score} color={x.score >= 60 ? '#3E8E6A' : x.score >= 40 ? '#E8A42A' : '#B0B4BA'} /><span className="small">{x.score}</span></div></td>
              <td><button className="link-btn" onClick={() => setOpen(open === x.key ? null : x.key)}>{open === x.key ? 'Hide' : 'Why?'}</button></td>
            </tr>
            {open === x.key && (
              <tr><td colSpan={3}>
                <div className="small" style={{ margin: '.2rem 0' }}>{x.what}</div>
                <ul className="small" style={{ margin: 0, paddingLeft: '1.1rem' }}>
                  {x.for.map((r) => <li key={r} style={{ color: '#2F6F4F' }}>▲ {r}</li>)}
                  {x.against.map((r) => <li key={r} style={{ color: '#8E1B14' }}>▼ {r}</li>)}
                </ul>
              </td></tr>
            )}
          </Fragment>
        ))}</tbody>
      </table>

      <h3 style={{ marginTop: '1.2rem' }}>3. Where to send the boats and booms</h3>
      {ic ? (
        <p style={{ margin: '.2rem 0 .6rem' }}>
          Send the response vessel from <b>{ic.base.name}</b>. The oil keeps moving, so aim for the <b>intercept point</b> {ic.lat.toFixed(4)}, {ic.lon.toFixed(4)}:
          {' '}{ic.distance_km} km away, reached about <b>{ic.travel_h} h</b> after leaving{ic.reached ? `, where the slick is expected ${ic.hours} h after the image.` : ' (the slick cannot be reached within the 72 h forecast; protect the coast instead).'}
          {' '}At-sea containment boom: about <b>{s.booms.at_sea_boom_m} m</b>.
        </p>
      ) : <p className="muted">No response base found nearby.</p>}
      <BaseMap bounds={box} height={360}>
        {s.slick.polygon && <Polygon positions={ll(s.slick.polygon)} pathOptions={{ color: '#8E1B14', weight: 2, fillOpacity: 0.25 }}><Tooltip>Detected slick</Tooltip></Polygon>}
        <Polyline positions={s.forecast_path} pathOptions={{ color: '#2C6E91', dashArray: '4 6', weight: 2 }}><Tooltip>Forecast path of the slick (72 h)</Tooltip></Polyline>
        {ic && <Polyline positions={ic.route} pathOptions={{ color: '#0F2A3D', weight: 3 }}><Tooltip>Vessel route (straight-line guide)</Tooltip></Polyline>}
        {ic && <CircleMarker center={[ic.base.lat, ic.base.lon]} radius={8} pathOptions={{ color: '#0F2A3D', fillColor: '#0F2A3D', fillOpacity: 1 }}><Tooltip permanent direction="top">{ic.base.name}</Tooltip></CircleMarker>}
        {ic && <CircleMarker center={[ic.lat, ic.lon]} radius={9} pathOptions={{ color: '#E8A42A', fillColor: '#E8A42A', fillOpacity: 0.9 }}><Tooltip permanent direction="bottom">Intercept (+{ic.hours} h)</Tooltip></CircleMarker>}
        {s.booms.sites.map((b) => (
          <CircleMarker key={b.name} center={[b.lat, b.lon]} radius={7} pathOptions={{ color: '#3E8E6A', fillColor: '#3E8E6A', fillOpacity: 0.9 }}>
            <Tooltip>{b.name}: {b.how.toLowerCase()}, {b.boom_m} m, {b.deploy_by_h > 0 ? `deploy before ${b.deploy_by_h} h` : 'deploy immediately'}</Tooltip>
          </CircleMarker>
        ))}
      </BaseMap>
      <p className="small muted">Dark line: vessel route · orange: intercept point · blue dashed: forecast path · green: boom positions.</p>
      {s.booms.sites.length > 0 && (
        <table className="table">
          <thead><tr><th>Protect</th><th>Type</th><th>Oil arrives</th><th>Deploy boom before</th><th>Boom</th></tr></thead>
          <tbody>{s.booms.sites.map((b) => (
            <tr key={b.name}><td>{b.name}</td><td>{b.type}</td><td>{b.eta_h > 0 ? `${b.eta_h} h` : 'Already reached'}</td><td><b>{b.deploy_by_h > 0 ? `${b.deploy_by_h} h` : 'Now'}</b></td><td>{b.how}, {b.boom_m} m</td></tr>
          ))}</tbody>
        </table>
      )}

      <h3 style={{ marginTop: '1.2rem' }}>4. Response checklist</h3>
      <div style={{ display: 'flex', gap: '.6rem', alignItems: 'center', marginBottom: '.5rem' }}>
        <Bar value={cl.progress} color="#3E8E6A" /><span className="small" style={{ whiteSpace: 'nowrap' }}>{cl.done} of {cl.total} done ({cl.progress}%)</span>
      </div>
      {phases.map((ph) => (
        <div key={ph}>
          <h4 style={{ margin: '.8rem 0 .3rem' }}>{ph}</h4>
          <table className="table">
            <tbody>{cl.tasks.filter((x) => x.phase === ph).map((x) => (
              <tr key={x.id}>
                <td style={{ width: '55%', textDecoration: x.status === 'done' ? 'line-through' : 'none' }}>{x.task}</td>
                <td className="small muted">{x.owner}</td>
                <td style={{ whiteSpace: 'nowrap' }}>
                  {editable ? (
                    <select value={x.status} onChange={(e) => setStatus(x.id, e.target.value)}>
                      {Object.entries(STATUS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
                    </select>
                  ) : STATUS[x.status]}
                  {x.updated_by && <div className="small muted">{x.updated_by}</div>}
                </td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      ))}
      <SolutionExtra incidentId={incidentId} />

      <p className="small muted" style={{ marginTop: '1rem' }}>{s.note}</p>
    </div>
  )
}
