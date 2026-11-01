import { useEffect, useState } from 'react'
import api, { errorText } from '../api/client'
import { fmtNum } from './Bits.jsx'

const money = (v, c) => `${c} ${Number(v || 0).toLocaleString()}`

export default function ResponsePanel({ incidentId }) {
  const [r, setR] = useState(null)
  const [opt, setOpt] = useState({ vessel_speed_kn: '', skimmers: '', cost_per_m3_at_sea: '', cost_per_km_shoreline: '' })
  const [error, setError] = useState('')

  const load = () => {
    const params = Object.fromEntries(Object.entries(opt).filter(([, v]) => v !== ''))
    api.get(`/incidents/${incidentId}/response`, { params }).then((x) => setR(x.data)).catch((e) => setError(errorText(e)))
  }
  useEffect(load, [incidentId]) // eslint-disable-line react-hooks/exhaustive-deps

  if (error) return <div className="panel"><p className="warn-note">{error}</p></div>
  if (!r) return <div className="panel"><p className="muted">Loading…</p></div>
  const e = r.equipment, c = r.cost, t = r.timing
  const set = (k) => (ev) => setOpt({ ...opt, [k]: ev.target.value })

  return (
    <div className="panel">
      <h3>Nearest response bases</h3>
      <table className="table">
        <thead><tr><th>Base</th><th>Sea distance</th><th>Travel time</th></tr></thead>
        <tbody>
          {r.nearest.map((s) => (
            <tr key={s.name}><td><strong>{s.name}</strong><div className="small muted">{s.role}</div></td>
              <td>{fmtNum(s.distance_km, 0)} km</td><td>{fmtNum(s.eta_h, 1)} h</td></tr>
          ))}
        </tbody>
      </table>
      <p className={t.arrives_before_landfall ? 'ok-note' : 'warn-note'}>
        {t.first_landfall_h == null
          ? `First unit can be on scene in about ${fmtNum(t.first_unit_eta_h, 1)} h. The oil is not expected to reach the coast within 72 h.`
          : t.arrives_before_landfall
            ? `First unit arrives in about ${fmtNum(t.first_unit_eta_h, 1)} h, before landfall (about ${t.first_landfall_h} h): containment at sea is possible.`
            : `Landfall (about ${t.first_landfall_h} h) is expected before the first unit arrives (${fmtNum(t.first_unit_eta_h, 1)} h): prepare shoreline protection now.`}
      </p>

      <h3>Equipment (first estimate)</h3>
      <dl className="facts">
        <dt>Containment boom</dt><dd>about {Number(e.boom_m).toLocaleString()} m to encircle the slick (perimeter {fmtNum(e.slick_perimeter_km, 1)} km)</dd>
        <dt>Skimmers</dt><dd>{e.skimmers} × {e.skimmer_m3_per_h} m³/h</dd>
        <dt>Oil that can be recovered at sea</dt><dd>{e.recoverable_m3[0]}–{e.recoverable_m3[1]} m³ (about {e.skimming_hours[0]}–{e.skimming_hours[1]} h of skimming)</dd>
      </dl>

      <h3>Cost estimate</h3>
      <table className="table">
        <tbody>
          <tr><td>Recovery at sea ({c.volume_m3[0]}–{c.volume_m3[1]} m³)</td><td>{money(c.at_sea[0], c.currency)} – {money(c.at_sea[1], c.currency)}</td></tr>
          <tr><td>Shoreline cleaning ({fmtNum(c.shoreline_km, 1)} km)</td><td>{money(c.shoreline, c.currency)}</td></tr>
          <tr><td><strong>Total</strong></td><td><strong>{money(c.total[0], c.currency)} – {money(c.total[1], c.currency)}</strong></td></tr>
          {c.avoidable_if_contained_at_sea > 0 && (
            <tr><td>Shoreline cost avoidable by early containment</td><td className="ok">{money(c.avoidable_if_contained_at_sea, c.currency)}</td></tr>
          )}
        </tbody>
      </table>

      <details>
        <summary className="small">Change the assumptions</summary>
        <div className="form-grid" style={{ marginTop: '.5rem' }}>
          <label>Vessel speed (knots)<input value={opt.vessel_speed_kn} onChange={set('vessel_speed_kn')} placeholder={r.assumptions.vessel_speed_kn} /></label>
          <label>Number of skimmers<input value={opt.skimmers} onChange={set('skimmers')} placeholder={r.assumptions.skimmers} /></label>
          <label>Cost per m³ recovered at sea<input value={opt.cost_per_m3_at_sea} onChange={set('cost_per_m3_at_sea')} placeholder={r.assumptions.cost_per_m3_at_sea} /></label>
          <label>Cost per km of shoreline<input value={opt.cost_per_km_shoreline} onChange={set('cost_per_km_shoreline')} placeholder={r.assumptions.cost_per_km_shoreline} /></label>
        </div>
        <button className="btn small" onClick={load}>Recalculate</button>
      </details>
      <p className="muted small">{r.note}</p>
    </div>
  )
}
