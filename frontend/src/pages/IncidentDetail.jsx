import { useEffect, useMemo, useState } from 'react'
import { CircleMarker, ImageOverlay, LayersControl, LayerGroup, Polygon, Polyline, Tooltip } from 'react-leaflet'
import { Link, useNavigate, useParams } from 'react-router-dom'
import api, { download, errorText } from '../api/client'
import BaseMap, { LEVEL_COLOR } from '../components/BaseMap.jsx'
import { ErrorNote, PriorityBadge, ScoreBar, StatusBadge, fmtNum, fmtTime } from '../components/Bits.jsx'
import { useAuth } from '../context/AuthContext.jsx'
import AssistantPanel from '../components/AssistantPanel.jsx'
import ResponsePanel from '../components/ResponsePanel.jsx'
import OpticalPanel from '../components/OpticalPanel.jsx'
import SolutionPanel from '../components/SolutionPanel.jsx'
import '../shap.css'
import '../infra.css'
import '../lookalike.css'
import '../cases.css'
import '../zones.css'

const TABS = ['Ask AI', 'Suspects', 'Solution', 'Forecast', 'Impact', 'Optical', 'Response', 'Actions', 'Notes', 'Timeline']
const STATUSES = [['detected', 'Detected'], ['under_review', 'Under review'], ['confirmed', 'Confirmed'], ['closed', 'Closed'], ['rejected', 'Rejected']]
const etaText = (h) => (h == null || h <= 0 ? 'already within reach' : `in about ${h} h`)
const SHAP_LABEL = {
  min_dist_km: 'Distance to back-tracked spill', alignment: 'Course along spill axis', overlap: 'Spill lies on its path',
  hours_before: 'Time before image', max_gap_h: 'AIS gap', speed_ratio: 'Slow-down near spill',
  loiter_h: 'Loitering', type_risk: 'Vessel type', length_m: 'Vessel length',
}

function ShapBars({ shap }) {
  const items = Object.entries(shap || {}).filter(([, v]) => Math.abs(v) >= 0.01).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 5)
  if (!items.length) return null
  const max = Math.max(...items.map(([, v]) => Math.abs(v)))
  return (
    <div className="shap">
      <div className="muted small">Why XGBoost scored it this way (SHAP):</div>
      {items.map(([k, v]) => (
        <div key={k} className="shap-row" title={`${v > 0 ? '+' : ''}${v.toFixed(2)} (log-odds)`}>
          <span className="shap-label">{SHAP_LABEL[k] || k}</span>
          <span className="shap-track"><span className={v > 0 ? 'shap-bar up' : 'shap-bar down'} style={{ width: `${(Math.abs(v) / max) * 100}%` }} /></span>
          <span className="shap-val">{v > 0 ? '▲ raises' : '▼ lowers'}</span>
        </div>
      ))}
    </div>
  )
}

const ll = (ring) => (ring || []).map(([lon, lat]) => [lat, lon])

function positionAt(track, t) {
  if (!track?.length) return null
  const times = track.map((p) => new Date(p.t).getTime())
  if (t < times[0] || t > times[times.length - 1]) return null
  let i = times.findIndex((x) => x >= t)
  if (i <= 0) return [track[0].lat, track[0].lon]
  const f = (t - times[i - 1]) / (times[i] - times[i - 1] || 1)
  return [track[i - 1].lat + f * (track[i].lat - track[i - 1].lat), track[i - 1].lon + f * (track[i].lon - track[i - 1].lon)]
}

export default function IncidentDetail() {
  const { id } = useParams()
  const navigate = useNavigate()
  const { can } = useAuth()
  const [inc, setInc] = useState(null)
  const [tab, setTab] = useState('Suspects')
  const [selected, setSelected] = useState(null)
  const [hours, setHours] = useState(24)
  const [time, setTime] = useState(null)
  const [error, setError] = useState('')
  const [statusNote, setStatusNote] = useState('')
  const [note, setNote] = useState('')
  const [people, setPeople] = useState([])
  const [assignTo, setAssignTo] = useState('')
  const [dueAt, setDueAt] = useState('')
  const [events, setEvents] = useState(null)
  useEffect(() => { api.get('/cases/assignees').then((r) => setPeople(r.data)).catch(() => {}) }, [])
  useEffect(() => { if (tab === 'Timeline') api.get(`/incidents/${id}/timeline`).then((r) => setEvents(r.data)).catch(() => setEvents([])) }, [tab, id, inc])

  const load = () => api.get(`/incidents/${id}`).then((r) => setInc(r.data)).catch((e) => setError(errorText(e)))
  useEffect(() => { load() }, [id]) // eslint-disable-line react-hooks/exhaustive-deps

  const range = useMemo(() => {
    const all = (inc?.suspects || []).flatMap((s) => s.track.map((p) => new Date(p.t).getTime()))
    return all.length ? [Math.min(...all), Math.max(...all)] : null
  }, [inc])
  useEffect(() => { if (range) setTime(range[1]) }, [range])

  // Playback: ships move along their AIS tracks, then the oil forecast steps forward hour by hour
  const [playing, setPlaying] = useState(null)
  useEffect(() => {
    if (!playing || !inc) return undefined
    const hs = (inc.forecast || []).map((f) => f.hours)
    const timer = setInterval(() => {
      if (playing === 'ships' && range) {
        setTime((t) => {
          const next = (t ?? range[0]) + (range[1] - range[0]) / 120
          if (next >= range[1]) { setPlaying(hs.length ? 'forecast-start' : null); return range[1] }
          return next
        })
      } else if (playing === 'forecast-start') {
        setHours(hs[0]); setPlaying('forecast')
      } else if (playing === 'forecast') {
        setHours((h) => {
          const i = hs.indexOf(h)
          if (i < 0 || i >= hs.length - 1) { setPlaying(null); return hs[hs.length - 1] }
          return hs[i + 1]
        })
      }
    }, playing === 'ships' ? 80 : 1000)
    return () => clearInterval(timer)
  }, [playing, inc, range])
  const play = (what) => {
    if (playing) { setPlaying(null); return }
    if (what === 'ships' && range) { setTime(range[0]); setPlaying('ships') }
    else if ((inc?.forecast || []).length) { setPlaying('forecast-start') }
  }

  if (error) return <ErrorNote text={error} />
  if (!inc) return <div className="center-note">Loading…</div>

  const b = inc.scene.bounds
  const bounds = [[b[1], b[0]], [b[3], b[2]]]
  const forecast = (inc.forecast || []).find((f) => f.hours === hours)
  const editable = can('admin', 'analyst')
  const deleteIncident = async () => {
    if (!window.confirm(`Delete incident #${id} with its suspects, notes and alerts? This cannot be undone.`)) return
    try { await api.delete(`/incidents/${id}`); navigate('/incidents') } catch (e) { setError(errorText(e)) }
  }

  const changeStatus = async (status) => {
    try { await api.patch(`/incidents/${id}/status`, { status, note: statusNote || null }); setStatusNote(''); load() } catch (e) { setError(errorText(e)) }
  }
  const openSummary = async (lang) => {
    const w = window.open('', '_blank')
    try {
      const r = await api.get(`/incidents/${id}/summary.html`, { params: { lang }, responseType: 'text' })
      w.document.open(); w.document.write(r.data); w.document.close()
    } catch (e) { w.close(); setError(errorText(e)) }
  }
  const assign = async () => {
    try {
      await api.patch(`/incidents/${id}/assign`, { user_id: assignTo ? Number(assignTo) : null, due_at: dueAt ? new Date(dueAt).toISOString() : null })
      setDueAt(''); load()
    } catch (e) { setError(errorText(e)) }
  }
  const review = async (s, review_status) => {
    try { await api.patch(`/suspects/${s.id}`, { review_status }); load() } catch (e) { setError(errorText(e)) }
  }
  const addNote = async (e) => {
    e.preventDefault()
    try { await api.post(`/incidents/${id}/notes`, { text: note }); setNote(''); load() } catch (err) { setError(errorText(err)) }
  }

  return (
    <div className="page wide">
      <header className="page-head">
        <div>
          <p className="muted small"><Link to="/incidents">Incidents</Link> / #{inc.id}</p>
          <h1>{inc.title}</h1>
          <p className="head-meta">
            <PriorityBadge value={inc.priority} /> <StatusBadge value={inc.status} />
            {(inc.zones || []).map((z) => <span key={z} className="zone-tag">📍 {z}</span>)}
            <span>Image {fmtTime(inc.detected_at)}</span>
            <span>{fmtNum(inc.area_km2, 3)} km²</span>
            <span>Detection confidence {fmtNum(inc.confidence)}</span>
            {inc.spill?.model_disagreement != null && <span>Model disagreement {fmtNum(inc.spill.model_disagreement)}</span>}
            {inc.spill?.wind_speed_ms != null && <span>Wind {fmtNum(inc.spill.wind_speed_ms, 1)} m/s</span>}
            {inc.spill?.reliability && (
              <span className={`badge ${inc.spill.reliability === 'low' ? 'pri-medium' : 'pri-low'}`}>
                {inc.spill.reliability === 'low' ? 'Low reliability' : 'Reliable conditions'}
              </span>
            )}
          </p>
          {inc.spill?.infrastructure?.possible_non_ship_source && (
            <p className="warn-note">{inc.spill.infrastructure.note}</p>
          )}
          {inc.spill?.reliability_notes?.length > 0 && <p className="warn-note">{inc.spill.reliability_notes.join(' ')}</p>}
        </div>
        <div className="head-actions">
          <button className="btn primary" onClick={() => download(`/incidents/${id}/report`, `incident_${id}_report.pdf`)}>Download PDF report</button>
          <button className="btn" onClick={() => download(`/incidents/${id}/export/geojson`, `incident_${id}.geojson`)}>Export GeoJSON</button>
          <button className="btn" onClick={() => download(`/incidents/${id}/export/suspects.csv`, `incident_${id}_suspects.csv`)}>Export suspects CSV</button>
          <button className="btn" onClick={() => download(`/incidents/${id}/export/evidence.zip`, `incident_${id}_evidence.zip`)}>Evidence package (.zip)</button>
          <button className="btn" onClick={() => openSummary('en')}>One-page summary</button>
          <button className="btn" onClick={() => openSummary('hi')}>सारांश (हिंदी)</button>
          {can('admin') && <button className="btn danger" onClick={deleteIncident}>Delete incident</button>}
        </div>
      </header>

      <div className="incident-grid">
        <section className="map-col">
          <BaseMap bounds={bounds} height={560}>
            <LayersControl position="topright">
              <LayersControl.Overlay checked name="Radar image">
                <ImageOverlay url={inc.scene.sar_url} bounds={bounds} opacity={0.85} />
              </LayersControl.Overlay>
              <LayersControl.Overlay checked name="Detection mask">
                <ImageOverlay url={inc.scene.mask_url} bounds={bounds} opacity={0.9} />
              </LayersControl.Overlay>
              {inc.scene.uncertainty_url && (
                <LayersControl.Overlay name="Model disagreement (ensemble)">
                  <ImageOverlay url={inc.scene.uncertainty_url} bounds={bounds} opacity={0.9} />
                </LayersControl.Overlay>
              )}
              <LayersControl.Overlay checked name="Spill outline">
                <Polygon positions={ll(inc.spill.polygon)} pathOptions={{ color: '#8E1B14', weight: 2, fillOpacity: 0.15 }}>
                  <Tooltip>Detected spill, {fmtNum(inc.area_km2, 3)} km²</Tooltip>
                </Polygon>
              </LayersControl.Overlay>
              <LayersControl.Overlay checked name="Back-tracked drift">
                <Polyline positions={(inc.backtrack || []).map((p) => [p[1], p[0]])} pathOptions={{ color: '#7A3B12', dashArray: '6 6', weight: 2 }} />
              </LayersControl.Overlay>
              <LayersControl.Overlay checked name="Forecast">
                <LayerGroup>
                  {forecast?.uncertainty_polygon?.length > 0 && (
                    <Polygon positions={ll(forecast.uncertainty_polygon)} pathOptions={{ color: '#2C6E91', fillOpacity: 0.06, weight: 1, dashArray: '4 4' }}>
                      <Tooltip>Uncertainty zone after {forecast.hours} h (wind stronger/weaker and turned ±20°)</Tooltip>
                    </Polygon>
                  )}
                  {forecast && (
                    <Polygon positions={ll(forecast.polygon)} pathOptions={{ color: '#2C6E91', fillOpacity: 0.18, weight: 1.5 }}>
                      <Tooltip>Predicted oil extent after {forecast.hours} h</Tooltip>
                    </Polygon>
                  )}
                  {forecast?.lstm_centroid && (
                    <CircleMarker center={forecast.lstm_centroid} radius={7} pathOptions={{ color: '#E8A42A', fillColor: '#E8A42A', fillOpacity: 0.9, weight: 2 }}>
                      <Tooltip>LSTM forecast centre after {forecast.hours} h</Tooltip>
                    </CircleMarker>
                  )}
                  {forecast?.particles.map((p, i) => <CircleMarker key={i} center={[p[1], p[0]]} radius={1.5} pathOptions={{ color: '#2C6E91', weight: 0, fillOpacity: 0.7 }} />)}
                </LayerGroup>
              </LayersControl.Overlay>
              <LayersControl.Overlay checked name="Ship tracks">
                <LayerGroup>
                  {inc.suspects.map((s) => (
                    <Polyline key={s.id} positions={s.track.map((p) => [p.lat, p.lon])}
                      pathOptions={{ color: LEVEL_COLOR[s.level], weight: selected === s.id ? 5 : 2.5, opacity: selected && selected !== s.id ? 0.35 : 0.95 }}
                      eventHandlers={{ click: () => { setSelected(s.id); setTab('Suspects') } }}>
                      <Tooltip>#{s.rank} {s.vessel_name} ({fmtNum(s.score)})</Tooltip>
                    </Polyline>
                  ))}
                  {time && inc.suspects.map((s) => {
                    const pos = positionAt(s.track, time)
                    return pos && <CircleMarker key={`p${s.id}`} center={pos} radius={6} pathOptions={{ color: '#0F2A3D', fillColor: LEVEL_COLOR[s.level], fillOpacity: 1, weight: 2 }}><Tooltip>{s.vessel_name}</Tooltip></CircleMarker>
                  })}
                </LayerGroup>
              </LayersControl.Overlay>
              <LayersControl.Overlay checked name="Ships seen by radar">
                <LayerGroup>
                  {(inc.ships?.targets || []).map((t, i) => (
                    <CircleMarker key={`t${i}`} center={[t.lat, t.lon]} radius={t.dark ? 7 : t.fixed ? 5 : 3}
                      pathOptions={t.dark ? { color: '#6B2FA0', fillColor: '#6B2FA0', fillOpacity: 0.85, weight: 2 }
                        : t.fixed ? { color: '#0F2A3D', fillColor: '#FFFFFF', fillOpacity: 1, weight: 2 }
                        : { color: '#5F6F7A', fillOpacity: 0.6, weight: 1 }}>
                      <Tooltip>{t.dark ? 'Ship with NO AIS signal (dark vessel)'
                        : t.fixed ? `Fixed structure, likely an oil platform (${t.fixed_note})`
                        : `Radar target matched to ${t.vessel_name || 'MMSI ' + t.mmsi}`}</Tooltip>
                    </CircleMarker>
                  ))}
                </LayerGroup>
              </LayersControl.Overlay>
              {inc.spill?.infrastructure?.nearby?.length > 0 && (
                <LayersControl.Overlay checked name="Oil platforms & pipelines">
                  <LayerGroup>
                    {inc.spill.infrastructure.nearby.map((f) => (f.geometry.type === 'Point'
                      ? <CircleMarker key={f.name} center={[f.geometry.coordinates[1], f.geometry.coordinates[0]]} radius={6}
                          pathOptions={{ color: '#111111', fillColor: '#F28C28', fillOpacity: 1, weight: 2 }}>
                          <Tooltip>{f.name} ({f.operator}), {f.distance_km} km from the spill</Tooltip>
                        </CircleMarker>
                      : <Polyline key={f.name} positions={f.geometry.coordinates.map(([lon, lat]) => [lat, lon])}
                          pathOptions={{ color: '#F28C28', weight: 3, dashArray: '10 5' }}>
                          <Tooltip>{f.name}, {f.distance_km} km from the spill</Tooltip>
                        </Polyline>))}
                  </LayerGroup>
                </LayersControl.Overlay>
              )}
              <LayersControl.Overlay checked name="Threatened sites">
                <LayerGroup>
                  {(inc.impact?.threatened_sites || []).map((t) => (
                    <CircleMarker key={t.name} center={[t.lat, t.lon]} radius={7} pathOptions={{ color: '#3E8E6A', fillOpacity: 0.8 }}>
                      <Tooltip>{t.name}: oil {etaText(t.eta_h)}</Tooltip>
                    </CircleMarker>
                  ))}
                </LayerGroup>
              </LayersControl.Overlay>
            </LayersControl>
          </BaseMap>

          <div className="map-controls">
            <div className="play-row">
              <button className="btn primary" onClick={() => play('ships')} disabled={!range && !playing}
                title={range ? 'Ships move along their AIS tracks, then the oil forecast plays' : 'No AIS ship tracks for this spill'}>
                {playing ? '■ Stop' : '▶ Play ships, then oil forecast'}
              </button>
              <button className="btn" onClick={() => play('forecast')} disabled={!!playing || !(inc.forecast || []).length}>▶ Play oil forecast only</button>
              {playing && <span className="muted small">{playing === 'ships' ? 'Ships moving along their AIS tracks…' : `Oil forecast: +${hours} h`}</span>}
              {!playing && !range && <span className="muted small">No ship tracks for this spill (no AIS ship passed close enough), so only the oil forecast can be played.</span>}
            </div>
            {range && (
              <label className="slider">
                <span>Ship positions at {fmtTime(new Date(time).toISOString())}</span>
                <input type="range" min={range[0]} max={range[1]} step={60000} value={time || range[1]} onChange={(e) => setTime(Number(e.target.value))} />
              </label>
            )}
            <label className="slider">
              <span>Forecast: +{hours} h</span>
              <input type="range" min={0} max={(inc.forecast || []).length - 1} step={1}
                value={(inc.forecast || []).findIndex((f) => f.hours === hours)}
                onChange={(e) => setHours(inc.forecast[Number(e.target.value)].hours)} />
            </label>
          </div>

          {inc.spill?.lookalike_check && (
            <div className="panel lookalike-box">
              <h3>Look-alike check (second stage)</h3>
              <p className="muted small">Physical evidence the detection models do not see directly: wind, contrast, edges, shape and texture{inc.spill.lookalike_check.learned_oil_prob != null ? `, plus a model trained on expert-labelled oil slicks and look-alikes (rules ${fmtNum(inc.spill.lookalike_check.rule_score)}, trained model ${fmtNum(inc.spill.lookalike_check.learned_oil_prob)}, shown for information)` : ''}.</p>
              <div className="la-score">
                <span className={`badge ${inc.spill.lookalike_check.verdict === 'Likely oil' ? 'pri-high' : inc.spill.lookalike_check.verdict === 'Uncertain' ? 'pri-medium' : 'pri-low'}`}>{inc.spill.lookalike_check.verdict}</span>
                <div className="la-bar"><div style={{ width: `${inc.spill.lookalike_check.oil_score * 100}%` }} /></div>
                <strong>oil score {fmtNum(inc.spill.lookalike_check.oil_score)}</strong>
              </div>
              <ul className="plain la-list">
                {inc.spill.lookalike_check.evidence.map((e) => (
                  <li key={e.name}><span className={e.effect > 0 ? 'up' : 'down'}>{e.effect > 0 ? '▲ oil' : '▼ look-alike'}</span> <strong>{e.name}</strong> ({e.value}): {e.text}</li>
                ))}
              </ul>
            </div>
          )}
          {editable && (
            <div className="panel status-box">
              <h3>Case status</h3>
              <div className="status-row">
                {STATUSES.map(([v, label]) => (
                  <button key={v} className={inc.status === v ? 'btn small on' : 'btn small'} onClick={() => changeStatus(v)} disabled={inc.status === v}>{label}</button>
                ))}
              </div>
              <input placeholder="Reason for the change (optional, saved as a note)" value={statusNote} onChange={(e) => setStatusNote(e.target.value)} />
              <h3 style={{ marginTop: '1rem' }}>Assignment</h3>
              <p className="small">
                {inc.assigned_to ? <>Assigned to <strong>{inc.assigned_to.name}</strong></> : 'Not assigned yet'}
                {inc.due_at && <>, due {fmtTime(inc.due_at)}</>}
                {inc.overdue && <span className="overdue-badge"> · OVERDUE</span>}
              </p>
              <div className="assign-row">
                <label>Analyst
                  <select value={assignTo} onChange={(e) => setAssignTo(e.target.value)}>
                    <option value="">Unassigned</option>
                    {people.map((p) => <option key={p.id} value={p.id}>{p.name} ({p.role})</option>)}
                  </select>
                </label>
                <label>Deadline (empty = by priority)
                  <input type="datetime-local" value={dueAt} onChange={(e) => setDueAt(e.target.value)} />
                </label>
                <button className="btn" onClick={assign}>Save</button>
              </div>
            </div>
          )}
          {inc.evidence && (
            <div className="panel">
              <h3>Evidence fingerprints</h3>
              <p className="muted small">SHA-256 of the input files, analysed {fmtTime(inc.evidence.analysed_at_utc)}. If a file is changed later, its fingerprint will not match.</p>
              <table className="table">
                <tbody>{inc.evidence.files.map((f) => <tr key={f.sha256}><td>{f.role}</td><td>{f.name}</td><td><code>{f.sha256.slice(0, 16)}…</code></td></tr>)}</tbody>
              </table>
            </div>
          )}
        </section>

        <section className="side-col panel">
          <div className="tabs" role="tablist">
            {TABS.map((t) => <button key={t} role="tab" aria-selected={tab === t} className={tab === t ? 'tab on' : 'tab'} onClick={() => setTab(t)}>{t}</button>)}
          </div>

          {tab === 'Suspects' && (
            <div className="tab-body">
              {inc.spill?.infrastructure && (
                <div className={inc.spill.infrastructure.possible_non_ship_source ? 'infra-box warn' : 'infra-box'}>
                  <h3>{inc.spill.infrastructure.possible_non_ship_source
                    ? (inc.spill.infrastructure.level === 'likely' ? 'Likely non-ship source' : 'Possible non-ship source')
                    : 'Platforms & pipelines'}</h3>
                  <p className="small">{inc.spill.infrastructure.note}</p>
                  {inc.spill.infrastructure.nearby.length > 0 && (
                    <table className="table">
                      <thead><tr><th>Installation</th><th>Type</th><th>From spill</th><th>From drift path</th></tr></thead>
                      <tbody>{inc.spill.infrastructure.nearby.map((f) => (
                        <tr key={f.name}><td title={f.role}>{f.name}</td><td>{f.kind}</td><td>{f.distance_km} km</td>
                          <td>{f.drift_distance_km != null ? `${f.drift_distance_km} km` : '–'}</td></tr>))}
                      </tbody>
                    </table>
                  )}
                  <p className="muted small">Orange on the map. Positions are approximate (public NOAA/BOEM maps).</p>
                </div>
              )}
              {inc.ships?.targets?.some((t) => t.fixed) && (
                <p className="muted small">{inc.ships.targets.filter((t) => t.fixed).length} bright radar targets are fixed structures (oil platforms), not ships: they match a known platform position or sit at the same spot in another image taken hours apart. They are shown as white circles and are not counted as dark vessels.</p>
              )}
              {inc.suspects.length === 0 && <p className="muted">No ship passed close enough in the AIS data. Check satellite AIS or radar for vessels with their transponder off.</p>}
              {inc.spill?.nearby_dark_vessels?.length > 0 && (
                <div className="dark-box">
                  <h3>Ships without AIS near the spill</h3>
                  <p className="muted small">Bright radar targets with no AIS ship nearby at the image time. They may have switched their transponder off. Oil platforms also look like this, so verify.</p>
                  <ul className="plain">
                    {inc.spill.nearby_dark_vessels.slice(0, 6).map((d, i) => (
                      <li key={i}>{d.lat.toFixed(4)}, {d.lon.toFixed(4)}: {d.distance_km} km from the spill</li>
                    ))}
                  </ul>
                </div>
              )}
              {inc.suspects.map((s) => (
                <article key={s.id} className={selected === s.id ? 'suspect on' : 'suspect'} onClick={() => setSelected(s.id)}>
                  <div className="suspect-head">
                    <span className="rank">#{s.rank}</span>
                    <div>
                      <Link to={`/vessels/${s.mmsi}`} className="vessel-name">{s.vessel_name}</Link>
                      <div className="muted small">MMSI {s.mmsi}, {s.vessel_type}</div>
                    </div>
                    <span className={`review rv-${s.review_status}`}>{s.review_status}</span>
                  </div>
                  <ScoreBar value={s.score} level={s.level} />
                  <div className="muted small">Rule score {fmtNum(s.rule_score)}{s.ml_score != null ? `, XGBoost ${fmtNum(s.ml_score)}` : ''}{s.features?.attention_score != null ? `, cross-attention ${fmtNum(s.features.attention_score)}` : ''}, share of suspicion {(s.confidence * 100).toFixed(0)}%</div>
                  <ul className="evidence">{s.evidence.map((e, i) => <li key={i}>{e}</li>)}</ul>
                  <ShapBars shap={s.features?.shap} />
                  {editable && (
                    <div className="row-btns">
                      <button className="btn small" onClick={() => review(s, 'confirmed')}>Confirm suspect</button>
                      <button className="btn small" onClick={() => review(s, 'dismissed')}>Dismiss</button>
                      {s.review_status !== 'pending' && <button className="link-btn" onClick={() => review(s, 'pending')}>Reset</button>}
                    </div>
                  )}
                </article>
              ))}
            </div>
          )}

          {tab === 'Suspects' && inc.spill?.excluded_responders?.length > 0 && (
            <div className="tab-body">
              <h3>Excluded response vessels</h3>
              <p className="muted small">These ships are near the spill because they respond to it (rescue, fire, pilot, anti-pollution), so they are not suspects.</p>
              <p>{inc.spill.excluded_responders.map((x) => x.vessel_name).join(', ')}</p>
            </div>
          )}

          {tab === 'Forecast' && (
            <div className="tab-body">
              <p className="muted small">Solid area: most likely oil position. Dashed area: uncertainty zone (wind 20% stronger/weaker and turned ±20°). Particle model: surface current plus 3% of the wind. Wind {fmtNum(inc.environment?.wind_u, 1)}/{fmtNum(inc.environment?.wind_v, 1)} m/s, current {fmtNum(inc.environment?.current_u, 2)}/{fmtNum(inc.environment?.current_v, 2)} m/s ({inc.environment?.source}).</p>
              {inc.environment?.notes?.length > 0 && (
                <p className="muted small">Weather data: {inc.environment.notes.join(' · ')}</p>
              )}
              <table className="table">
                <thead><tr><th>After</th><th>Centre (physics)</th><th>Centre (LSTM)</th><th>On shore</th><th>Time (UTC)</th></tr></thead>
                <tbody>
                  {inc.forecast.map((f) => (
                    <tr key={f.hours} className={f.hours === hours ? 'hl' : ''} onClick={() => setHours(f.hours)}>
                      <td>{f.hours} h</td><td>{f.centroid_lat.toFixed(4)}, {f.centroid_lon.toFixed(4)}</td>
                      <td>{f.lstm_centroid ? `${f.lstm_centroid[0].toFixed(4)}, ${f.lstm_centroid[1].toFixed(4)}` : '–'}</td>
                      <td>{(f.beached_fraction * 100).toFixed(0)}%</td><td>{fmtTime(f.time)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {tab === 'Impact' && (
            <div className="tab-body">
              <dl className="facts">
                <dt>Severity</dt><dd>{inc.impact.severity_score} / 100</dd>
                <dt>Estimated volume</dt><dd>{inc.impact.volume.m3[0]}–{inc.impact.volume.m3[1]} m³ ({inc.impact.volume.barrels[0]}–{inc.impact.volume.barrels[1]} barrels)</dd>
                <dt>First landfall (main slick)</dt><dd>{inc.impact.first_landfall_h != null ? `after about ${inc.impact.first_landfall_h} h` : 'not within 72 h'}</dd>
                <dt>Coastline affected</dt><dd>{inc.impact.coastline_affected_km} km</dd>
              </dl>
              <h3>Threatened sites</h3>
              {inc.impact.threatened_sites.length === 0 ? <p className="muted">No listed sensitive site is reached within 72 h.</p> : (
                <table className="table">
                  <thead><tr><th>Site</th><th>Type</th><th>Oil reaches site</th><th>Sensitivity</th></tr></thead>
                  <tbody>{inc.impact.threatened_sites.map((t) => <tr key={t.name}><td>{t.name}</td><td>{t.type}</td><td>{etaText(t.eta_h)}</td><td>{t.sensitivity}/5</td></tr>)}</tbody>
                </table>
              )}
              {inc.impact.threatened_sites.some((t) => !t.eta_h) && <p className="muted small">"Already within reach" means the site is close enough to the slick now that oil can reach it at any time.</p>}
              <p className="muted small">Volume uses a 0.3–50 µm thickness range: radar shows the area of a slick, not its thickness.</p>
            </div>
          )}

          {tab === 'Actions' && (
            <div className="tab-body">
              <p><PriorityBadge value={inc.decision.priority} /> priority score {inc.decision.priority_score}</p>
              <blockquote className="alert-text">{inc.decision.alert_message}</blockquote>
              <ol className="actions">
                {inc.decision.actions.map((a, i) => (
                  <li key={i} className={`urg-${a.urgency}`}><strong>{a.action}</strong><span>{a.detail}</span></li>
                ))}
              </ol>
            </div>
          )}

          {tab === 'Ask AI' && <AssistantPanel incidentId={id} />}
          {tab === 'Response' && <ResponsePanel incidentId={id} />}
          {tab === 'Solution' && <SolutionPanel incidentId={id} />}
          {tab === 'Optical' && <OpticalPanel incidentId={id} />}
          {tab === 'Timeline' && (
            <div className="panel">
              <p className="muted small">Everything that happened to this case, in order.</p>
              {!events ? <p className="muted">Loading…</p> : (
                <ul className="timeline">
                  {events.map((e, i) => (
                    <li key={i} className={`k-${e.kind}`}>
                      <div className="small muted">{fmtTime(e.at)}{e.who ? ` · ${e.who}` : ''}</div>
                      <div>{e.text}</div>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}
          {tab === 'Notes' && (
            <div className="tab-body">
              {inc.notes.length === 0 && <p className="muted">No notes yet.</p>}
              {inc.notes.map((n) => <div key={n.id} className="note"><div className="muted small">{n.user}, {fmtTime(n.created_at)}</div>{n.text}</div>)}
              {editable && (
                <form onSubmit={addNote} className="note-form">
                  <textarea value={note} onChange={(e) => setNote(e.target.value)} placeholder="Add an observation, a call made, or a sample taken" rows={3} required />
                  <button className="btn">Add note</button>
                </form>
              )}
            </div>
          )}
        </section>
      </div>
    </div>
  )
}
