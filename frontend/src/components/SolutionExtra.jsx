import { useEffect, useState } from 'react'
import api, { errorText } from '../api/client'
import { useAuth } from '../context/AuthContext.jsx'

const LANGS = [['en', 'English'], ['hi', 'हिंदी'], ['or', 'ଓଡ଼ିଆ']]
const fmtTime = (t) => (t ? new Date(t).toLocaleString() : '–')

function Bar({ value, color = '#3E8E6A' }) {
  return (
    <div style={{ background: '#E9EEF0', borderRadius: 4, height: 10, width: '100%' }}>
      <div style={{ width: `${Math.min(value || 0, 100)}%`, background: color, height: 10, borderRadius: 4 }} />
    </div>
  )
}

export default function SolutionExtra({ incidentId }) {
  const { can } = useAuth()
  const editable = can('admin', 'analyst')
  const [x, setX] = useState(null)
  const [fu, setFu] = useState(null)
  const [fuBusy, setFuBusy] = useState(false)
  const [lang, setLang] = useState('en')
  const [copied, setCopied] = useState(false)
  const [form, setForm] = useState({ date: '', oil_m3: '', shore_km: '', cost: '', note: '' })
  const [error, setError] = useState('')

  useEffect(() => {
    api.get(`/incidents/${incidentId}/solution/extra`).then((r) => setX(r.data)).catch((e) => setError(errorText(e)))
  }, [incidentId])

  const findPasses = () => {
    setFuBusy(true); setError('')
    api.get(`/incidents/${incidentId}/solution/followup`).then((r) => setFu(r.data)).catch((e) => setError(errorText(e))).finally(() => setFuBusy(false))
  }
  const copy = () => {
    navigator.clipboard?.writeText(x.warnings[lang]).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500) })
  }
  const saveText = () => {
    const blob = new Blob([x.warnings[lang]], { type: 'text/plain;charset=utf-8' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob); a.download = `incident_${incidentId}_warning_${lang}.txt`; a.click()
    URL.revokeObjectURL(a.href)
  }
  const addEntry = (e) => {
    e.preventDefault()
    const num = (v) => (v === '' ? 0 : Number(v))
    if ([form.oil_m3, form.shore_km, form.cost].every((v) => v === '' || Number(v) === 0)) { setError('Enter at least one amount.'); return }
    if ([form.oil_m3, form.shore_km, form.cost].some((v) => v !== '' && (Number.isNaN(Number(v)) || Number(v) < 0))) { setError('Amounts must be positive numbers.'); return }
    setError('')
    api.post(`/incidents/${incidentId}/solution/cleanup`, { date: form.date || null, oil_m3: num(form.oil_m3), shore_km: num(form.shore_km), cost: num(form.cost), note: form.note || null })
      .then((r) => { setX({ ...x, cleanup: r.data }); setForm({ date: '', oil_m3: '', shore_km: '', cost: '', note: '' }) })
      .catch((err) => setError(errorText(err)))
  }
  const delEntry = (i) => {
    api.delete(`/incidents/${incidentId}/solution/cleanup/${i}`).then((r) => setX({ ...x, cleanup: r.data })).catch((err) => setError(errorText(err)))
  }
  const openCaseFile = async () => {
    const w = window.open('', '_blank')
    try {
      const r = await api.get(`/incidents/${incidentId}/casefile.html`, { responseType: 'text' })
      w.document.open(); w.document.write(r.data); w.document.close()
    } catch (err) { w.close(); setError(errorText(err)) }
  }

  if (!x) return error ? <p className="warn-note">{error}</p> : <p className="muted">Loading…</p>
  const c = x.cleanup, set = (k) => (ev) => setForm({ ...form, [k]: ev.target.value })
  const money = (v) => `${c.targets.currency} ${Math.round(v || 0).toLocaleString()}`

  return (
    <div>
      {error && <p className="warn-note">{error}</p>}

      <h3 style={{ marginTop: '1.4rem' }}>5. Next satellite passes (confirm and track the slick)</h3>
      {!fu ? (
        <p><button className="btn" onClick={findPasses} disabled={fuBusy}>{fuBusy ? 'Searching the Copernicus catalogue…' : 'Find next satellite passes'}</button>
          <span className="small muted" style={{ marginLeft: '.5rem' }}>Needs internet; takes a few seconds.</span></p>
      ) : (
        <>
          <p style={{ margin: '.2rem 0 .6rem' }}>{fu.advice}</p>
          {fu.images_after.length > 0 && (
            <table className="table">
              <thead><tr><th>Images already taken after the detection</th><th>Satellite</th><th>Time</th><th>After detection</th></tr></thead>
              <tbody>{fu.images_after.map((im) => <tr key={im.name + im.time}><td className="small">{im.name}</td><td>{im.satellite}</td><td>{fmtTime(im.time)}</td><td>+{im.hours_after} h</td></tr>)}</tbody>
            </table>
          )}
          <table className="table">
            <thead><tr><th>Predicted next pass</th><th>Satellite</th><th>After detection</th><th>Image available?</th></tr></thead>
            <tbody>{fu.predicted.length ? fu.predicted.map((p) => (
              <tr key={p.satellite + p.time}><td>{fmtTime(p.time)}</td><td>{p.satellite}</td><td>+{p.hours_after} h</td><td>{p.in_catalogue ? 'Yes' : 'Not yet'}</td></tr>
            )) : <tr><td colSpan={4} className="muted">No earlier image of this spot found, so the next pass cannot be predicted.</td></tr>}</tbody>
          </table>
          <p className="small muted">{fu.note}</p>
        </>
      )}

      <h3 style={{ marginTop: '1.4rem' }}>6. Warning for fishermen and coastal communities</h3>
      <div style={{ display: 'flex', gap: '.4rem', margin: '.3rem 0 .5rem', flexWrap: 'wrap' }}>
        {LANGS.map(([k, l]) => <button key={k} className={lang === k ? 'btn primary' : 'btn'} onClick={() => setLang(k)}>{l}</button>)}
        <button className="btn" onClick={copy}>{copied ? 'Copied ✓' : 'Copy'}</button>
        <button className="btn" onClick={saveText}>Download .txt</button>
      </div>
      <pre style={{ whiteSpace: 'pre-wrap', background: '#FAFBFB', border: '1px solid #DDE4E7', borderRadius: 6, padding: '.7rem', fontFamily: 'inherit', margin: 0 }}>{x.warnings[lang]}</pre>
      <p className="small muted">{x.warnings.note}</p>

      <h3 style={{ marginTop: '1.4rem' }}>7. Clean-up progress</h3>
      {[['Oil recovered', 'oil_m3', 'oil', ' m³'], ['Coast cleaned', 'shore_km', 'shore', ' km']].map(([label, k, p, unit]) => (
        <div key={k} style={{ display: 'grid', gridTemplateColumns: '140px 1fr 230px', gap: '.6rem', alignItems: 'center', margin: '.3rem 0' }}>
          <span>{label}</span><Bar value={c.progress[p]} />
          <span className="small">{c.totals[k]}{unit} of {c.targets[k]}{unit}{c.progress[p] != null ? ` (${c.progress[p]}%)` : ''}</span>
        </div>
      ))}
      <div style={{ display: 'grid', gridTemplateColumns: '140px 1fr 230px', gap: '.6rem', alignItems: 'center', margin: '.3rem 0' }}>
        <span>Money spent</span><Bar value={c.progress.budget} color="#E8A42A" />
        <span className="small">{money(c.totals.cost)} of {money(c.targets.cost)}{c.progress.budget != null ? ` (${c.progress.budget}%)` : ''}</span>
      </div>
      {editable && (
        <form onSubmit={addEntry} style={{ display: 'flex', gap: '.4rem', flexWrap: 'wrap', alignItems: 'flex-end', margin: '.7rem 0' }}>
          <label className="small">Date<br /><input type="date" value={form.date} onChange={set('date')} /></label>
          <label className="small">Oil recovered (m³)<br /><input type="number" min="0" step="any" value={form.oil_m3} onChange={set('oil_m3')} style={{ width: 110 }} /></label>
          <label className="small">Coast cleaned (km)<br /><input type="number" min="0" step="any" value={form.shore_km} onChange={set('shore_km')} style={{ width: 110 }} /></label>
          <label className="small">Cost ({c.targets.currency})<br /><input type="number" min="0" step="any" value={form.cost} onChange={set('cost')} style={{ width: 120 }} /></label>
          <label className="small" style={{ flex: 1, minWidth: 160 }}>Note<br /><input value={form.note} onChange={set('note')} style={{ width: '100%' }} placeholder="e.g. 2 skimmers at the intercept point" /></label>
          <button className="btn primary" type="submit">Add day's work</button>
        </form>
      )}
      {c.entries.length > 0 && (
        <table className="table">
          <thead><tr><th>Date</th><th>Oil (m³)</th><th>Coast (km)</th><th>Cost</th><th>Note</th><th>By</th>{editable && <th />}</tr></thead>
          <tbody>{c.entries.map((e) => (
            <tr key={e.index}><td>{e.date}</td><td>{e.oil_m3}</td><td>{e.shore_km}</td><td>{money(e.cost)}</td><td>{e.note}</td><td className="small muted">{e.by}</td>
              {editable && <td><button className="link-btn" onClick={() => delEntry(e.index)}>Delete</button></td>}</tr>
          ))}</tbody>
        </table>
      )}
      <p className="small muted">{c.note}</p>

      <h3 style={{ marginTop: '1.4rem' }}>8. Case file for the authorities</h3>
      <p style={{ margin: '.2rem 0 .5rem' }}>A printable file with the incident, the suspected ship and its evidence, file fingerprints, samples to collect and the "polluter pays" damage estimate.</p>
      <button className="btn primary" onClick={openCaseFile}>Open case file (print / save as PDF)</button>
    </div>
  )
}
