import { useEffect, useState } from 'react'
import api, { errorText } from '../api/client'
import { ErrorNote, fmtTime } from '../components/Bits.jsx'
import { useAuth } from '../context/AuthContext.jsx'
import '../system.css'

export default function SystemHealth() {
  const { can } = useAuth()
  const [h, setH] = useState(null)
  const [error, setError] = useState('')
  const [test, setTest] = useState('')

  const load = () => api.get('/system/health').then((r) => setH(r.data)).catch((e) => setError(errorText(e)))
  useEffect(() => {
    load()
    const t = setInterval(load, 60000)
    return () => clearInterval(t)
  }, [])

  const sendTest = async () => {
    setTest('Sending…')
    try {
      const r = await api.post('/system/test-alert')
      const parts = Object.entries(r.data.sent).map(([k, v]) => `${k}: ${v ? 'sent ✓' : r.data.configured[k] ? 'failed ✗' : 'not set up'}`)
      setTest(parts.join(' · '))
    } catch (e) { setTest(errorText(e)) }
  }

  if (error) return <ErrorNote text={error} />
  if (!h) return <div className="center-note">Loading…</div>

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h1>System health</h1>
          <p className="muted">{h.summary}. Server running for {h.uptime_hours} h. Refreshes every minute.</p>
        </div>
        <span className={`health-pill ${h.status}`}>{h.status === 'ok' ? 'All systems ready' : 'Needs attention'}</span>
      </header>

      <section className="panel">
        <h2>Checks</h2>
        <ul className="plain health-checks">
          {h.checks.map((c) => (
            <li key={c.name}><span className={c.ok ? 'tick' : 'cross'}>{c.ok ? '✓' : '!'}</span> <strong>{c.name}</strong> <span className="muted small">{c.detail}</span></li>
          ))}
        </ul>
        {can('admin') && (
          <p className="small"><button className="btn" onClick={sendTest}>Send test alert</button> <span className="muted">{test}</span></p>
        )}
      </section>

      <div className="grid-2">
        <section className="panel">
          <h2>Activity</h2>
          <div className="health-counts">
            {Object.entries(h.counts).map(([k, v]) => <div key={k}><strong>{v}</strong><span>{k.replace(/_/g, ' ')}</span></div>)}
          </div>
          {h.last_analysis && (
            <p className="small">Last analysis: #{h.last_analysis.id}, {h.last_analysis.state} ({h.last_analysis.message}), {fmtTime(h.last_analysis.at)}</p>
          )}
          <p className="small muted">Disk: {h.disk.free_gb} GB free of {h.disk.total_gb} GB</p>
        </section>

        <section className="panel">
          <h2>AI models</h2>
          <p className="small"><strong>Detector:</strong> {h.models.detector}</p>
          <table className="table">
            <thead><tr><th>File</th><th>Size</th><th>Updated</th></tr></thead>
            <tbody>
              {h.models.files.map((f) => (
                <tr key={f.name}><td>{f.ok ? '✓' : '✗'} {f.name}</td><td>{f.ok ? `${f.size_mb} MB` : 'missing'}</td><td>{f.ok ? fmtTime(f.updated) : '–'}</td></tr>
              ))}
            </tbody>
          </table>
        </section>
      </div>

      <section className="panel">
        <h2>Data freshness</h2>
        <table className="table">
          <thead><tr><th>Dataset</th><th>Folder</th><th>Files</th><th>Size</th><th>Newest file</th></tr></thead>
          <tbody>
            {h.datasets.map((d) => (
              <tr key={d.name}>
                <td>{d.exists && d.files ? '✓' : '✗'} {d.name}</td><td className="small">{d.folder}</td>
                <td>{d.files}</td><td>{d.size_mb} MB</td><td>{d.latest ? fmtTime(d.latest) : '–'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  )
}
