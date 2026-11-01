import { useEffect, useState } from 'react'
import api, { errorText } from '../api/client'
import { fmtNum } from './Bits.jsx'
import { useAuth } from '../context/AuthContext.jsx'

const pct = (v) => (v == null ? '–' : `${(v * 100).toFixed(0)}%`)

export default function FeedbackPanel() {
  const [f, setF] = useState(null)
  const [msg, setMsg] = useState('')
  const [busy, setBusy] = useState(false)
  const { can } = useAuth()
  const [job, setJob] = useState(null)
  const load = () => api.get('/feedback').then((r) => setF(r.data)).catch((e) => setMsg(errorText(e)))
  const loadJob = () => api.get('/feedback/retrain').then((r) => setJob(r.data)).catch(() => {})
  useEffect(() => { load(); loadJob() }, [])
  useEffect(() => {
    if (!job?.running) return undefined
    const t = setInterval(async () => {
      const r = await api.get('/feedback/retrain').catch(() => null)
      if (!r) return
      setJob(r.data)
      if (!r.data.running) load()
    }, 3000)
    return () => clearInterval(t)
  }, [job?.running])
  const retrainNow = async () => {
    if (!window.confirm('Export the analyst decisions and retrain the models from them now? This runs in the background and can take a few minutes.')) return
    setMsg('')
    try { const r = await api.post('/feedback/retrain'); setJob(r.data) } catch (e) { setMsg(errorText(e)) }
  }
  const exportNow = async () => {
    setBusy(true)
    try {
      const r = await api.post('/feedback/export')
      setMsg(`Exported ${r.data.detection_pairs} radar crops, ${r.data.lookalike_rows} look-alike rows and ${r.data.attribution_rows} suspect rows to Datasets/feedback. Next: ${r.data.next_step}`)
      load()
    } catch (e) { setMsg(errorText(e)) } finally { setBusy(false) }
  }
  if (!f) return null
  const rt = f.retrain
  return (
    <section className="panel">
      <h2>Learning from analyst feedback</h2>
      <p className="muted small">Every incident an analyst confirms or rejects, and every suspect ship confirmed or dismissed, becomes new ground truth. It is exported as training data so the models improve with use.</p>
      <table className="table">
        <tbody>
          <tr><td>Incidents reviewed</td><td>{f.incidents_confirmed} confirmed as oil, {f.incidents_rejected} rejected as false alarms (of {f.incidents_total})</td></tr>
          <tr><td>Look-alike check agrees with analysts</td><td>{pct(f.lookalike_check_agreement)} {f.lookalike_checked ? `(${f.lookalike_checked} cases)` : ''}</td></tr>
          <tr><td>Suspects reviewed</td><td>{f.suspects_confirmed} confirmed, {f.suspects_dismissed} dismissed</td></tr>
          <tr><td>Confirmed ship was ranked #1</td><td>{pct(f.confirmed_ranked_first)}</td></tr>
          <tr><td>Last export</td><td>{f.last_export ? new Date(f.last_export).toLocaleString() : 'never'}</td></tr>
          {rt?.lookalike?.learned_accuracy_loo != null && (
            <tr><td>Look-alike model learned from feedback</td><td>{pct(rt.lookalike.learned_accuracy_loo)} leave-one-out accuracy on {rt.lookalike.reviewed} cases (rule check {pct(rt.lookalike.rule_check_accuracy)})</td></tr>
          )}
          {rt && <tr><td>Training data ready</td><td>{rt.detection_pairs} radar crops with masks, {fmtNum(rt.attribution?.reviewed ?? 0, 0)} reviewed suspects</td></tr>}
        </tbody>
      </table>
      <div className="feedback-actions">
        {can('admin', 'analyst') && <button className="btn" onClick={exportNow} disabled={busy || job?.running}>{busy ? 'Exporting…' : 'Export feedback as training data'}</button>}
        {can('admin') && (
          <button className="btn primary" onClick={retrainNow} disabled={job?.running || job?.script_found === false}
            title={job?.script_found === false ? 'scripts/retrain_from_feedback.py not found' : ''}>
            {job?.running ? 'Retraining…' : 'Retrain models from feedback'}
          </button>
        )}
      </div>
      {job && (job.running || job.finished_at) && (
        <div className={`retrain-status ${job.running ? 'running' : job.ok ? 'ok' : 'failed'}`}>
          <p><b>{job.running ? '⏳ Retraining in progress' : job.ok ? '✅ Last retraining finished' : '⚠️ Last retraining failed'}</b>
            {' '}<span className="muted small">
              started {job.started_at ? new Date(job.started_at).toLocaleString() : '–'}
              {job.finished_at ? `, finished ${new Date(job.finished_at).toLocaleString()}` : ''}
            </span></p>
          {job.exported && <p className="muted small">Training data used: {job.exported.detection_pairs} radar crops, {job.exported.lookalike_rows} look-alike decisions, {job.exported.attribution_rows} suspect decisions.</p>}
          {job.message && !job.running && <p className="small">{job.message}</p>}
          {job.log?.length > 0 && (
            <details open={job.running || job.ok === false}>
              <summary className="small">Output of the retraining script</summary>
              <pre className="retrain-log">{job.log.join('\n')}</pre>
            </details>
          )}
        </div>
      )}
      {msg && <p className="muted small">{msg}</p>}
    </section>
  )
}
