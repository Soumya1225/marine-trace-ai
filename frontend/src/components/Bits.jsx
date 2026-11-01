export function PriorityBadge({ value }) {
  return <span className={`badge pri-${(value || 'low').toLowerCase()}`}>{value}</span>
}

const STATUS_LABEL = {
  detected: 'Detected', under_review: 'Under review', confirmed: 'Confirmed', closed: 'Closed', rejected: 'Rejected',
}
export function StatusBadge({ value }) {
  return <span className={`badge st-${value}`}>{STATUS_LABEL[value] || value}</span>
}

export function ScoreBar({ value, level }) {
  const pct = Math.round((value || 0) * 100)
  return (
    <div className="score">
      <div className="score-track"><div className={`score-fill lv-${level}`} style={{ width: `${pct}%` }} /></div>
      <span className="score-num">{(value || 0).toFixed(2)}</span>
    </div>
  )
}

export function Empty({ title, children }) {
  return (
    <div className="empty">
      <h3>{title}</h3>
      <div className="muted">{children}</div>
    </div>
  )
}

export function ErrorNote({ text }) {
  return text ? <div className="error-note" role="alert">{text}</div> : null
}

export const fmtTime = (iso) => (iso ? new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' }) : '–')
export const fmtNum = (n, d = 2) => (n === null || n === undefined ? '–' : Number(n).toFixed(d))
