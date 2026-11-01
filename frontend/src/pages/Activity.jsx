import { useEffect, useState } from 'react'
import api, { errorText } from '../api/client'
import { ErrorNote, fmtTime } from '../components/Bits.jsx'

export default function Activity() {
  const [rows, setRows] = useState([])
  const [error, setError] = useState('')
  useEffect(() => { api.get('/activity').then((r) => setRows(r.data)).catch((e) => setError(errorText(e))) }, [])
  return (
    <div className="page">
      <header className="page-head"><h1>Activity log</h1></header>
      <ErrorNote text={error} />
      <table className="table panel">
        <thead><tr><th>When</th><th>Who</th><th>Action</th><th>Detail</th></tr></thead>
        <tbody>{rows.map((a) => <tr key={a.id}><td>{fmtTime(a.created_at)}</td><td>{a.user}</td><td>{a.action.replaceAll('_', ' ')}</td><td>{a.detail}</td></tr>)}</tbody>
      </table>
    </div>
  )
}
