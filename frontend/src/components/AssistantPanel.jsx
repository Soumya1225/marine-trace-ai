import { useEffect, useRef, useState } from 'react'
import api, { errorText } from '../api/client'
import '../assistant.css'

const START = ['Which ship is most suspicious and why?', 'Where will the oil be in 24 hours?', 'Which beaches are threatened?',
  'Is this really oil or a look-alike?', 'What should we do now?', 'Were there any ships without AIS?']

export default function AssistantPanel({ incidentId }) {
  const [msgs, setMsgs] = useState([{ who: 'bot', text: 'Ask me anything about this incident: the suspects, why a ship is suspected, the forecast, threatened sites, reliability or what to do next.' }])
  const [q, setQ] = useState('')
  const [busy, setBusy] = useState(false)
  const end = useRef(null)
  useEffect(() => { end.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }) }, [msgs])

  const ask = async (question) => {
    const text = (question ?? q).trim()
    if (!text || busy) return
    setQ(''); setBusy(true)
    setMsgs((m) => [...m, { who: 'me', text }])
    try {
      const r = await api.post(`/incidents/${incidentId}/ask`, { question: text })
      setMsgs((m) => [...m, { who: 'bot', text: r.data.answer, sources: r.data.sources, engine: r.data.engine }])
    } catch (e) {
      setMsgs((m) => [...m, { who: 'bot', text: errorText(e) }])
    } finally { setBusy(false) }
  }

  return (
    <div className="panel assistant">
      <div className="chat">
        {msgs.map((m, i) => (
          <div key={i} className={`bubble ${m.who}`}>
            <div>{m.text}</div>
            {m.sources && <div className="src">From: {m.sources.join(', ')} · {m.engine}</div>}
          </div>
        ))}
        {busy && <div className="bubble bot muted">Thinking…</div>}
        <div ref={end} />
      </div>
      <div className="chips ask-chips">
        {START.map((s) => <button key={s} className="btn small" onClick={() => ask(s)} disabled={busy}>{s}</button>)}
      </div>
      <form className="ask-row" onSubmit={(e) => { e.preventDefault(); ask() }}>
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Type a question, e.g. Why is ALLURE suspected?" />
        <button className="btn primary" disabled={busy || !q.trim()}>Ask</button>
      </form>
      <p className="muted small">Answers come only from this incident&apos;s data, and each one says which part it used. Vessel scores show where to investigate; they are not proof.</p>
    </div>
  )
}
