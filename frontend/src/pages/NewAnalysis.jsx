import { useEffect, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import api, { errorText } from '../api/client'
import { useAuth } from '../context/AuthContext.jsx'
import { ErrorNote, fmtTime } from '../components/Bits.jsx'
import '../check.css'

const STEPS = ['Satellite image', 'AIS ship data', 'Settings']
const PICTURE = /\.(png|jpe?g)$/i
const FULL_SCENE = /\.(zip|tiff?)$/i
const VERDICT = {
  oil: ['pri-critical', 'Oil detected'], possible_lookalike: ['pri-medium', 'Possibly a look-alike'],
  no_oil: ['pri-low', 'No oil'], not_radar: ['pri-high', 'Not a radar image'],
}

export default function NewAnalysis() {
  const [scenes, setScenes] = useState([])
  const [ais, setAis] = useState([])
  const [sceneId, setSceneId] = useState(null)
  const [aisIds, setAisIds] = useState([])
  const [params] = useSearchParams()
  const [opts, setOpts] = useState({ acquisition_time: '', bbox: params.get('bbox') || '', wind_u: '', wind_v: '', current_u: '', current_v: '' })
  const [zones, setZones] = useState([])
  useEffect(() => { api.get('/zones').then((r) => setZones(r.data.filter((z) => z.active))).catch(() => {}) }, [])
  const [job, setJob] = useState(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState('')
  const [jobs, setJobs] = useState([])
  const { can } = useAuth()
  const [check, setCheck] = useState(null)
  const poll = useRef(null)

  const loadJobs = () => api.get('/analyses').then((r) => setJobs(r.data.slice(0, 8))).catch(() => {})
  const deleteJob = async (j) => {
    const n = j.incident_ids.length
    if (!window.confirm(`Delete analysis #${j.id}${n ? ` and its ${n} incident(s) (#${j.incident_ids.join(', #')})` : ''}? This cannot be undone.`)) return
    setError('')
    try { await api.delete(`/analyses/${j.id}`); loadJobs() } catch (e) { setError(errorText(e)) }
  }
  useEffect(() => {
    loadJobs()
    const t = setInterval(loadJobs, 5000)
    return () => clearInterval(t)
  }, [])

  const load = async () => {
    const [s, a] = await Promise.all([api.get('/scenes'), api.get('/ais')])
    setScenes(s.data); setAis(a.data)
    return { scenes: s.data, ais: a.data }
  }
  useEffect(() => { load().catch((e) => setError(errorText(e))); return () => clearInterval(poll.current) }, [])

  const scene = scenes.find((s) => s.id === sceneId)

  const upload = async (kind, file, extra = {}) => {
    if (!file) return
    setError('')
    if (kind === 'scene') {
      setCheck(null)
      if (PICTURE.test(file.name)) {
        // check the picture first: is it a radar image of the sea, and what does the model see?
        setBusy(`Checking ${file.name}…`)
        try {
          const f = new FormData()
          f.append('file', file)
          const c = await api.post('/quick-check', f)
          setCheck(c.data)
          if (c.data.verdict === 'not_radar') { setBusy(''); return }   // stop: never upload a non-radar picture
        } catch (e) { setError(errorText(e)); setBusy(''); return }
      } else {
        setCheck({ verdict: 'satellite_file', title: 'Satellite data file', message: 'Sentinel-1 / GeoTIFF files are radar data by definition; the detailed check runs during the analysis.' })
      }
    }
    setBusy(`Uploading ${file.name}…`)
    const form = new FormData()
    form.append('file', file)
    Object.entries(extra).forEach(([k, v]) => v && form.append(k, v))
    try {
      const r = await api.post(kind === 'scene' ? '/scenes' : '/ais', form)
      await load()
      kind === 'scene' ? setSceneId(r.data.id) : setAisIds((ids) => [...ids, r.data.id])
    } catch (e) { setError(errorText(e)) } finally { setBusy('') }
  }

  const register = async (kind, path) => {
    if (!path) return
    setBusy('Registering file…'); setError('')
    try {
      const r = await api.post(kind === 'scene' ? '/scenes/register' : '/ais/register', { path })
      await load()
      kind === 'scene' ? setSceneId(r.data.id) : setAisIds((ids) => [...ids, r.data.id])
    } catch (e) { setError(errorText(e)) } finally { setBusy('') }
  }

  const remove = async (kind, id, name) => {
    if (!window.confirm(`Remove ${name} from the list?`)) return
    setError('')
    try {
      await api.delete(kind === 'scene' ? `/scenes/${id}` : `/ais/${id}`)
      if (kind === 'scene' && sceneId === id) { setSceneId(null); setCheck(null) }
      if (kind === 'ais') setAisIds((ids) => ids.filter((x) => x !== id))
      await load()
    } catch (e) { setError(errorText(e)) }
  }

  const demo = async () => {
    setBusy('Creating demo data…'); setError('')
    try {
      const r = await api.post('/demo/setup')
      await load()
      setSceneId(r.data.scene.id); setAisIds([r.data.ais.id])
    } catch (e) { setError(errorText(e)) } finally { setBusy('') }
  }

  const start = async () => {
    setError('')
    if (scene && FULL_SCENE.test(scene.filename) && !opts.bbox.trim()) {
      setError('Enter the area to analyse (step 3). A full Sentinel-1 image covers about 250 km including land, which gives false spills inland and takes very long.')
      return
    }
    const num = (v) => (v === '' ? null : Number(v))
    const body = {
      scene_id: sceneId, ais_ids: aisIds,
      acquisition_time: opts.acquisition_time ? new Date(opts.acquisition_time + 'Z').toISOString() : null,
      bbox: opts.bbox ? opts.bbox.split(',').map(Number) : null,
      wind_u: num(opts.wind_u), wind_v: num(opts.wind_v), current_u: num(opts.current_u), current_v: num(opts.current_v),
    }
    try {
      const r = await api.post('/analyses', body)
      setJob(r.data)
      poll.current = setInterval(async () => {
        const j = await api.get(`/analyses/${r.data.id}`)
        setJob(j.data)
        if (['done', 'failed'].includes(j.data.state)) clearInterval(poll.current)
      }, 2000)
    } catch (e) { setError(errorText(e)) }
  }

  const set = (k) => (e) => setOpts({ ...opts, [k]: e.target.value })
  const toggleAis = (id) => setAisIds((ids) => (ids.includes(id) ? ids.filter((x) => x !== id) : [...ids, id]))

  return (
    <div className="page narrow">
      <header className="page-head">
        <div>
          <h1>New analysis</h1>
          <p className="muted">Detect spills in a satellite image, match them with ship tracks, forecast drift and assess impact.</p>
        </div>
        <button className="btn" onClick={demo} disabled={!!busy}>Create demo data</button>
      </header>
      <ErrorNote text={error} />
      {busy && <div className="info-note">{busy}</div>}

      <section className="panel step">
        <h2><span className="step-no">1</span>{STEPS[0]}</h2>
        {scenes.length > 0 && (
          <div className="pick-list">
            {scenes.map((s) => (
              <label key={s.id} className={sceneId === s.id ? 'pick on' : 'pick'}>
                <input type="radio" name="scene" checked={sceneId === s.id} onChange={() => setSceneId(s.id)} />
                <span>{s.filename}</span>
                <span className="muted small">{s.acquisition_time ? fmtTime(s.acquisition_time) : 'time unknown'}, {s.size_mb} MB</span>
                <button type="button" className="link-btn" onClick={(e) => { e.preventDefault(); remove('scene', s.id, s.filename) }}>Remove</button>
              </label>
            ))}
          </div>
        )}
        <div className="add-row">
          <label className="file-btn">Upload image (.zip, .tif, .png)
            <input type="file" accept=".zip,.tif,.tiff,.png,.jpg,.jpeg" onChange={(e) => upload('scene', e.target.files[0], { bbox: opts.bbox })} hidden />
          </label>
          <PathInput placeholder="or use a file on this computer, e.g. Datasets/raw/sentinel1/S1B_...zip" onSubmit={(p) => register('scene', p)} />
        </div>
        <ImageCheck check={check} />
      </section>

      <section className="panel step">
        <h2><span className="step-no">2</span>{STEPS[1]}</h2>
        <p className="muted small">Pick the AIS day files that cover the 12 hours before the image time. Large daily files are filtered to the image area automatically.</p>
        {ais.length > 0 && (
          <div className="pick-list">
            {ais.map((a) => (
              <label key={a.id} className={aisIds.includes(a.id) ? 'pick on' : 'pick'}>
                <input type="checkbox" checked={aisIds.includes(a.id)} onChange={() => toggleAis(a.id)} />
                <span>{a.filename}</span><span className="muted small">{a.size_mb} MB</span>
                <button type="button" className="link-btn" onClick={(e) => { e.preventDefault(); remove('ais', a.id, a.filename) }}>Remove</button>
              </label>
            ))}
          </div>
        )}
        <div className="add-row">
          <label className="file-btn">Upload AIS (.csv)
            <input type="file" accept=".csv" onChange={(e) => upload('ais', e.target.files[0])} hidden />
          </label>
          <PathInput placeholder="or a file on this computer, e.g. Datasets/raw/ais/AIS_2021_10_03.csv" onSubmit={(p) => register('ais', p)} />
        </div>
      </section>

      <section className="panel step">
        <h2><span className="step-no">3</span>{STEPS[2]}</h2>
        <div className="form-grid">
          <label>Image time (UTC){scene?.acquisition_time ? ' – detected from file' : ''}
            <input type="datetime-local" value={opts.acquisition_time} onChange={set('acquisition_time')} />
          </label>
          <label>Area to analyse (min lon, min lat, max lon, max lat)
            <input placeholder="-118.25,33.42,-117.85,33.72" value={opts.bbox} onChange={set('bbox')} />
            {zones.length > 0 && (
              <select value="" onChange={(e) => e.target.value && setOpts({ ...opts, bbox: e.target.value })}>
                <option value="">…or pick a watch zone</option>
                {zones.map((z) => <option key={z.id} value={z.bbox.join(',')}>{z.name}</option>)}
              </select>
            )}
          </label>
          <label>Wind east / north (m/s)
            <span className="pair"><input value={opts.wind_u} onChange={set('wind_u')} placeholder="auto" /><input value={opts.wind_v} onChange={set('wind_v')} placeholder="auto" /></span>
          </label>
          <label>Current east / north (m/s)
            <span className="pair"><input value={opts.current_u} onChange={set('current_u')} placeholder="auto" /><input value={opts.current_v} onChange={set('current_v')} placeholder="auto" /></span>
          </label>
        </div>
        <p className="muted small">Leave wind and current empty to use the downloaded ERA5 / Copernicus Marine files, or the default values.</p>
        <button className="btn primary" disabled={!sceneId || (job && job.state === 'running')} onClick={start}>Run analysis</button>
      </section>

      {job && (
        <section className="panel">
          <h2>Analysis #{job.id}</h2>
          <div className="progress"><div className={`progress-fill ${job.state}`} style={{ width: `${job.progress}%` }} /></div>
          <p>{job.message}</p>
          {job.state === 'failed' && <details><summary>Error details</summary><pre className="pre">{job.error}</pre></details>}
          {job.state === 'done' && (
            <div>
              <p className="muted">Detector: {job.summary?.detector}. Ships in AIS data: {job.summary?.ais_vessels}.</p>
              {job.incident_ids.length === 0 ? <p>No oil spill found in this image.</p> : (
                <ul className="plain">
                  {job.incident_ids.map((id) => <li key={id}><Link to={`/incidents/${id}`}>Open incident #{id}</Link></li>)}
                </ul>
              )}
            </div>
          )}
        </section>
      )}
      <RecentJobs jobs={jobs} onDelete={can('admin') ? deleteJob : null} />
    </div>
  )
}

function ImageCheck({ check }) {
  if (!check) return null
  if (check.verdict === 'satellite_file') return <p className="muted small">{check.title}: {check.message}</p>
  const [badge, label] = VERDICT[check.verdict] || ['', check.title]
  return (
    <div className={check.verdict === 'not_radar' ? 'check-box rejected' : 'check-box'}>
      {check.overlay && <img src={check.overlay} alt="model preview" />}
      <div>
        <p><span className={`badge ${badge}`}>{label}</span> <strong>{check.verdict === 'not_radar' ? 'Image rejected' : 'Radar image check passed'}</strong></p>
        <p className="small">{check.message}</p>
        {check.verdict === 'not_radar'
          ? <p className="muted small">The image was not uploaded. Choose a Sentinel-1 radar image (.zip, .tif, or a grey radar picture).</p>
          : <p className="muted small">Quick preview by {check.detector}. Fill in the area and time in step 3 and run the analysis for spill outlines, suspect ships, drift and impact.</p>}
      </div>
    </div>
  )
}

function RecentJobs({ jobs, onDelete }) {
  if (!jobs.length) return null
  return (
    <section className="panel">
      <h2>Recent analyses</h2>
      <p className="muted small">Analyses keep running on the server even if you leave this page.</p>
      <table className="table">
        <thead><tr><th title="Serial number in this list (always 1, 2, 3 …)">S.No.</th><th title="Permanent analysis ID">ID</th><th>Image</th><th>Status</th><th>Progress</th><th>Result</th>{onDelete && <th />}</tr></thead>
        <tbody>
          {jobs.map((j, idx) => (
            <tr key={j.id}>
              <td>{jobs.length - idx}</td>
              <td className="muted">#{j.id}</td>
              <td>{j.scene}</td>
              <td>{j.state}</td>
              <td><div className="progress"><div className={`progress-fill ${j.state}`} style={{ width: `${j.progress}%` }} /></div></td>
              <td>
                {j.state === 'done'
                  ? (j.incident_ids.length ? j.incident_ids.map((id) => <Link key={id} to={`/incidents/${id}`} style={{ marginRight: 8 }}>#{id}</Link>) : 'No spill found')
                  : j.message}
              </td>
              {onDelete && (
                <td>{j.state !== 'running' && j.state !== 'queued' && (
                  <button type="button" className="link-btn" onClick={() => onDelete(j)}>Delete</button>
                )}</td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  )
}

function PathInput({ placeholder, onSubmit }) {
  const [v, setV] = useState('')
  return (
    <form className="path-input" onSubmit={(e) => { e.preventDefault(); onSubmit(v.trim()); setV('') }}>
      <input value={v} onChange={(e) => setV(e.target.value)} placeholder={placeholder} />
      <button className="btn" disabled={!v.trim()}>Use file</button>
    </form>
  )
}
