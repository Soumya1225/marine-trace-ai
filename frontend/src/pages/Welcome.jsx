import { useEffect, useState } from 'react'
import { Link, Navigate } from 'react-router-dom'
import api from '../api/client'
import { useAuth } from '../context/AuthContext.jsx'
import '../welcome.css'

const STEPS = [
  ['Watch', 'Every 6 hours the system checks Copernicus for new Sentinel-1 radar images over Indian coastal watch zones, and follows live AIS ship positions.'],
  ['Detect', 'Three deep-learning models (U-Net, DeepLabV3+, SegFormer) mark oil slicks in the radar image; a second stage rejects look-alikes using wind, contrast, edges and shape.'],
  ['Trace back', 'Real wind (ERA5) and ocean currents (Copernicus Marine) drive a drift model backwards to where the oil was released.'],
  ['Find the ship', 'Every ship track is compared with the back-traced spill. An XGBoost ranker scores the suspects and SHAP explains why. Ships with AIS switched off are flagged.'],
  ['Forecast', '6 to 72 hour drift forecast with an uncertainty zone, landfall time and threatened beaches, mangroves and ports.'],
  ['Act', 'Priority, recommended actions, Telegram / e-mail alerts, case assignment with deadlines and a one-click evidence package.'],
]
const RESULTS = [
  ['0.87', 'oil detection F1 score', '839 test radar images (SOS dataset)'],
  ['72%', 'right ship ranked first', 'XGBoost ranker, top 3: 88%'],
  ['100%', 'radar images passed the input check', 'phone photos and other pictures rejected'],
  ['0.07 km', 'LSTM drift error at 24 h', 'vs 6.1 km for persistence'],
]

export default function Welcome() {
  const { user } = useAuth()
  const [s, setS] = useState(null)
  useEffect(() => { api.get('/system/public-stats').then((r) => setS(r.data)).catch(() => {}) }, [])
  if (user) return <Navigate to="/" replace />

  return (
    <div className="welcome">
      <header className="w-top">
        <div className="w-brand"><span className="brand-mark" aria-hidden="true" /> Marine Trace AI <span className="muted">· SpillWatch</span></div>
        <Link className="btn primary" to="/login">Sign in</Link>
      </header>

      <section className="w-hero">
        <div>
          <p className="w-kicker">Smart India Hackathon · SIH26143</p>
          <h1>Find the oil.<br />Find the ship.</h1>
          <p className="w-lead">Marine Trace AI detects oil spills in satellite radar images, traces them back with wind and currents,
            and ranks the ships most likely responsible, with an explanation an investigator can check.</p>
          <div className="w-cta">
            <Link className="btn primary" to="/login">Open the dashboard</Link>
            <a className="btn" href="#how">How it works</a>
          </div>
        </div>
        <div className="w-live">
          <div className="w-live-head">{s?.live ? <span className="dot on" /> : <span className="dot" />} Live system</div>
          {[['Watch zones', s?.zones], ['Satellite images found', s?.images_found], ['Ships tracked live', s?.live_ships],
            ['Spills analysed', s?.incidents], ['Oil area detected', s ? `${s.area_km2} km²` : null]].map(([k, v]) => (
            <div key={k} className="w-stat"><strong>{v ?? '–'}</strong><span>{k}</span></div>
          ))}
        </div>
      </section>

      <section id="how" className="w-section">
        <h2>How it works</h2>
        <ol className="w-steps">
          {STEPS.map(([t, d], i) => <li key={t}><span className="w-no">{i + 1}</span><div><h3>{t}</h3><p>{d}</p></div></li>)}
        </ol>
      </section>

      <section className="w-section">
        <h2>Tested results</h2>
        <div className="w-results">
          {RESULTS.map(([v, k, n]) => <div key={k}><strong>{v}</strong><span>{k}</span><small>{n}</small></div>)}
        </div>
      </section>

      <section className="w-section w-case">
        <h2>Real case: Huntington Beach, California, 3 October 2021</h2>
        <p>On two real Sentinel-1 images with real NOAA AIS data, the system found the 4.4 km² main slick, marked it Critical,
          warned that Newport Beach was already within reach, excluded the rescue and response vessels, and flagged the calm
          afternoon image as unreliable. The real source was a damaged pipeline, and the system correctly spread suspicion
          across many vessels instead of blaming one ship.</p>
      </section>

      <section className="w-section">
        <h2>Built with</h2>
        <p className="w-tags">{['Sentinel-1 SAR', 'AIS', 'ERA5', 'Copernicus Marine', 'PyTorch', 'U-Net', 'DeepLabV3+', 'SegFormer', 'XGBoost', 'SHAP', 'LSTM',
          'FastAPI', 'React', 'Leaflet', 'Docker'].map((t) => <span key={t}>{t}</span>)}</p>
      </section>

      <footer className="w-foot">
        <span>MCA final-year project · Centurion University of Technology and Management</span>
        <span>Vessel scores show where to investigate; they are not proof of liability.</span>
      </footer>
    </div>
  )
}
