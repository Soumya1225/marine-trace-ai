import { useEffect, useState } from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import api from '../api/client'
import { useAuth } from '../context/AuthContext.jsx'

// Menu follows the order of the work: watch the sea -> detect -> investigate -> check the models -> admin.
// `hint` shows as a tooltip when the mouse rests on a menu item.
const LINKS = [
  { section: '1. Dashboard' },
  { to: '/', label: 'Overview', end: true, hint: 'Summary map, hotspots and latest spills' },
  { to: '/alerts', label: 'Alerts', badge: true, hint: 'New spill warnings (Medium, High, Critical)' },
  { section: '2. Monitoring' },
  { to: '/zones', label: 'Watch zones', hint: 'Sea areas to monitor' },
  { to: '/feed', label: 'Satellite feed', hint: 'New Sentinel-1 radar images over the watch zones' },
  { to: '/live', label: 'Live ships', hint: 'Real-time ship positions (AIS)' },
  { section: '3. Detection' },
  { to: '/analyze', label: 'New analysis', roles: ['admin', 'analyst'], hint: 'Upload a satellite image and AIS data, run the AI' },
  { to: '/incidents', label: 'Incidents', hint: 'All detected oil spills, with suspect ships and forecast' },
  { section: '4. Investigation' },
  { to: '/cases', label: 'Case board', hint: 'Assign, review, confirm or reject each spill' },
  { to: '/vessels', label: 'Vessels', hint: 'Ships linked to spills and their history' },
  { to: '/compare', label: 'Compare images', hint: 'Earlier vs later image: did the oil move as forecast?' },
  { section: '5. Model evaluation' },
  { to: '/models', label: 'Model performance', hint: 'Accuracy, ROC-AUC and comparison of the AI models' },
  { section: '6. Administration', roles: ['admin', 'analyst'] },
  { to: '/users', label: 'Users', roles: ['admin'], hint: 'Add users, set roles and passwords' },
  { to: '/system', label: 'System health', roles: ['admin', 'analyst'], hint: 'Checks that models, data and settings are ready' },
  { to: '/activity', label: 'Activity log', roles: ['admin', 'analyst'], hint: 'Who did what, and when' },
]

export default function Layout({ children }) {
  const { user, logout, can } = useAuth()
  const [unread, setUnread] = useState(0)
  const navigate = useNavigate()

  useEffect(() => {
    const load = () => api.get('/alerts').then((r) => setUnread(r.data.filter((a) => !a.read).length)).catch(() => {})
    load()
    const t = setInterval(load, 30000)
    return () => clearInterval(t)
  }, [])

  return (
    <div className="shell">
      <aside className="rail">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <div>
            <div className="brand-name">SpillWatch</div>
            <div className="brand-sub">Detection & vessel attribution</div>
          </div>
        </div>
        <nav>
          {LINKS.filter((l) => !l.roles || can(...l.roles)).map((l, i) => l.section ? (
            <div key={`s${i}`} className="nav-section">{l.section}</div>
          ) : (
            <NavLink key={l.to} to={l.to} end={l.end} title={l.hint} className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}>
              {l.label}
              {l.badge && unread > 0 && <span className="nav-badge">{unread}</span>}
            </NavLink>
          ))}
        </nav>
        <div className="rail-user">
          <div>{user?.name}</div>
          <div className="muted small">{user?.role}</div>
          <button className="link-btn" onClick={() => { logout(); navigate('/login') }}>Sign out</button>
        </div>
      </aside>
      <main className="content">{children}</main>
    </div>
  )
}
