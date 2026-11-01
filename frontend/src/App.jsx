import { Navigate, Route, Routes } from 'react-router-dom'
import Layout from './components/Layout.jsx'
import { useAuth } from './context/AuthContext.jsx'
import Activity from './pages/Activity.jsx'
import Alerts from './pages/Alerts.jsx'
import Dashboard from './pages/Dashboard.jsx'
import IncidentDetail from './pages/IncidentDetail.jsx'
import Incidents from './pages/Incidents.jsx'
import Login from './pages/Login.jsx'
import Welcome from './pages/Welcome.jsx'
import ModelPerformance from './pages/ModelPerformance.jsx'
import Compare from './pages/Compare.jsx'
import LiveShips from './pages/LiveShips.jsx'
import Feed from './pages/Feed.jsx'
import Zones from './pages/Zones.jsx'
import Cases from './pages/Cases.jsx'
import SystemHealth from './pages/SystemHealth.jsx'
import NewAnalysis from './pages/NewAnalysis.jsx'
import Users from './pages/Users.jsx'
import VesselProfile from './pages/VesselProfile.jsx'
import Vessels from './pages/Vessels.jsx'

function Protected({ children, roles }) {
  const { user, loading } = useAuth()
  if (loading) return <div className="center-note">Loading…</div>
  if (!user) return <Navigate to="/welcome" replace />
  if (roles && !roles.includes(user.role)) return <div className="center-note">Your role cannot open this page.</div>
  return <Layout>{children}</Layout>
}

export default function App() {
  return (
    <Routes>
      <Route path="/welcome" element={<Welcome />} />
      <Route path="/login" element={<Login />} />
      <Route path="/" element={<Protected><Dashboard /></Protected>} />
      <Route path="/analyze" element={<Protected roles={['admin', 'analyst']}><NewAnalysis /></Protected>} />
      <Route path="/incidents" element={<Protected><Incidents /></Protected>} />
      <Route path="/cases" element={<Protected><Cases /></Protected>} />
      <Route path="/incidents/:id" element={<Protected><IncidentDetail /></Protected>} />
      <Route path="/vessels" element={<Protected><Vessels /></Protected>} />
      <Route path="/vessels/:mmsi" element={<Protected><VesselProfile /></Protected>} />
      <Route path="/live" element={<Protected><LiveShips /></Protected>} />
      <Route path="/feed" element={<Protected><Feed /></Protected>} />
      <Route path="/zones" element={<Protected><Zones /></Protected>} />
      <Route path="/compare" element={<Protected><Compare /></Protected>} />
      <Route path="/models" element={<Protected><ModelPerformance /></Protected>} />
      <Route path="/alerts" element={<Protected><Alerts /></Protected>} />
      <Route path="/activity" element={<Protected roles={['admin', 'analyst']}><Activity /></Protected>} />
      <Route path="/system" element={<Protected roles={['admin', 'analyst']}><SystemHealth /></Protected>} />
      <Route path="/users" element={<Protected roles={['admin']}><Users /></Protected>} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
