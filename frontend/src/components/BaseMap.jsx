import { useEffect } from 'react'
import { MapContainer, TileLayer, useMap } from 'react-leaflet'

// Keeps the map on the given area. MapContainer only reads `bounds` when it is first created, so without this
// the map stays where it was (e.g. on Huntington Beach) when the page is reused for an incident somewhere else.
function FitBounds({ bounds }) {
  const map = useMap()
  const key = bounds ? JSON.stringify(bounds) : ''
  useEffect(() => {
    if (!bounds) return
    const ok = bounds.flat().every((v) => Number.isFinite(v))
    if (ok) map.fitBounds(bounds, { padding: [10, 10] })
  }, [key]) // eslint-disable-line react-hooks/exhaustive-deps
  return null
}

export default function BaseMap({ center = [33.6, -118.0], zoom = 10, bounds, children, height = 520 }) {
  return (
    <MapContainer center={center} zoom={zoom} bounds={bounds} style={{ height, width: '100%' }} scrollWheelZoom>
      <TileLayer
        attribution='&copy; OpenStreetMap contributors'
        url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
        maxZoom={19}
      />
      <FitBounds bounds={bounds} />
      {children}
    </MapContainer>
  )
}

export const LEVEL_COLOR = { high: '#C8322B', medium: '#E8A42A', low: '#3E8E6A' }
export const PRIORITY_COLOR = { Critical: '#8E1B14', High: '#C8322B', Medium: '#E8A42A', Low: '#3E8E6A' }
