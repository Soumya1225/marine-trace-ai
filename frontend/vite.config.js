import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// API calls go to the FastAPI backend on port 8000
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,           // answer on IPv4 and IPv6 (makes the public link reliable)
    // allow the public demo link made by public_link.bat (Cloudflare quick tunnel)
    allowedHosts: ['localhost', '127.0.0.1', '.trycloudflare.com'],
    proxy: {
      '/api': 'http://127.0.0.1:8000',
      '/files': 'http://127.0.0.1:8000',
      '/figures': 'http://127.0.0.1:8000',
    },
  },
})
