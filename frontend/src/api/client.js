import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401 && !err.config.url.includes('/auth/login')) {
      localStorage.removeItem('token')
      window.location.href = '/login'
    }
    return Promise.reject(err)
  }
)

export const errorText = (err) =>
  err?.response?.data?.detail
    ? typeof err.response.data.detail === 'string'
      ? err.response.data.detail
      : JSON.stringify(err.response.data.detail)
    : err?.message || 'Something went wrong'

// download a protected file (PDF, CSV, GeoJSON) with the auth header
export async function download(url, filename) {
  const res = await api.get(url, { responseType: 'blob' })
  const link = document.createElement('a')
  link.href = URL.createObjectURL(res.data)
  link.download = filename
  link.click()
  URL.revokeObjectURL(link.href)
}

export default api
