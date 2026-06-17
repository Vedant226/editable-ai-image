import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Proxy /api/* to the inpaint backend so the frontend can call /api/inpaint
// and /api/health with no CORS setup during development.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
