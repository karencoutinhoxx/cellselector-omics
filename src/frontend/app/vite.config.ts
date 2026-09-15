import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/health':     'http://localhost:8001',
      '/genes':      'http://localhost:8001',
      '/recommend':  'http://localhost:8001',
      '/cell-lines': 'http://localhost:8001',
      '/stats':      'http://localhost:8001',
      // Scoped to the actual API sub-paths, not a bare '/graph' prefix —
      // the frontend also owns the page route "/graph" (GraphExplorer),
      // and a blanket '/graph' proxy would swallow that SPA route too.
      '/graph/explore':                 'http://localhost:8001',
      '/graph/pathway-neighbors':       'http://localhost:8001',
      '/graph/cell-lines-via-pathway':  'http://localhost:8001',
    },
  },
})
