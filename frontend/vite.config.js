import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Dev server on :5173 proxies /api → FastAPI backend on :8000,
// so no CORS or API-key setup is needed in development.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  preview: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
});
