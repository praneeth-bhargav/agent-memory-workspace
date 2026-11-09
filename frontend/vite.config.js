import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

export default defineConfig({
  server: { proxy: { '/api': process.env.VITE_API_PROXY || 'http://127.0.0.1:8000' } },
  build: { rollupOptions: { input: {
    'memory-lab': fileURLToPath(new URL('./memory-lab.html', import.meta.url)),
  } } },
});
