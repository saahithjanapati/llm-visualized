import { defineConfig } from 'vite';
import basicSsl from '@vitejs/plugin-basic-ssl';

export default defineConfig({
  root: '.',
  publicDir: 'public',
  assetsInclude: ['**/*.exr'],
  plugins: [basicSsl()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    target: 'esnext'
  },
  server: {
    https: true,
    open: '/index.html'
  }
});
