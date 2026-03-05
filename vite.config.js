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
    target: 'esnext',
    // `three` currently bundles into a large but expected vendor chunk.
    // Keep warnings focused on unexpected regressions in app-owned chunks.
    chunkSizeWarningLimit: 700,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id) return undefined;
          if (id.includes('/node_modules/three/examples/jsm/')) return 'vendor-three-examples';
          if (id.includes('/node_modules/three-mesh-bvh/')) return 'vendor-three-bvh';
          if (id.includes('/node_modules/three/src/renderers/')) return 'vendor-three-renderers';
          if (id.includes('/node_modules/three/')) return 'vendor-three-core';
          if (id.includes('/node_modules/')) return 'vendor';
          if (id.includes('/src/ui/selectionPanel')) return 'ui-selection-panel';
          if (id.includes('/src/engine/layers/') || id.includes('/src/animations/')) return 'scene-runtime';
          return undefined;
        }
      }
    }
  },
  server: {
    https: true,
    open: '/index.html'
  }
});
