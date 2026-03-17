import { defineConfig } from 'vite';
import { existsSync, rmSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import basicSsl from '@vitejs/plugin-basic-ssl';

const projectRoot = fileURLToPath(new URL('.', import.meta.url));
const UNUSED_PUBLIC_ASSETS = [
  'position_embeddings_grid.png',
  'position_embeddings_grid_1024x768.png',
  'position_embeddings_grid_1024x768_clamped_-0.5_0.5.png',
  'position_embeddings_grid_1024x768_clamped_-1_1.png'
];

function stripUnusedPublicAssets() {
  let outDir = resolve(projectRoot, 'dist');

  return {
    name: 'strip-unused-public-assets',
    apply: 'build',
    configResolved(config) {
      outDir = resolve(config.root, config.build.outDir);
    },
    closeBundle() {
      for (const assetPath of UNUSED_PUBLIC_ASSETS) {
        const absolutePath = resolve(outDir, assetPath);
        if (existsSync(absolutePath)) {
          rmSync(absolutePath, { force: true });
        }
      }
    }
  };
}

export default defineConfig({
  root: '.',
  publicDir: 'public',
  assetsInclude: ['**/*.exr', '**/*.glb'],
  plugins: [basicSsl(), stripUnusedPublicAssets()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    target: 'esnext',
    // `three` currently bundles into a large but expected vendor chunk.
    // Keep warnings focused on unexpected regressions in app-owned chunks.
    chunkSizeWarningLimit: 700,
    rollupOptions: {
      input: {
        main: resolve(projectRoot, 'index.html'),
        info: resolve(projectRoot, 'info/index.html')
      },
      output: {
        manualChunks(id) {
          if (!id) return undefined;
          if (id.includes('/node_modules/three/examples/jsm/')) return 'vendor-three-examples';
          if (id.includes('/node_modules/three-mesh-bvh/')) return 'vendor-three-bvh';
          if (id.includes('/node_modules/three/src/renderers/')) return 'vendor-three-renderers';
          if (id.includes('/node_modules/three/')) return 'vendor-three-core';
          if (id.includes('/node_modules/')) return 'vendor';
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
