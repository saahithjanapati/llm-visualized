import { defineConfig } from 'vite';
import { existsSync, readFileSync, readdirSync, rmSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import basicSsl from '@vitejs/plugin-basic-ssl';

const projectRoot = fileURLToPath(new URL('.', import.meta.url));
const UNUSED_PUBLIC_OUTPUT_PATHS = [
  '.DS_Store',
  'AGENT.md',
  'assets/.DS_Store',
  'assets/AGENT.md',
  'assets/geometries',
  'flops_per_step.csv',
  'flops_per_step.svg',
  'position_embeddings_grid.png',
  'position_embeddings_grid_1024x768.png',
  'position_embeddings_grid_1024x768_clamped_-0.5_0.5.png',
  'position_embeddings_grid_1024x768_clamped_-1_1.png',
  'twelve-layer-stack.css'
];
const constantsPath = resolve(projectRoot, 'src/utils/constants.js');

function shouldStripUnusedQkvAsset() {
  const constantsSource = readFileSync(constantsPath, 'utf8');
  const match = constantsSource.match(/export const USE_INSTANCED_MATRIX_SLICES = (true|false);/);
  return match?.[1] === 'true';
}

function stripUnusedBuildAssets() {
  const stripQkvAsset = shouldStripUnusedQkvAsset();
  const qkvAssetPattern = /^precomputed_components_qkv-.*\.glb$/;

  let outDir = resolve(projectRoot, 'dist');

  return {
    name: 'strip-unused-build-assets',
    apply: 'build',
    configResolved(config) {
      outDir = resolve(config.root, config.build.outDir);
    },
    closeBundle() {
      for (const assetPath of UNUSED_PUBLIC_OUTPUT_PATHS) {
        const absolutePath = resolve(outDir, assetPath);
        if (existsSync(absolutePath)) {
          rmSync(absolutePath, { force: true, recursive: true });
        }
      }
      if (!stripQkvAsset) return;
      const assetsDir = resolve(outDir, 'assets');
      if (!existsSync(assetsDir)) return;
      for (const fileName of readdirSync(assetsDir)) {
        if (!qkvAssetPattern.test(fileName)) continue;
        rmSync(resolve(assetsDir, fileName), { force: true });
      }
    }
  };
}

function redirectDirectoryIndexPages() {
  function buildMiddleware(root) {
    return function directoryIndexRedirect(req, res, next) {
      const requestUrl = String(req.url || '');
      if (!requestUrl.length) {
        next();
        return;
      }
      const method = String(req.method || 'GET').toUpperCase();
      if (method !== 'GET' && method !== 'HEAD') {
        next();
        return;
      }

      const url = new URL(requestUrl, 'https://llm-visualized.local');
      const pathname = url.pathname;
      if (!pathname || pathname === '/' || pathname.endsWith('/')) {
        next();
        return;
      }
      if (pathname.includes('.')) {
        next();
        return;
      }

      const indexPath = resolve(root, `.${pathname}`, 'index.html');
      if (!existsSync(indexPath)) {
        next();
        return;
      }

      res.statusCode = 302;
      res.setHeader('Location', `${pathname}/${url.search}`);
      res.end();
    };
  }

  return {
    name: 'redirect-directory-index-pages',
    configureServer(server) {
      server.middlewares.use(buildMiddleware(server.config.root));
    },
    configurePreviewServer(server) {
      server.middlewares.use(buildMiddleware(server.config.root));
    }
  };
}

export default defineConfig({
  root: '.',
  publicDir: 'public',
  assetsInclude: ['**/*.exr', '**/*.glb'],
  plugins: [basicSsl(), stripUnusedBuildAssets(), redirectDirectoryIndexPages()],
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
        info: resolve(projectRoot, 'info/index.html'),
        essay: resolve(projectRoot, 'essay/index.html')
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
