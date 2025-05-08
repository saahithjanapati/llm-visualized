import { defineConfig } from 'vite';
import { resolve } from 'path';
// Remove fs import if no longer needed
// import fs from 'fs'; 
import basicSsl from '@vitejs/plugin-basic-ssl'; // Import the SSL plugin

// Remove the entire generateIndexPlugin function
// function generateIndexPlugin() { ... }

export default defineConfig({
  // root can remain '.'
  root: '.', 
  // publicDir can remain 'public'
  publicDir: 'public', 
  plugins: [
    // Remove generateIndexPlugin()
    basicSsl() // Add the SSL plugin
  ],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    // Remove rollupOptions.input, Vite handles HTML entry points automatically
    rollupOptions: {
        input: {
            main: resolve(__dirname, 'index.html'),

            layer: resolve(__dirname, 'tests/layer-animation-test.html'),
            mha: resolve(__dirname, 'tests/multi-head-attention-test.html'),
            gptModel: resolve(__dirname, 'tests/gpt-model-test.html'),
            vectorInstancedPrism: resolve(__dirname, 'tests/test_vector_instanced_prism.html'),
            prismRowVisualization: resolve(__dirname, 'tests/test_prism_row_visualization.html'),
            prismLayerNormAnimation: resolve(__dirname, 'tests/test_vector_instanced_prism_norm_anim.html'),
            prismAdditionAnimation: resolve(__dirname, 'tests/test_vector_instanced_prism_add_anim.html')
        }
    } 
  },
  server: {
    https: true,
    // Ensure open points to the root index.html
    open: '/index.html' // or just '/' which defaults to index.html
  }
}); 