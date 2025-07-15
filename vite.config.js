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
            app: resolve(__dirname, 'app.html'),
            layer: resolve(__dirname, 'tests/layer-animation-test.html'),
            gptModel: resolve(__dirname, 'tests/gpt-model-test.html'),
            vectorInstancedPrism: resolve(__dirname, 'tests/test_vector_instanced_prism.html'),
            prismRowVisualization: resolve(__dirname, 'tests/test_prism_row_visualization.html'),
            prismLayerNormAnimation: resolve(__dirname, 'tests/test_vector_instanced_prism_norm_anim.html'),
            prismAdditionAnimation: resolve(__dirname, 'tests/test_vector_instanced_prism_add_anim.html'),
            prismMultiplicationAnimation: resolve(__dirname, 'tests/test_prism_multiplication_animation.html'),
            layerNorm: resolve(__dirname, 'tests/test_layer_norm.html'),
            mhaAnimation: resolve(__dirname, 'tests/mha_visualization_test.html'),
            mhaPassThrough: resolve(__dirname, 'tests/test_mhsa_pass_through.html'),
            twelveLayerStack: resolve(__dirname, 'tests/twelve-layer-stack.html'),
            canMachinesThink: resolve(__dirname, 'tests/can-machines-think-test.html'),
            staticColor: resolve(__dirname, 'tests/twelve-layer-static.html'),

        }
    } 
  },
  server: {
    https: true,
    // Ensure open points to the root index.html
    open: '/index.html' // or just '/' which defaults to index.html
  }
}); 