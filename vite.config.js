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
            testVector: resolve(__dirname, 'tests/test_vector.html'),
            testWeightMatrix: resolve(__dirname, 'tests/test_weight_matrix.html'),
            risingVectors: resolve(__dirname, 'tests/rising-vectors-test.html'),
            vectorMatrix: resolve(__dirname, 'tests/vector-matrix-test.html'),
            vectorAddition: resolve(__dirname, 'tests/vector_addition_test.html'),
            vectorMultiplication: resolve(__dirname, 'tests/vector_multiplication_test.html'),
            vectorNormalization: resolve(__dirname, 'tests/vector_normalization_test.html'),
            layerNorm: resolve(__dirname, 'tests/test_layer_norm.html'),
            layerNormPipeline: resolve(__dirname, 'tests/layer-norm-pipeline-test.html'),
            layer: resolve(__dirname, 'tests/layer-animation-test.html'),
            attention: resolve(__dirname, 'tests/attention-head-test.html'),
            mha: resolve(__dirname, 'tests/multi-head-attention-test.html'),
            vectorAdditionInstanced: resolve(__dirname, 'tests/vector_addition_instanced_test.html'),
            gptModel: resolve(__dirname, 'tests/gpt-model-test.html'),
            vectorInstanced: resolve(__dirname, 'tests/test_vector_instanced.html'),
            vectorInstancedPrism: resolve(__dirname, 'tests/test_vector_instanced_prism.html'),
            minimalInstancedColor: resolve(__dirname, 'tests/minimal_instanced_color_test.html'),
            prismRowVisualization: resolve(__dirname, 'tests/test_prism_row_visualization.html')
        }
    } 
  },
  server: {
    https: true,
    // Ensure open points to the root index.html
    open: '/index.html' // or just '/' which defaults to index.html
  }
}); 