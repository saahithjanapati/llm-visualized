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
        }
    } 
  },
  server: {
    https: true,
    // Ensure open points to the root index.html
    open: '/index.html' // or just '/' which defaults to index.html
  }
}); 