import * as THREE from 'three';
import { initVectorMatrixScene } from './src/animations/VectorMatrixScene.js';

// Initialize the Vector Matrix Scene
const canvas = document.createElement('canvas');
document.body.appendChild(canvas);
const cleanup = initVectorMatrixScene(canvas);

// Handle cleanup if the page is unloaded
window.addEventListener('beforeunload', () => {
    if (cleanup) {
        cleanup();
    }
});
