import * as THREE from 'three';
import { initMainScene } from './src/animations/MainScene.js';

// Initialize the Main Scene
const canvas = document.createElement('canvas');
document.body.appendChild(canvas);
const cleanup = initMainScene(canvas);

// Handle cleanup if the page is unloaded
window.addEventListener('beforeunload', () => {
    if (cleanup) {
        cleanup();
    }
});
