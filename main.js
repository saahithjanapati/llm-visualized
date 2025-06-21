import { startEngine } from './src/engine/CoreEngine.js';
import Gpt2Layer from './src/engine/layers/Gpt2Layer.js';
import { createRandomSource } from './src/data/RandomActivationSource.js';

// Create / reuse canvas element
const canvas = document.createElement('canvas');
canvas.style.width = '100%';
canvas.style.height = '100%';
document.body.appendChild(canvas);

const random = createRandomSource();
const NUM_LAYERS = 1;
const layers = Array.from({ length: NUM_LAYERS }, (_, i) => new Gpt2Layer(i, random));

const cleanup = startEngine(canvas, layers);

// Handle cleanup if the page is unloaded
window.addEventListener('beforeunload', () => {
    cleanup && cleanup();
});
