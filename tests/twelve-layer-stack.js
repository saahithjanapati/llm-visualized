import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { FontLoader } from 'three/examples/jsm/loaders/FontLoader.js';
import { TextGeometry } from 'three/examples/jsm/geometries/TextGeometry.js';
import { EXRLoader } from 'three/examples/jsm/loaders/EXRLoader.js';
import { CAPTION_TEXT_Y_POS } from '../src/utils/constants.js';
import { initPipeline } from './initPipeline.js';
import { setupEmbeddingVisuals } from './embeddingVisuals.js';
import { setupEquationsOverlay } from './equationsOverlay.js';
import { setupSettingsPanel } from './settingsPanel.js';

// Skip intro typing screen for direct animation entry
const SKIP_INTRO = true;

const { pipeline, NUM_LAYERS, gptCanvas } = await initPipeline();
const { checkTopEmbeddingActivation } = setupEmbeddingVisuals(pipeline, NUM_LAYERS);
const eqControls = setupEquationsOverlay(pipeline, NUM_LAYERS, checkTopEmbeddingActivation);
let __modalPaused = false;
setupSettingsPanel(pipeline, eqControls, (v) => { __modalPaused = v; });
 // ------------------------------------------------------------------
// Intro – "Can machines think?" Ghost-type animation (almost verbatim
// from can-machines-think-test.html but adapted for this page & API).
// ------------------------------------------------------------------
const introCanvas = document.getElementById('introCanvas');
 const renderer = new THREE.WebGLRenderer({ canvas: introCanvas, antialias: true });
// Disable shadows for the intro scene (mobile perf)
renderer.shadowMap.enabled = false;
try {
    // Match engine DPR strategy for consistent crispness
    const cap = (window && window.__RENDER_DPR_CAP) ? window.__RENDER_DPR_CAP : 2.0;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, cap));
} catch (_) {
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2.0));
}
renderer.setSize(window.innerWidth, window.innerHeight);
 const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);
 const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 1, 20000);
camera.position.set(0, 0, 1500);
 const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
 // Gate and lifecycle for intro RAF/rendering so we can stop it after hand-off
let __introActive = true;
let __introRaf = 0;
let __introCleaned = false;
function cleanupIntro() {
    if (__introCleaned) return;
    __introCleaned = true;
    __introActive = false;
    try { cancelAnimationFrame(__introRaf); } catch (_) {}
    try { controls && controls.dispose && controls.dispose(); } catch (_) {}
    try { renderer && renderer.dispose && renderer.dispose(); } catch (_) { /* no-op */ }
    try {
        scene.traverse((child) => {
            if (child.geometry) child.geometry.dispose();
            if (child.material) {
                if (Array.isArray(child.material)) child.material.forEach((m) => m && m.dispose && m.dispose());
                else if (child.material.dispose) child.material.dispose();
            }
        });
    } catch (_) { /* no-op */ }
}
 // Ambient light replaced by HDRI environment; ambient removed.
const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(5, 10, 7.5);
// Disable shadows for intro directional light
dirLight.castShadow = false;
scene.add(dirLight);
 const TEXT = 'Can machines think?';
const TYPE_DELAY = 120;
 let cursorMesh;
 // Load font and setup typing
const loader = new FontLoader();
loader.load('https://threejs.org/examples/fonts/helvetiker_regular.typeface.json', (font) => {
    const charGroup = new THREE.Group();
    scene.add(charGroup);
     const charMeshes = [];
    const charWidths = [];
    let xOffset = 0;
    const charSpacing = 10;
    const extraCursorGap = 60;
     for (const ch of TEXT) {
        if (ch === ' ') {
            xOffset += 120;
            charWidths.push(120);
            charMeshes.push(null);
            continue;
        }
         const geo = new TextGeometry(ch, { font, size: 200, height: 40, curveSegments: 4,
            bevelEnabled: true, bevelThickness: 2, bevelSize: 2, bevelOffset: 0, bevelSegments: 1 });
        geo.computeBoundingBox();
        const width = geo.boundingBox.max.x - geo.boundingBox.min.x;
        const mat = new THREE.MeshNormalMaterial({ transparent: true, opacity: 0 });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.x = xOffset;
        // Do not use shadows for intro text
        mesh.castShadow = false;
        mesh.receiveShadow = false;
        charGroup.add(mesh);
        charMeshes.push(mesh);
        charWidths.push(width);
        xOffset += width + charSpacing;
    }
     charGroup.position.x = -xOffset / 2 + charSpacing / 2;
    const bounds = new THREE.Box3().setFromObject(charGroup);
    const textHeight = bounds.max.y - bounds.min.y;
     const cursorGeo = new THREE.BoxGeometry(15, textHeight, 20);
    const cursorMat = new THREE.MeshBasicMaterial({ color: 0x888888 });
    cursorMesh = new THREE.Mesh(cursorGeo, cursorMat);
    cursorMesh.position.set(charGroup.position.x, bounds.min.y + textHeight / 2, 25);
    cursorMesh.castShadow = false;
    cursorMesh.receiveShadow = false;
    scene.add(cursorMesh);
     const revealChar = (index) => {
        if (index >= charMeshes.length) {
            blinkCursor();
            // Schedule switch-over once typing finishes & a short pause.
            setTimeout(transitionToGPT, 1500);
            return;
        }
         const mesh = charMeshes[index];
        const width = charWidths[index];
        const gap = (index === charMeshes.length - 1) ? extraCursorGap : charSpacing;
         new TWEEN.Tween(cursorMesh.position)
            .to({ x: cursorMesh.position.x + width + gap }, TYPE_DELAY * 0.9)
            .start();
         if (mesh) {
            new TWEEN.Tween(mesh.material).to({ opacity: 1 }, TYPE_DELAY).start();
        }
         setTimeout(() => revealChar(index + 1), TYPE_DELAY);
    };
     const blinkCursor = () => setInterval(() => { cursorMesh.visible = !cursorMesh.visible; }, 500);
     setTimeout(() => revealChar(0), 500);
});
 // ─────────────────────────────────────────────────────────────────--
// HDRI Environment Map Lighting
// ─────────────────────────────────────────────────────────────────--
const exrLoader = new EXRLoader().setDataType(THREE.HalfFloatType);
exrLoader.load('../rogland_clear_night_64.exr', (texture) => {
    texture.mapping = THREE.EquirectangularReflectionMapping;
     // Apply environment to intro and GPT scenes
    scene.environment = texture;
    pipeline.engine.scene.environment = texture;
     // Remove any ambient lights from both scenes
    scene.traverse((obj) => { if (obj.isAmbientLight) scene.remove(obj); });
    pipeline.engine.scene.traverse((obj) => { if (obj.isAmbientLight) pipeline.engine.scene.remove(obj); });
     // Optionally adjust renderer tone mapping for HDR appearance
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
    pipeline.engine.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    pipeline.engine.renderer.toneMappingExposure = 1.0;
     // HDRI now ready – remove intro canvas if it hasn't been hidden yet
    introCanvas.style.display = 'none';
    // Stop intro loop and dispose even if transition text hasn't fired yet
    try { cleanupIntro(); } catch (_) {}
     // Hide loading GIF
    const gif = document.getElementById('loadingGif');
    if (gif) gif.style.display = 'none';
}, undefined, (err) => {
    console.warn('HDRI failed to load:', err);
    // Fail-safe: still remove intro overlay so the app continues
    introCanvas.style.display = 'none';
    // Ensure intro loop is stopped on failure too
    try { cleanupIntro(); } catch (_) {}
    const gif = document.getElementById('loadingGif');
    if (gif) gif.style.display = 'none';
});
 // Resize handling for both canvases
window.addEventListener('resize', () => {
    const { innerWidth, innerHeight } = window;
    // intro
    if (typeof __introActive === 'undefined' || __introActive) {
        camera.aspect = innerWidth / innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(innerWidth, innerHeight);
    }
    // GPT canvas handled internally by CoreEngine resize listener.
});
 // Track modal pause to also pause intro loop rendering
let __modalPaused = false;
 // Animation loop for intro (gated so we can stop it after transition)
const loopIntro = () => {
    if (!__introActive) return;
    __introRaf = requestAnimationFrame(loopIntro);
    if (__modalPaused) return;
    controls.update();
    if (typeof TWEEN !== 'undefined') TWEEN.update();
    renderer.render(scene, camera);
};
if (!SKIP_INTRO) {
    loopIntro();
} else {
    try { transitionToGPT(); } catch (_) {
        try { const ic = document.getElementById('introCanvas'); if (ic) ic.style.display = 'none'; } catch (_) {}
    }
    try { const gif = document.getElementById('loadingGif'); if (gif) gif.style.display = 'none'; } catch (_) {}
}
 // ------------------------------------------------------------------
// Transition helper
// ------------------------------------------------------------------
function transitionToGPT() {
    introCanvas.style.display = 'none';
    gptCanvas.style.display   = 'block';
    // Stop and dispose the intro resources immediately upon hand-off
    cleanupIntro();
}
 // Cleanup on unload
const cleanup = () => {
    pipeline.dispose();
    renderer.dispose();
    scene.traverse((child) => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) {
            if (Array.isArray(child.material)) child.material.forEach((m) => m.dispose());
            else child.material.dispose();
        }
    });
};
 window.addEventListener('beforeunload', () => cleanup());
 // ------------------------------------------------------------------
// Typewriter caption underneath GPT tower (single scene)
// ------------------------------------------------------------------
const TYPE_TEXT  = 'Can machines think?';
const TYPE_DELAY_CAP = 120; // ms
 const captionLoader = new FontLoader();
captionLoader.load('https://threejs.org/examples/fonts/helvetiker_regular.typeface.json', (font) => {
    const charGroup = new THREE.Group();
    pipeline.engine.scene.add(charGroup);
     const charMeshes = [];
    const charWidths = [];
    let xOffset = 0;
    const charSpacing = 100; // Adjust spacing for big text size
     for (const ch of TYPE_TEXT) {
        if (ch === ' ') {
            xOffset += 400; // space width
            charMeshes.push(null);
            charWidths.push(400);
            continue;
        }
         const geo = new TextGeometry(ch, {
            font,
            size: 800,
            height: 80,
            curveSegments: 6,
            bevelEnabled: true,
            bevelThickness: 4,
            bevelSize: 3,
            bevelOffset: 0,
            bevelSegments: 1
        });
        geo.computeBoundingBox();
        const width = geo.boundingBox.max.x - geo.boundingBox.min.x;
        const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, transparent: true, opacity: 0 });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.x = xOffset;
        // Disable shadows for caption meshes
        mesh.castShadow = false;
        mesh.receiveShadow = false;
        charGroup.add(mesh);
        charMeshes.push(mesh);
        charWidths.push(width);
        xOffset += width + charSpacing;
    }
     // Center group beneath tower using configurable constant
    charGroup.position.set(-xOffset / 2 + charSpacing / 2, CAPTION_TEXT_Y_POS, 0);
     const revealChar = (index) => {
        if (index >= charMeshes.length) return;
         const mesh = charMeshes[index];
        if (mesh) {
            new TWEEN.Tween(mesh.material).to({ opacity: 1 }, TYPE_DELAY_CAP).start();
        }
         setTimeout(() => revealChar(index + 1), TYPE_DELAY_CAP);
    };
     // Kick off typing after short delay
    setTimeout(() => revealChar(0), 500);
     // Hide loading GIF now that caption is starting
    const gif = document.getElementById('loadingGif');
    if (gif) gif.style.display = 'none';
});
