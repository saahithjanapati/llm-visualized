import * as THREE from 'three';
import { MHSAAnimation } from '../src/animations/MHSAAnimation.js';
import { loadPrecomputedGeometries } from '../src/utils/precomputedGeometryLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { FontLoader } from 'three/examples/jsm/loaders/FontLoader.js';
import { TextGeometry } from 'three/examples/jsm/geometries/TextGeometry.js';
import { EXRLoader } from 'three/examples/jsm/loaders/EXRLoader.js';
import { LayerPipeline } from '../src/engine/LayerPipeline.js';
import {
    CAPTION_TEXT_Y_POS,
    setPlaybackSpeed,
    EMBEDDING_MATRIX_PARAMS_VOCAB,
    EMBEDDING_MATRIX_PARAMS_POSITION,
    LN_PARAMS,
    LAYER_NORM_1_Y_POS,
    MLP_MATRIX_PARAMS_DOWN,
    INACTIVE_COMPONENT_COLOR,
    EMBEDDING_BOTTOM_TOP_ALIGN_OFFSET_FROM_LN1_BOTTOM,
    EMBEDDING_BOTTOM_Y_ADJUST,
    EMBEDDING_BOTTOM_VOCAB_X_OFFSET,
    EMBEDDING_BOTTOM_PAIR_GAP_X,
    EMBEDDING_BOTTOM_POS_X_OFFSET,
    TOP_EMBED_VOCAB_X_OFFSET,
    TOP_EMBED_Y_GAP_ABOVE_TOWER,
    TOP_EMBED_Y_ADJUST,
    TOP_LN_TO_TOP_EMBED_GAP
} from '../src/utils/constants.js';
import { WeightMatrixVisualization } from '../src/components/WeightMatrixVisualization.js';
import { LayerNormalizationVisualization } from '../src/components/LayerNormalizationVisualization.js';
import { MHA_FINAL_Q_COLOR, MHA_FINAL_K_COLOR } from '../src/animations/LayerAnimationConstants.js';

// Optionally load pre-baked geometries; returns instantly if disabled
await loadPrecomputedGeometries('../precomputed_components.glb');

// Skip intro typing screen for direct animation entry
const SKIP_INTRO = true;

// Set default playback speed to fast on load
try { setPlaybackSpeed('fast'); } catch (_) { /* no-op */ }
// ------------------------------------------------------------------
// GPT-2 tower – initialise immediately (but canvas hidden until intro
// completes).  This allows heavy geometries/textures to load while the
// user watches the intro text.
// ------------------------------------------------------------------

// Enable self-attention phase so MHSA runs to completion at each layer
MHSAAnimation.ENABLE_SELF_ATTENTION = true;

const gptCanvas = document.getElementById('gptCanvas');

const NUM_LAYERS = 12;
const camPos    = new THREE.Vector3(0, 11000, 16000);
const camTarget = new THREE.Vector3(0, 9000, 0);

const pipeline = new LayerPipeline(gptCanvas, NUM_LAYERS, {
    cameraPosition: camPos,
    cameraTarget: camTarget
});

// Top embedding activation state
let vocabTopRef = null; // reference to top vocab embedding
let __topEmbedActivated = false; // color switched to blue

// Show the GPT canvas right away so the tower is visible even while the
// HDRI environment map streams in.  The intro text still overlays it,
// so visual continuity is preserved.
gptCanvas.style.display = 'block';
 // Disable shadow rendering for the GPT pipeline scene (mobile perf)
try {
    const eng = pipeline.engine;
    eng.renderer.shadowMap.enabled = false;
    // Ensure objects and lights do not attempt to use shadows
    eng.scene.traverse((obj) => {
        if (obj.isMesh) {
            obj.castShadow = false;
            obj.receiveShadow = false;
        }
        if (obj.isLight) {
            obj.castShadow = false;
        }
    });
} catch (_) { /* no-op */ }
 // ------------------------------------------------------------------
// Embedding matrices (static visuals)
//   - Bottom: vocab (aligned with residual at x=0), positional to its right
//   - Top: flipped vocab (narrow end down toward layers), aligned with residual at x=0
// ------------------------------------------------------------------
try {
    const headBlue = new THREE.Color(MHA_FINAL_Q_COLOR);
    const headGreen = new THREE.Color(MHA_FINAL_K_COLOR);
     // Bottom pair aligned so their TOP surfaces sit on the residual stream height near LN1 branching
    const residualYBase = LAYER_NORM_1_Y_POS - LN_PARAMS.height / 2 + EMBEDDING_BOTTOM_TOP_ALIGN_OFFSET_FROM_LN1_BOTTOM;
     // --- Bottom Vocab Embedding (x = 0, aligned with residual stream) ---
    const bottomVocabCenterY = residualYBase - EMBEDDING_MATRIX_PARAMS_VOCAB.height / 2 + EMBEDDING_BOTTOM_Y_ADJUST;
    const vocabBottom = new WeightMatrixVisualization(
        null,
        new THREE.Vector3(0 + EMBEDDING_BOTTOM_VOCAB_X_OFFSET, bottomVocabCenterY, 0),
        EMBEDDING_MATRIX_PARAMS_VOCAB.width,
        EMBEDDING_MATRIX_PARAMS_VOCAB.height,
        EMBEDDING_MATRIX_PARAMS_VOCAB.depth,
        EMBEDDING_MATRIX_PARAMS_VOCAB.topWidthFactor,
        EMBEDDING_MATRIX_PARAMS_VOCAB.cornerRadius,
        EMBEDDING_MATRIX_PARAMS_VOCAB.numberOfSlits,
        EMBEDDING_MATRIX_PARAMS_VOCAB.slitWidth,
        EMBEDDING_MATRIX_PARAMS_VOCAB.slitDepthFactor,
        EMBEDDING_MATRIX_PARAMS_VOCAB.slitBottomWidthFactor,
        EMBEDDING_MATRIX_PARAMS_VOCAB.slitTopWidthFactor
    );
    vocabBottom.group.userData.label = 'Vocab Embedding';
    vocabBottom.setColor(headBlue);
    vocabBottom.setMaterialProperties({ opacity: 1.0, transparent: false, emissiveIntensity: 0.05 });
    pipeline.engine.scene.add(vocabBottom.group);
     // --- Bottom Positional Embedding (to the right of vocab) ---
    const gapX = EMBEDDING_BOTTOM_PAIR_GAP_X;
    const posX = (EMBEDDING_MATRIX_PARAMS_VOCAB.width / 2) + (EMBEDDING_MATRIX_PARAMS_POSITION.width / 2) + gapX + EMBEDDING_BOTTOM_POS_X_OFFSET + EMBEDDING_BOTTOM_VOCAB_X_OFFSET;
    // Align bases: compute positional centerY so its bottom equals vocab bottom
    const vocabBottomY = bottomVocabCenterY - EMBEDDING_MATRIX_PARAMS_VOCAB.height / 2;
    const bottomPosCenterY = vocabBottomY + EMBEDDING_MATRIX_PARAMS_POSITION.height / 2;
    const posBottom = new WeightMatrixVisualization(
        null,
        new THREE.Vector3(posX, bottomPosCenterY, 0),
        EMBEDDING_MATRIX_PARAMS_POSITION.width,
        EMBEDDING_MATRIX_PARAMS_POSITION.height,
        EMBEDDING_MATRIX_PARAMS_POSITION.depth,
        EMBEDDING_MATRIX_PARAMS_POSITION.topWidthFactor,
        EMBEDDING_MATRIX_PARAMS_POSITION.cornerRadius,
        EMBEDDING_MATRIX_PARAMS_POSITION.numberOfSlits,
        EMBEDDING_MATRIX_PARAMS_POSITION.slitWidth,
        EMBEDDING_MATRIX_PARAMS_POSITION.slitDepthFactor,
        EMBEDDING_MATRIX_PARAMS_POSITION.slitBottomWidthFactor,
        EMBEDDING_MATRIX_PARAMS_POSITION.slitTopWidthFactor
    );
    posBottom.group.userData.label = 'Positional Embedding';
    posBottom.setColor(headGreen);
    posBottom.setMaterialProperties({ opacity: 1.0, transparent: false, emissiveIntensity: 0.05 });
    pipeline.engine.scene.add(posBottom.group);
     // --- Top LayerNorm + Vocab Embedding (aligned with residual stream) ---
    const lastLayer = pipeline._layers[NUM_LAYERS - 1];
    if (lastLayer && lastLayer.mlpDown && lastLayer.mlpDown.group) {
        const tmp = new THREE.Vector3();
        lastLayer.mlpDown.group.getWorldPosition(tmp);
        const towerTopY = tmp.y + MLP_MATRIX_PARAMS_DOWN.height / 2;
        const topGap = TOP_EMBED_Y_GAP_ABOVE_TOWER;
         // Place a LayerNorm ring just above the tower top
        const topLnCenterY = towerTopY + topGap + LN_PARAMS.height / 2;
        const lnTop = new LayerNormalizationVisualization(
            new THREE.Vector3(0 + TOP_EMBED_VOCAB_X_OFFSET, topLnCenterY, 0),
            LN_PARAMS.width,
            LN_PARAMS.height,
            LN_PARAMS.depth,
            LN_PARAMS.wallThickness,
            LN_PARAMS.numberOfHoles,
            LN_PARAMS.holeWidth,
            LN_PARAMS.holeWidthFactor
        );
        lnTop.group.userData.label = 'LayerNorm (Top)';
        lnTop.setColor(new THREE.Color(INACTIVE_COMPONENT_COLOR));
        lnTop.setMaterialProperties({ opacity: 1.0, transparent: false, emissiveIntensity: 0.05 });
        pipeline.engine.scene.add(lnTop.group);
         // Position the final vocab embedding above the top LayerNorm
        const topVocabCenterY = topLnCenterY + (LN_PARAMS.height / 2) + TOP_LN_TO_TOP_EMBED_GAP + (EMBEDDING_MATRIX_PARAMS_VOCAB.height / 2) + TOP_EMBED_Y_ADJUST;
         const vocabTop = new WeightMatrixVisualization(
            null,
            new THREE.Vector3(0 + TOP_EMBED_VOCAB_X_OFFSET, topVocabCenterY, 0),
            EMBEDDING_MATRIX_PARAMS_VOCAB.width,
            EMBEDDING_MATRIX_PARAMS_VOCAB.height,
            EMBEDDING_MATRIX_PARAMS_VOCAB.depth,
            EMBEDDING_MATRIX_PARAMS_VOCAB.topWidthFactor,
            EMBEDDING_MATRIX_PARAMS_VOCAB.cornerRadius,
            EMBEDDING_MATRIX_PARAMS_VOCAB.numberOfSlits,
            EMBEDDING_MATRIX_PARAMS_VOCAB.slitWidth,
            EMBEDDING_MATRIX_PARAMS_VOCAB.slitDepthFactor,
            EMBEDDING_MATRIX_PARAMS_VOCAB.slitBottomWidthFactor,
            EMBEDDING_MATRIX_PARAMS_VOCAB.slitTopWidthFactor
        );
        // Flip vertically so the narrow end faces the layers below
        vocabTop.group.rotation.z = Math.PI;
        vocabTop.group.userData.label = 'Vocab Embedding (Top)';
        // Start black until vectors reach the entrance
        vocabTop.setColor(new THREE.Color(0x000000));
        vocabTop.setMaterialProperties({ opacity: 1.0, transparent: false, emissiveIntensity: 0.0 });
        vocabTopRef = vocabTop;
        pipeline.engine.scene.add(vocabTop.group);
    }
} catch (_) { /* optional – embedding visuals are non-critical */ }
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
 // ------------------------------------------------------------------
// Status overlay updater
// ------------------------------------------------------------------
const statusDiv = document.getElementById('statusOverlay');
const equationsPanel = document.getElementById('equationsPanel');
const equationsTitle = document.getElementById('equationsTitle');
const equationsBody = document.getElementById('equationsBody');
 // Persisted user preference for showing equations
const EQ_STORAGE_KEY = 'showEquations';
function getShowEquationsPref() {
    try {
        const v = localStorage.getItem(EQ_STORAGE_KEY);
        return v === null ? true : v === '1';
    } catch (_) { return true; }
}
function setShowEquationsPref(val) {
    try { localStorage.setItem(EQ_STORAGE_KEY, val ? '1' : '0'); } catch (_) {}
}
 let showEquations = getShowEquationsPref();
if (equationsPanel) equationsPanel.style.display = showEquations ? 'block' : 'none';
 // Equations mapping (LaTeX)
const EQ = {
    ln1: String.raw`x_{\text{ln}} = \mathrm{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2+\epsilon}} + \beta` ,
    qkv_per_head: String.raw`Q_k = x_{\text{ln}} W_k^Q,\quad K_k = x_{\text{ln}} W_k^K,\quad V_k = x_{\text{ln}} W_k^V\quad (k=1..h)`,
    qkv_packed: String.raw`Q = x_{\text{ln}} W^Q,\quad K = x_{\text{ln}} W^K,\quad V = x_{\text{ln}} W^V,\quad Q,K,V\in\mathbb{R}^{B\times T\times h\times d_h}`,
    attn: String.raw`A_k = \frac{Q_k K_k^\top}{\sqrt{d_h}} + M_{\text{causal}},\quad \alpha_k = \mathrm{softmax}(A_k)\ (\text{row-wise}),\quad H_k = \alpha_k V_k`,
    concat_proj: String.raw`H = \mathrm{Concat}(H_1,\ldots,H_h)\in\mathbb{R}^{B\times T\times d},\quad \mathrm{SA}(x) = H W^O + b_o`,
    resid1: String.raw`u = x + \mathrm{SA}(x_{\text{ln}})`,
    ln2: String.raw`u_{\text{ln}} = \mathrm{LN}(u) = \gamma' \odot \frac{u - \mu'}{\sqrt{{\sigma'}^{2}+\epsilon}} + \beta'`,
    mlp: String.raw`z = u_{\text{ln}} W_{\text{up}} + b_{\text{up}},\quad z' = \mathrm{GELU}(z),\quad \mathrm{MLP}(u_{\text{ln}}) = z' W_{\text{down}} + b_{\text{down}}`,
    resid2: String.raw`x_{\text{out}} = u + \mathrm{MLP}(u_{\text{ln}})`
};
 function renderEq(tex, title) {
    if (!equationsPanel || !equationsBody) return;
    if (!showEquations) return;
    equationsTitle.textContent = title || 'Equations';
    try {
        if (window.katex && window.katex.render) {
            equationsBody.innerHTML = '';
            window.katex.render(tex, equationsBody, { throwOnError: false, displayMode: true });
        } else {
            equationsBody.textContent = tex;
        }
    } catch (_) {
        equationsBody.textContent = tex;
    }
}
 // Cache last rendered key to avoid re-render churn
let __lastEqKey = '';
function updateEquations(layer) {
    if (!showEquations) return;
    if (!layer) return;
    const lanes = Array.isArray(layer.lanes) ? layer.lanes : [];
    if (!lanes.length) return;
     // Stage inference mirrors the status overlay but adds sub-phases
    const mlpActive = lanes.some(l => l.mlpUpStarted || l.ln2Phase === 'mlpReady' || l.ln2Phase === 'done');
    const ln2Active = !mlpActive && lanes.some(l => l.ln2Phase && l.ln2Phase !== 'notStarted');
    const ln1Active = !mlpActive && !ln2Active && lanes.some(l => ['waiting','right','insideLN'].includes(l.horizPhase));
    const mhsaActive = !mlpActive && !ln2Active && !ln1Active && (
        (layer && layer._mhsaStart === true) ||
        lanes.some(l => ['riseAboveLN','readyMHSA','travelMHSA','postMHSAAddition','waitingForLN2'].includes(l.horizPhase))
    );
     let key = '';
    let title = '';
     if (mlpActive) {
        // During/after MLP; if addition back to residual is underway, show residual2
        const adding = lanes.some(l => l.ln2Phase === 'done' && !l.additionComplete);
        key = adding ? 'resid2' : 'mlp';
        title = adding ? 'Residual Add 2' : 'MLP (FFN)';
    } else if (ln2Active) {
        key = 'ln2';
        title = 'LayerNorm 2';
    } else if (mhsaActive) {
        const m = layer.mhsaAnimation;
        const phase = m && m.mhaPassThroughPhase;
        const outPhase = m && m.outputProjMatrixAnimationPhase; // 'waiting' | 'vectors_entering' | 'vectors_inside' | 'completed'
         // Keep showing attention while the conveyor is active or until
        // the combined vectors actually start moving into W^O.
        if (phase === 'parallel_pass_through_active') {
            key = 'attn';
            title = 'Scaled Dot-Product Attention';
        } else if (phase === 'mha_pass_through_complete') {
            const postAdd = lanes.some(l => ['postMHSAAddition','waitingForLN2'].includes(l.horizPhase));
            if (postAdd || outPhase === 'vectors_inside' || outPhase === 'completed') {
                // Show residual add once vectors are passing through W^O or after
                key = 'resid1';
                title = 'Residual Add 1';
            } else if (m?.rowMergePhase === 'merging' || outPhase === 'vectors_entering') {
                // As soon as vectors begin traveling back toward W^O
                key = 'concat_proj';
                title = 'Concat Heads + W^O';
            } else {
                // Between conveyor end and travel start (fade/prepare)
                key = 'attn';
                title = 'Scaled Dot-Product Attention';
            }
        } else {
            // Early MHSA stages before conveyor fully starts
            key = 'qkv_per_head';
            title = 'Q/K/V Projections';
        }
    } else if (ln1Active || layer.isActive) {
        key = 'ln1';
        title = 'LayerNorm 1';
    }
     if (!key) return;
    if (key === __lastEqKey) return;
    __lastEqKey = key;
     // LN1 note (don’t assign back to x) — add as small subtitle in title
    const t = (key === 'ln1') ? 'LayerNorm 1 (no in-place assign)' : title;
    renderEq(EQ[key], t);
}
// Switch top vocab color to blue once residual vector reaches the entrance
function checkTopEmbeddingActivation() {
    try {
        if (__topEmbedActivated) return;
        const lastLayer = pipeline._layers[NUM_LAYERS - 1];
        if (!lastLayer || !vocabTopRef) return;
        const stopY = lastLayer.__topEmbedStopYLocal;
        if (typeof stopY !== 'number') return;
        const lanes = Array.isArray(lastLayer.lanes) ? lastLayer.lanes : [];
        for (const lane of lanes) {
            const v = lane && lane.originalVec;
            if (v && v.group && v.group.position.y >= stopY - 0.01) {
                const headBlue = new THREE.Color(MHA_FINAL_Q_COLOR);
                vocabTopRef.setColor(headBlue);
                vocabTopRef.setMaterialProperties({ emissiveIntensity: 0.05 });
                __topEmbedActivated = true;
                break;
            }
        }
    } catch (_) { /* ignore */ }
}
 function updateStatus() {
    if (!statusDiv) return;
    const idx = pipeline._currentLayerIdx;
    const total = NUM_LAYERS;
    const layer = pipeline._layers[idx];
    let displayStage = 'Loading...';
     if (layer && Array.isArray(layer.lanes) && layer.lanes.length) {
        const lanes = layer.lanes;
         const mlpActive = lanes.some(l => l.mlpUpStarted || l.ln2Phase === 'mlpReady' || l.ln2Phase === 'done');
        const ln2Active = !mlpActive && lanes.some(l => l.ln2Phase && l.ln2Phase !== 'notStarted');
        const ln1Active = !mlpActive && !ln2Active && lanes.some(l => ['waiting', 'right', 'insideLN'].includes(l.horizPhase));
        const mhsaActive = !mlpActive && !ln2Active && !ln1Active && (
            (layer && layer._mhsaStart === true) ||
            lanes.some(l => ['riseAboveLN', 'readyMHSA', 'travelMHSA', 'postMHSAAddition', 'waitingForLN2'].includes(l.horizPhase))
        );
         if (mlpActive) {
            displayStage = 'Multi-Layer Perceptron Block';
        } else if (ln2Active) {
            displayStage = 'LayerNorm 2';
        } else if (ln1Active) {
            displayStage = 'LayerNorm 1';
        } else if (mhsaActive) {
            displayStage = 'Multi Head Attention';
        } else if (layer.isActive) {
            displayStage = 'LayerNorm 1';
        }
    }
     statusDiv.textContent = `Layer ${idx + 1} / ${total}\n${displayStage}`;
    try { updateEquations(layer); } catch (_) {}
    // Also check if we should activate the top embedding color
    try { checkTopEmbeddingActivation(); } catch (_) {}
}
setInterval(updateStatus, 250);
 // ─────────────────────────────────────────────────────────────────--
// Settings modal behavior (UI only; not wired to animation)
// ─────────────────────────────────────────────────────────────────--
const settingsBtn = document.getElementById('settingsBtn');
const settingsOverlay = document.getElementById('settingsOverlay');
const settingsClose = document.getElementById('settingsClose');
 // Map UI selection to preset application
function applySpeed(value) {
    try { setPlaybackSpeed(value); } catch (_) { /* no-op */ }
}
 function openSettings() {
    settingsOverlay.style.display = 'flex';
    settingsOverlay.setAttribute('aria-hidden', 'false');
    // Lock scroll (defensive; page is already overflow hidden)
    document.body.style.overflow = 'hidden';
    // Pause the main visualisation engine and intro loop
    try { pipeline?.engine?.pause?.(); } catch (_) { /* no-op */ }
    __modalPaused = true;
    try {
        const checked = settingsOverlay.querySelector('input[name="playbackSpeed"]:checked');
        if (checked) {
            const selectedLabel = checked.closest('.speed-option');
            updateSpeedChecked(selectedLabel?.dataset.value || 'medium');
        }
        // Reflect current raycasting state in the checkbox
        const rc = document.getElementById('toggleRaycast');
        if (rc && pipeline?.engine?.isRaycastingEnabled) {
            rc.checked = !!pipeline.engine.isRaycastingEnabled();
        }
        const eq = document.getElementById('toggleEquations');
        if (eq) eq.checked = !!showEquations;
    } catch (_) { /* no-op */ }
}
 function closeSettings() {
    settingsOverlay.style.display = 'none';
    settingsOverlay.setAttribute('aria-hidden', 'true');
    document.body.style.overflow = '';
    // Resume the main visualisation engine and intro loop
    try { pipeline?.engine?.resume?.(); } catch (_) { /* no-op */ }
    __modalPaused = false;
}
 function updateSpeedChecked(value) {
    const labels = settingsOverlay.querySelectorAll('.speed-option');
    labels.forEach((label) => {
        const v = label.getAttribute('data-value');
        label.setAttribute('data-checked', String(v === value));
        const input = label.querySelector('input');
        if (input) input.checked = v === value;
    });
}
 settingsBtn?.addEventListener('click', openSettings);
settingsClose?.addEventListener('click', closeSettings);
settingsOverlay?.addEventListener('click', (e) => {
    if (e.target === settingsOverlay) closeSettings();
});
window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && settingsOverlay?.getAttribute('aria-hidden') === 'false') {
        closeSettings();
    }
});
 // Toggle checked style on click
settingsOverlay?.querySelectorAll('.speed-option').forEach((label) => {
    label.addEventListener('click', (e) => {
        e.preventDefault();
        const value = label.getAttribute('data-value');
        if (!value) return;
        updateSpeedChecked(value);
        applySpeed(value);
    });
});
 // Raycasting toggle wiring
const rayToggle = document.getElementById('toggleRaycast');
rayToggle?.addEventListener('change', () => {
    try { pipeline?.engine?.setRaycastingEnabled?.(!!rayToggle.checked); } catch (_) { /* no-op */ }
});
 // Equations toggle wiring
const eqToggle = document.getElementById('toggleEquations');
eqToggle?.addEventListener('change', () => {
    showEquations = !!eqToggle.checked;
    setShowEquationsPref(showEquations);
    if (equationsPanel) equationsPanel.style.display = showEquations ? 'block' : 'none';
    // Force re-render next tick
    __lastEqKey = '';
});

// Physical material toggle wiring
const physToggle = document.getElementById('togglePhysical');
physToggle?.addEventListener('change', () => {
    applyPhysicalMaterial(!!physToggle.checked);
});

function applyPhysicalMaterial(enabled) {
    if (!pipeline?.engine?.scene) return;
    const fromCtor = enabled ? THREE.MeshStandardMaterial : THREE.MeshPhysicalMaterial;
    const toCtor = enabled ? THREE.MeshPhysicalMaterial : THREE.MeshStandardMaterial;
    pipeline.engine.scene.traverse((obj) => {
        if (!obj.isMesh) return;
        const swapMat = (mat) => {
            if (!(mat instanceof fromCtor)) return mat;
            const params = {
                color: mat.color?.clone(),
                metalness: mat.metalness ?? 0,
                roughness: mat.roughness ?? 1,
                transparent: mat.transparent,
                opacity: mat.opacity,
                emissive: mat.emissive?.clone?.(),
                emissiveIntensity: mat.emissiveIntensity ?? 0,
                flatShading: mat.flatShading,
                side: mat.side,
            };
            const newMat = new toCtor(params);
            mat.dispose();
            return newMat;
        };
        if (Array.isArray(obj.material)) {
            obj.material = obj.material.map(swapMat);
        } else if (obj.material) {
            obj.material = swapMat(obj.material);
        }
    });
}
