import * as THREE from 'three';
import { appState } from '../state/appState.js';
import { getPreference } from '../utils/preferences.js';
import { MHA_FINAL_Q_COLOR } from '../animations/LayerAnimationConstants.js';
import { USE_PHYSICAL_MATERIALS } from '../utils/constants.js';

// Initializes status overlay and equations panel updates.
export function initStatusOverlay(pipeline, NUM_LAYERS) {
    const statusDiv = document.getElementById('statusOverlay');
    const equationsPanel = document.getElementById('equationsPanel');
    const equationsTitle = document.getElementById('equationsTitle');
    const equationsBody = document.getElementById('equationsBody');

    appState.showEquations = getPreference('showEquations', true);
    appState.showHdrBackground = getPreference('showHdrBackground', false);
    appState.showOrbitingStars = getPreference('showOrbitingStars', true);
    if (equationsPanel) equationsPanel.style.display = appState.showEquations ? 'block' : 'none';
    appState.applyEnvironmentBackground(pipeline);
    pipeline?.setOrbitingStarsEnabled?.(appState.showOrbitingStars);

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
        if (!appState.showEquations) return;
        equationsTitle.textContent = title || 'Equations';
        if (window.katex?.render) {
            try {
                equationsBody.innerHTML = '';
                window.katex.render(tex, equationsBody, { throwOnError: false, displayMode: true });
            } catch (err) {
                console.error('KaTeX render failed:', err);
                equationsBody.textContent = tex;
            }
        } else {
            equationsBody.textContent = tex;
        }
    }

    appState.lastEqKey = '';

    function updateEquations(layer) {
        if (!appState.showEquations) return;
        if (!layer) return;
        const lanes = Array.isArray(layer.lanes) ? layer.lanes : [];
        if (!lanes.length) return;

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
            const adding = lanes.some(l => l.ln2Phase === 'done' && !l.additionComplete);
            key = adding ? 'resid2' : 'mlp';
            title = adding ? 'Residual Add 2' : 'MLP (FFN)';
        } else if (ln2Active) {
            key = 'ln2';
            title = 'LayerNorm 2';
        } else if (mhsaActive) {
            const m = layer.mhsaAnimation;
            const phase = m && m.mhaPassThroughPhase;
            const outPhase = m && m.outputProjMatrixAnimationPhase;
            if (phase === 'parallel_pass_through_active') {
                key = 'attn';
                title = 'Scaled Dot-Product Attention';
            } else if (phase === 'mha_pass_through_complete') {
                const postAdd = lanes.some(l => ['postMHSAAddition','waitingForLN2'].includes(l.horizPhase));
                if (postAdd || outPhase === 'vectors_inside' || outPhase === 'completed') {
                    key = 'resid1';
                    title = 'Residual Add 1';
                } else if (m?.rowMergePhase === 'merging' || outPhase === 'vectors_entering') {
                    key = 'concat_proj';
                    title = 'Concat Heads + W^O';
                } else {
                    key = 'attn';
                    title = 'Scaled Dot-Product Attention';
                }
            } else {
                key = 'qkv_per_head';
                title = 'Q/K/V Projections';
            }
        } else if (ln1Active || layer.isActive) {
            key = 'ln1';
            title = 'LayerNorm 1';
        }
        if (!key) return;
        if (key === appState.lastEqKey) return;
        appState.lastEqKey = key;
        const t = (key === 'ln1') ? 'LayerNorm 1 (no in-place assign)' : title;
        renderEq(EQ[key], t);
    }

    function checkTopEmbeddingActivation() {
        if (appState.topEmbedActivated) return;
        const lastLayer = pipeline._layers[NUM_LAYERS - 1];
        if (!lastLayer || !appState.vocabTopRef) return;
        const stopY = lastLayer.__topEmbedStopYLocal;
        if (typeof stopY !== 'number') return;
        const lanes = Array.isArray(lastLayer.lanes) ? lastLayer.lanes : [];
        for (const lane of lanes) {
            const v = lane && lane.originalVec;
            if (v && v.group && v.group.position.y >= stopY - 0.01) {
                const headBlue = new THREE.Color(MHA_FINAL_Q_COLOR);
                appState.vocabTopRef.setColor(headBlue);
                appState.vocabTopRef.setMaterialProperties({ emissiveIntensity: 0.05 });
                appState.topEmbedActivated = true;
                break;
            }
        }
    }

    let lastStatusText = '';

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

        updateEquations(layer);

        const nextStatusText = `Layer ${idx + 1} / ${total}\n${displayStage}`;
        if (nextStatusText === lastStatusText) {
            checkTopEmbeddingActivation();
            return;
        }

        lastStatusText = nextStatusText;
        statusDiv.textContent = nextStatusText;
        checkTopEmbeddingActivation();
    }

    const scheduleFrame = (typeof window !== 'undefined' && typeof window.requestAnimationFrame === 'function')
        ? window.requestAnimationFrame.bind(window)
        : (cb) => setTimeout(cb, 16);

    let framePending = false;
    let needsUpdate = false;

    const scheduleStatusUpdate = () => {
        if (framePending) return;
        framePending = true;
        scheduleFrame(() => {
            framePending = false;
            if (!needsUpdate) return;
            needsUpdate = false;
            updateStatus();
        });
    };

    const onProgress = () => {
        needsUpdate = true;
        scheduleStatusUpdate();
    };

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

    applyPhysicalMaterial(USE_PHYSICAL_MATERIALS);
    if (pipeline && typeof pipeline.addEventListener === 'function') {
        pipeline.addEventListener('progress', onProgress);
    }
    updateStatus();
}
