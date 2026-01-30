import * as THREE from 'three';
import { appState } from '../state/appState.js';
import { getPreference } from '../utils/preferences.js';
import {
    MHA_FINAL_Q_COLOR,
    MHA_FINAL_K_COLOR,
    MHA_FINAL_V_COLOR,
    MHA_OUTPUT_PROJECTION_MATRIX_COLOR,
    MLP_UP_MATRIX_COLOR,
    MLP_DOWN_MATRIX_COLOR
} from '../animations/LayerAnimationConstants.js';
import { USE_PHYSICAL_MATERIALS } from '../utils/constants.js';

// Initializes status overlay and equations panel updates.
export function initStatusOverlay(pipeline, NUM_LAYERS) {
    const statusDiv = document.getElementById('statusOverlay');
    const equationsPanel = document.getElementById('equationsPanel');
    const equationsTitle = document.getElementById('equationsTitle');
    const equationsBody = document.getElementById('equationsBody');
    const shouldShowEquations = () => appState.showEquations && !appState.equationsSuppressed;

    appState.showEquations = getPreference('showEquations', true);
    appState.showHdrBackground = getPreference('showHdrBackground', false);
    if (equationsPanel) equationsPanel.style.display = shouldShowEquations() ? 'block' : 'none';
    appState.applyEnvironmentBackground(pipeline);

    const colorHex = (hex) => `#${Number(hex).toString(16).padStart(6, '0')}`;
    const colorize = (hex, body) => `\\textcolor{${hex}}{${body}}`;
    const qColor = colorHex(MHA_FINAL_Q_COLOR);
    const kColor = colorHex(MHA_FINAL_K_COLOR);
    const vColor = colorHex(MHA_FINAL_V_COLOR);
    const woColor = colorHex(MHA_OUTPUT_PROJECTION_MATRIX_COLOR);
    const mlpUpColor = colorHex(MLP_UP_MATRIX_COLOR);
    const mlpDownColor = colorHex(MLP_DOWN_MATRIX_COLOR);
    const Q = colorize(qColor, 'Q');
    const K = colorize(kColor, 'K');
    const V = colorize(vColor, 'V');
    const WQ = colorize(qColor, 'W^Q');
    const WK = colorize(kColor, 'W^K');
    const WV = colorize(vColor, 'W^V');
    const WO = colorize(woColor, 'W^O');
    const WUp = colorize(mlpUpColor, 'W_{\\text{up}}');
    const WDown = colorize(mlpDownColor, 'W_{\\text{down}}');
    const bQ = colorize(qColor, 'b^Q');
    const bK = colorize(kColor, 'b^K');
    const bV = colorize(vColor, 'b^V');
    const bO = colorize(woColor, 'b_o');
    const bUp = colorize(mlpUpColor, 'b_{\\text{up}}');
    const bDown = colorize(mlpDownColor, 'b_{\\text{down}}');

    const EQ = {
        ln1: String.raw`x_{\text{ln}} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta`,
        qkv_per_head: `{\\scriptsize ${Q} = x_{\\text{ln}} ${WQ} + ${bQ} \\, ${K} = x_{\\text{ln}} ${WK} + ${bK} \\, ${V} = x_{\\text{ln}} ${WV} + ${bV}}`,
        qkv_packed: `{\\scriptsize ${Q} = x_{\\text{ln}} ${WQ} + ${bQ} \\, ${K} = x_{\\text{ln}} ${WK} + ${bK} \\, ${V} = x_{\\text{ln}} ${WV} + ${bV}}`,
        attn: `H_i = \\mathrm{softmax}\\left(\\frac{${Q}_i ${K}_i^\\top}{\\sqrt{d_h}} + M\\right) ${V}_i,\\; i=1\\dots h`,
        concat_proj: String.raw`\begin{aligned} H &= \mathrm{Concat}(H_1,\dots,H_h) \\ \mathrm{SA}(x) &= H ${WO} + ${bO} \end{aligned}`,
        resid1: String.raw`u = x + H ${WO} + ${bO}`,
        ln2: String.raw`u_{\text{ln}} = \frac{u - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta`,
        mlp: String.raw`\begin{aligned} z &= \mathrm{GELU}(u_{\text{ln}} ${WUp} + ${bUp}) \\ \mathrm{MLP}(u_{\text{ln}}) &= z ${WDown} + ${bDown} \end{aligned}`,
        resid2: String.raw`x_{\text{out}} = u + \mathrm{MLP}(u_{\text{ln}})`
    };

    function renderEq(tex, title) {
        if (!equationsPanel || !equationsBody) return;
        if (!shouldShowEquations()) return;
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
        if (!shouldShowEquations()) return;
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
            const postAdd = lanes.some(l => ['postMHSAAddition','waitingForLN2'].includes(l.horizPhase));
            const selfAttentionActive = Boolean(
                m?.enableSelfAttentionAnimation
                && m?.selfAttentionAnimator
                && (
                    m.selfAttentionAnimator.phase === 'running'
                    || (typeof m.selfAttentionAnimator.isConveyorActive === 'function'
                        && m.selfAttentionAnimator.isConveyorActive())
                )
            );
            const concatActive = m?.rowMergePhase === 'merging'
                || outPhase === 'vectors_entering'
                || outPhase === 'vectors_inside'
                || outPhase === 'completed';

            if (postAdd) {
                key = 'resid1';
                title = 'Residual Add 1';
            } else if (selfAttentionActive) {
                key = 'attn';
                title = 'Scaled Dot-Product Attention';
            } else if (concatActive) {
                key = 'concat_proj';
                title = 'Concat Heads + W^O';
            } else if (phase === 'parallel_pass_through_active' || phase === 'mha_pass_through_complete') {
                key = 'attn';
                title = 'Scaled Dot-Product Attention';
            } else {
                key = 'qkv_per_head';
                title = 'Q/K/V Projections';
            }
        } else if (ln1Active || layer.isActive) {
            key = 'ln1';
            title = 'LayerNorm 1';
        }
        if (!key) return;
        const eqBody = EQ[key];
        if (key === appState.lastEqKey) return;
        appState.lastEqKey = key;
        const t = (key === 'ln1') ? 'LayerNorm 1 (no in-place assign)' : title;
        renderEq(eqBody, t);
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

        const includeStageLine = !equationsPanel || !shouldShowEquations();
        const nextStatusText = includeStageLine
            ? `Layer ${idx + 1} / ${total}\n${displayStage}`
            : `Layer ${idx + 1} / ${total}`;
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

    const tickEquations = () => {
        if (shouldShowEquations() && pipeline?._layers?.length) {
            const idx = pipeline._currentLayerIdx ?? 0;
            updateEquations(pipeline._layers[idx]);
        }
        scheduleFrame(tickEquations);
    };
    tickEquations();
}
