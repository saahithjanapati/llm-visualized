import * as THREE from 'three';
import { MHSAAnimation } from '../../animations/MHSAAnimation.js';
import { loadPrecomputedGeometries } from '../../utils/precomputedGeometryLoader.js';
import { LayerPipeline } from '../../engine/LayerPipeline.js';
import {
    setPlaybackSpeed,
    setNumVectorLanes,
    NUM_VECTOR_LANES,
    USE_INSTANCED_MATRIX_SLICES
} from '../../utils/constants.js';
import { setAnimationLaneCount } from '../../animations/LayerAnimationConstants.js';
import { appState } from '../../state/appState.js';
import { initIntroAnimation } from '../../ui/introAnimation.js';
import { initStatusOverlay } from '../../ui/statusOverlay.js';
import { initParameterCounter } from '../../ui/parameterCounter.js';
import { initSettingsModal } from '../../ui/settingsModal.js';
import { initPauseButton } from '../../ui/pauseButton.js';
import { initConveyorSkipButton } from '../../ui/conveyorSkipButton.js';
import { initSkipToEndButton } from '../../ui/skipToEndButton.js';
import { initSkipLayerButton } from '../../ui/skipLayerButton.js';
import { initSkipNextPassButton } from '../../ui/skipNextPassButton.js';
import { initSkipLastPassButton } from '../../ui/skipLastPassButton.js';
import { initSkipMenu } from '../../ui/skipMenu.js';
import { initSelectionPanel } from '../../ui/selectionPanel.js';
import { initPromptTokenStrip } from '../../ui/promptTokenStrip.js';
import { loadActivationState } from './activation.js';
import { initFollowModeControls, initTopControlsAutohide } from './topControls.js';
import { buildPassState, initGenerationController } from './generationController.js';
import { formatTokenLabel } from './tokenLabels.js';
import {
    NUM_LAYERS,
    PROMPT_TOKENS,
    POSITION_TOKENS,
    DEFAULT_PROMPT_LANES,
    CAMERA_CONFIG
} from './config.js';

const vec3 = (arr) => new THREE.Vector3(arr[0], arr[1], arr[2]);

// Optionally load pre-baked geometries to skip heavy procedural work.
await loadPrecomputedGeometries('/precomputed_components_slice.glb');
// Skip full-depth QKV/output precompute when instanced slices are active,
// otherwise those matrices will appear longer than the rest.
if (!USE_INSTANCED_MATRIX_SLICES) {
    await loadPrecomputedGeometries('/precomputed_components_qkv.glb');
}

const statusDiv = document.getElementById('statusOverlay');
const {
    activationSource,
    laneCount,
    isFullTokenMode
} = await loadActivationState({
    defaultLaneCount: NUM_VECTOR_LANES,
    defaultPromptLanes: DEFAULT_PROMPT_LANES,
    promptTokens: PROMPT_TOKENS,
    positionTokens: POSITION_TOKENS,
    layerCount: NUM_LAYERS,
    statusElement: statusDiv
});

let initialLaneCount = laneCount;
if (!isFullTokenMode && activationSource && typeof activationSource.getTokenCount === 'function') {
    const tokenCount = activationSource.getTokenCount();
    const metaPromptCount = Array.isArray(activationSource?.meta?.prompt_tokens)
        ? activationSource.meta.prompt_tokens.length
        : null;
    const promptCount = Number.isFinite(metaPromptCount)
        ? metaPromptCount
        : DEFAULT_PROMPT_LANES;
    if (Number.isFinite(tokenCount) && tokenCount > 0) {
        initialLaneCount = Math.max(1, Math.min(tokenCount, promptCount || laneCount));
    }
}

const initialPassState = buildPassState({
    activationSource,
    laneCount: initialLaneCount,
    laneLayoutIndices: Array.from({ length: initialLaneCount }, (_, idx) => idx),
    totalLaneCount: initialLaneCount,
    fallbackTokenLabels: PROMPT_TOKENS,
    fallbackPositionLabels: POSITION_TOKENS
});

// Sync global lane counts so component spacing/animation widths align.
setNumVectorLanes(initialLaneCount);
setAnimationLaneCount(initialLaneCount);

// Skip intro typing screen for direct animation entry.
appState.skipIntro = true;

// Set default playback speed to medium on load.
let defaultSpeedPreset = null;
try { defaultSpeedPreset = setPlaybackSpeed('medium'); } catch (_) { /* no-op */ }

// GPT-2 tower - initialize immediately.
MHSAAnimation.ENABLE_SELF_ATTENTION = true;
const gptCanvas = document.getElementById('gptCanvas');
const camPos = vec3(CAMERA_CONFIG.initialPosition);
const camTarget = vec3(CAMERA_CONFIG.initialTarget);
const skipToEndCamPos = camPos.clone().add(vec3(CAMERA_CONFIG.skipToEndPositionOffset));
const skipToEndCamTarget = camTarget.clone().add(vec3(CAMERA_CONFIG.skipToEndTargetOffset));
const targetClampRadius = Math.max(
    CAMERA_CONFIG.targetClampRadiusBase,
    NUM_LAYERS * CAMERA_CONFIG.targetClampRadiusPerLayer
);
const pipeline = new LayerPipeline(gptCanvas, NUM_LAYERS, {
    cameraPosition: camPos,
    cameraTarget: camTarget,
    skipToEndCameraPosition: skipToEndCamPos,
    skipToEndCameraTarget: skipToEndCamTarget,
    targetClampCenter: camTarget,
    targetClampRadius,
    mobileZoomOutMultiplier: CAMERA_CONFIG.mobileZoomOutMultiplier,
    autoCameraHeadBias: CAMERA_CONFIG.autoCameraHeadBias,
    autoCameraDefaultCameraOffset: vec3(CAMERA_CONFIG.followDefaultCameraOffset),
    autoCameraDefaultTargetOffset: vec3(CAMERA_CONFIG.followDefaultTargetOffset),
    autoCameraEmbedVocabCameraOffset: vec3(CAMERA_CONFIG.followEmbedVocabCameraOffset),
    autoCameraEmbedVocabTargetOffset: vec3(CAMERA_CONFIG.followEmbedVocabTargetOffset),
    autoCameraEmbedPositionCameraOffset: vec3(CAMERA_CONFIG.followEmbedPositionCameraOffset),
    autoCameraEmbedPositionTargetOffset: vec3(CAMERA_CONFIG.followEmbedPositionTargetOffset),
    autoCameraEmbedAddCameraOffset: vec3(CAMERA_CONFIG.followEmbedAddCameraOffset),
    autoCameraEmbedAddTargetOffset: vec3(CAMERA_CONFIG.followEmbedAddTargetOffset),
    autoCameraMhsaCameraOffset: vec3(CAMERA_CONFIG.followMhsaCameraOffset),
    autoCameraMhsaTargetOffset: vec3(CAMERA_CONFIG.followMhsaTargetOffset),
    autoCameraMhsaMobileCameraOffset: vec3(CAMERA_CONFIG.followMhsaMobileCameraOffset),
    autoCameraMhsaMobileTargetOffset: vec3(CAMERA_CONFIG.followMhsaMobileTargetOffset),
    autoCameraConcatCameraOffset: vec3(CAMERA_CONFIG.followConcatCameraOffset),
    autoCameraConcatTargetOffset: vec3(CAMERA_CONFIG.followConcatTargetOffset),
    autoCameraConcatMobileCameraOffset: vec3(CAMERA_CONFIG.followConcatMobileCameraOffset),
    autoCameraConcatMobileTargetOffset: vec3(CAMERA_CONFIG.followConcatMobileTargetOffset),
    autoCameraLnCameraOffset: vec3(CAMERA_CONFIG.followLnCameraOffset),
    autoCameraLnTargetOffset: vec3(CAMERA_CONFIG.followLnTargetOffset),
    autoCameraTravelCameraOffset: vec3(CAMERA_CONFIG.followTravelCameraOffset),
    autoCameraTravelTargetOffset: vec3(CAMERA_CONFIG.followTravelTargetOffset),
    autoCameraTravelMobileCameraOffset: vec3(CAMERA_CONFIG.followTravelMobileCameraOffset),
    autoCameraTravelMobileTargetOffset: vec3(CAMERA_CONFIG.followTravelMobileTargetOffset),
    autoCameraFinalCameraOffset: vec3(CAMERA_CONFIG.followFinalCameraOffset),
    autoCameraFinalTargetOffset: vec3(CAMERA_CONFIG.followFinalTargetOffset),
    autoCameraFinalMobileCameraOffset: vec3(CAMERA_CONFIG.followFinalMobileCameraOffset),
    autoCameraFinalMobileTargetOffset: vec3(CAMERA_CONFIG.followFinalMobileTargetOffset),
    autoCameraLayerEndDesktopCameraOffset: vec3(CAMERA_CONFIG.followLayerEndDesktopCameraOffset),
    autoCameraLayerEndDesktopTargetOffset: vec3(CAMERA_CONFIG.followLayerEndDesktopTargetOffset),
    autoCameraMobileScale: CAMERA_CONFIG.autoCameraMobileScale,
    autoCameraMobileShiftX: CAMERA_CONFIG.autoCameraMobileShiftX,
    autoCameraMhsaMobileShiftX: CAMERA_CONFIG.autoCameraMhsaMobileShiftX,
    autoCameraTravelMobileShiftX: CAMERA_CONFIG.autoCameraTravelMobileShiftX,
    autoCameraScaleMinWidth: CAMERA_CONFIG.autoCameraScaleMinWidth,
    autoCameraScaleMaxWidth: CAMERA_CONFIG.autoCameraScaleMaxWidth,
    autoCameraSmoothAlpha: CAMERA_CONFIG.autoCameraSmoothAlpha,
    autoCameraOffsetLerpAlpha: CAMERA_CONFIG.autoCameraOffsetLerpAlpha,
    autoCameraViewBlendAlpha: CAMERA_CONFIG.autoCameraViewBlendAlpha,
    activationSource,
    laneCount: initialLaneCount
});

if (defaultSpeedPreset && typeof defaultSpeedPreset.engineSpeed === 'number') {
    pipeline?.engine?.setSpeed?.(defaultSpeedPreset.engineSpeed);
}

// Show GPT canvas immediately.
gptCanvas.style.display = 'block';
try {
    const eng = pipeline.engine;
    // Keep shadows off for this demo to reduce GPU cost.
    eng.renderer.shadowMap.enabled = false;
    eng.scene.traverse((obj) => {
        if (obj.isMesh) { obj.castShadow = false; obj.receiveShadow = false; }
        if (obj.isLight) { obj.castShadow = false; }
    });
} catch (_) { /* no-op */ }

// Initialize UI modules (status, settings, pause/skip, selection panel).
initIntroAnimation(pipeline, gptCanvas);
initStatusOverlay(pipeline, NUM_LAYERS);
initParameterCounter(pipeline, NUM_LAYERS);
initPauseButton(pipeline);
initConveyorSkipButton(pipeline);
initSkipToEndButton(pipeline);
initSkipLayerButton(pipeline);
initSkipMenu(pipeline);
initSettingsModal(pipeline);

const followModeBtn = document.getElementById('followModeBtn');
const followModeControls = initFollowModeControls({
    pipeline,
    appState,
    followModeBtn
});

const topControls = document.getElementById('topControls');
const settingsOverlay = document.getElementById('settingsOverlay');
const { showTopControls } = initTopControlsAutohide({ topControls, settingsOverlay });

const selectionPanel = initSelectionPanel({
    activationSource,
    laneTokenIndices: initialPassState.laneTokenIndices,
    tokenLabels: initialPassState.tokenLabels,
    engine: pipeline.engine,
    pipeline,
    pauseMainFlowOnMobileFocus: true
});

const findSelectionObjectByLabel = (label) => {
    if (!label || typeof label !== 'string') return null;
    const scene = pipeline?.engine?.scene;
    if (!scene || typeof scene.traverse !== 'function') return null;
    let fallback = null;
    let preferred = null;
    scene.traverse((node) => {
        if (!node || !node.userData || node.userData.label !== label) return;
        if (!fallback) fallback = node;
        if (node.visible === false) return;
        if (node.isLine || node.isLineSegments) return;
        if (!preferred) preferred = node;
    });
    return preferred || fallback;
};

const promptTokenStrip = initPromptTokenStrip({
    onTokenClick: (tokenEntry) => {
        const tokenLabel = typeof tokenEntry?.tokenLabel === 'string' ? tokenEntry.tokenLabel : '';
        const displayToken = formatTokenLabel(tokenLabel);
        const isGeneratedToken = tokenEntry?.entryType === 'generated';
        const info = {
            tokenLabel
        };
        if (Number.isFinite(tokenEntry?.laneIndex)) info.laneIndex = Math.floor(tokenEntry.laneIndex);
        if (Number.isFinite(tokenEntry?.tokenIndex)) info.tokenIndex = Math.floor(tokenEntry.tokenIndex);
        if (Number.isFinite(tokenEntry?.tokenId)) info.tokenId = Math.floor(tokenEntry.tokenId);
        if (tokenEntry?.logitEntry && typeof tokenEntry.logitEntry === 'object') {
            info.logitEntry = tokenEntry.logitEntry;
        }

        let label = `Token: ${displayToken}`;
        let selectionObject = null;
        if (isGeneratedToken) {
            const chosenLabel = (typeof tokenEntry?.selectionLabel === 'string' && tokenEntry.selectionLabel.length)
                ? tokenEntry.selectionLabel
                : `Chosen token: ${displayToken}`;
            label = chosenLabel;
            selectionObject = findSelectionObjectByLabel(chosenLabel);
        }

        followModeControls?.suppressPendingFollowDisable?.();
        const selection = {
            label,
            info,
            kind: 'label'
        };
        if (selectionObject) selection.object = selectionObject;
        selectionPanel.handleSelection(selection);
    }
});
if (pipeline.engine && typeof pipeline.engine.setRaycastSelectionHandler === 'function') {
    pipeline.engine.setRaycastSelectionHandler(selection => {
        if (!selection || !selection.label) {
            followModeControls?.setFollowMode?.(false);
            showTopControls();
            return;
        }
        followModeControls?.suppressPendingFollowDisable?.();
        selectionPanel.handleSelection(selection);
    });
}

const generationController = initGenerationController({
    pipeline,
    activationSource,
    initialLaneCount,
    initialPassState,
    fallbackTokenLabels: PROMPT_TOKENS,
    fallbackPositionLabels: POSITION_TOKENS,
    numLayers: NUM_LAYERS,
    cameraReturnPosition: camPos,
    cameraReturnTarget: camTarget,
    selectionPanel,
    promptTokenStrip
});

initSkipNextPassButton({ pipeline, generationController });
initSkipLastPassButton({ pipeline, generationController });
