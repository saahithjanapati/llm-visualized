import * as THREE from 'three';
import { CoreEngine } from './CoreEngine.js';
import Gpt2Layer from './layers/Gpt2Layer.js';
import { createRandomSource } from '../data/RandomActivationSource.js';
import {
    MLP_MATRIX_PARAMS_DOWN,
    EMBEDDING_MATRIX_PARAMS_VOCAB,
    TOP_EMBED_Y_GAP_ABOVE_TOWER,
    TOP_EMBED_Y_ADJUST,
    GLOBAL_ANIM_SPEED_MULT,
    SELF_ATTENTION_TIME_MULT,
    ANIM_RISE_SPEED_ORIGINAL,
    LN_PARAMS,
    LN_NORM_START_FRACTION_FROM_BOTTOM,
    PRISM_ADD_ANIM_BASE_DURATION,
    PRISM_ADD_ANIM_BASE_FLASH_DURATION,
    PRISM_ADD_ANIM_BASE_DELAY_BETWEEN_PRISMS,
    PRISM_ADD_ANIM_SPEED_MULT,
    HIDE_INSTANCE_Y_OFFSET,
    setGlobalAnimSpeedMult,
    setPrismAddAnimSpeedMult,
    setSelfAttentionTimeMult,
    VECTOR_LENGTH_PRISM,
    NUM_VECTOR_LANES
} from '../utils/constants.js';
import { VectorVisualizationInstancedPrism } from '../components/VectorVisualizationInstancedPrism.js';
import { startPrismAdditionAnimation } from '../utils/additionUtils.js';
import { PrismLayerNormAnimation } from '../animations/PrismLayerNormAnimation.js';

function simplePrismMultiply(srcVec, tgtVec, onComplete) {
    const srcCount = srcVec && Number.isFinite(srcVec.instanceCount) ? srcVec.instanceCount : VECTOR_LENGTH_PRISM;
    const tgtCount = tgtVec && Number.isFinite(tgtVec.instanceCount) ? tgtVec.instanceCount : VECTOR_LENGTH_PRISM;
    const srcLen = srcVec && Array.isArray(srcVec.rawData) ? srcVec.rawData.length : srcCount;
    const tgtLen = tgtVec && Array.isArray(tgtVec.rawData) ? tgtVec.rawData.length : tgtCount;
    const length = Math.min(srcCount, tgtCount, srcLen, tgtLen);
    for (let i = 0; i < length; i++) {
        tgtVec.rawData[i] = (srcVec.rawData[i] || 0) * (tgtVec.rawData[i] || 0);
    }
    const numKeyColors = Math.min(30, Math.max(1, tgtVec.rawData.length || 1));
    tgtVec.updateKeyColorsFromData(tgtVec.rawData, numKeyColors);
    if (onComplete) onComplete();
}

const COLOR_DARK_GRAY = new THREE.Color(0x333333);
const COLOR_LIGHT_YELLOW = new THREE.Color(0xffffff);
const COLOR_BRIGHT_YELLOW = new THREE.Color(0xffffff);

const TMP_WORLD_POS = new THREE.Vector3();
const TMP_CENTER_A = new THREE.Vector3();
const TMP_CENTER_B = new THREE.Vector3();
const TMP_CENTER_MAT_A = new THREE.Matrix4();
const TMP_CENTER_MAT_B = new THREE.Matrix4();

/**
 * LayerPipeline orchestrates a single bundle of vectors ("lanes") through an
 * arbitrary stack of GPT-2 transformer layers.  Unlike the old approach that
 * created a fresh bundle for every layer at T=0, this controller initialises
 * ONE set of vectors, then – once a layer finishes – hands those same Three
 * objects off to the next Gpt2Layer positioned above the previous.
 */
export class LayerPipeline extends EventTarget {
    /**
     * @param {HTMLCanvasElement} canvas – Render target for the CoreEngine.
     * @param {number}           numLayers – Total layers in the stack.
     * @param {Object}           [opts] – Additional CoreEngine options (camera, speed, etc.).
     * @param {() => any}        [opts.randomFactory] – Factory that produces a fresh random-source for each layer.
     */
    constructor(canvas, numLayers = 12, opts = {}) {
        super();
        if (!canvas) throw new Error('LayerPipeline requires a renderCanvas element');
        this._numLayers = Math.max(1, numLayers);
        this._canvas    = canvas;
        this._opts      = opts;
        this._randFactory = typeof opts.randomFactory === 'function' ? opts.randomFactory : createRandomSource;
        this._activationSource = opts.activationSource || null;
        this._laneCount = Math.max(1, Math.floor(opts.laneCount || NUM_VECTOR_LANES));

        this._layers = [];
        this._currentLayerIdx = 0;
        this._skipToEndActive = false;
        this._skipToEndRestore = null;
        this._skipToEndRaf = null;
        this._forwardPassComplete = false;
        this._skipLayerActive = false;
        this._skipLayerRestore = null;
        this._skipLayerLast = false;

        this._autoCameraFollow = opts.autoCameraFollow !== false;
        const smoothAlpha = typeof opts.autoCameraSmoothAlpha === 'number' ? opts.autoCameraSmoothAlpha : 0.14;
        this._autoCameraSmoothAlpha = Math.min(1, Math.max(0, smoothAlpha));
        const offsetAlpha = typeof opts.autoCameraOffsetLerpAlpha === 'number' ? opts.autoCameraOffsetLerpAlpha : 0.14;
        this._autoCameraOffsetLerpAlpha = Math.min(1, Math.max(0, offsetAlpha));
        const viewBlendAlpha = typeof opts.autoCameraViewBlendAlpha === 'number' ? opts.autoCameraViewBlendAlpha : 0.12;
        this._autoCameraViewBlendAlpha = Math.min(1, Math.max(0, viewBlendAlpha));
        const mlpBlendAlpha = typeof opts.autoCameraViewBlendAlphaMlpReturn === 'number'
            ? opts.autoCameraViewBlendAlphaMlpReturn
            : (this._autoCameraViewBlendAlpha * 0.35);
        this._autoCameraViewBlendAlphaMlpReturn = Math.min(1, Math.max(0, mlpBlendAlpha));
        this._autoCameraViewBlendAlphaActive = this._autoCameraViewBlendAlpha;
        const mobileScale = typeof opts.autoCameraMobileScale === 'number' ? opts.autoCameraMobileScale : 1.0;
        this._autoCameraScaleMax = Math.max(1.0, mobileScale);
        const mobileShiftX = typeof opts.autoCameraMobileShiftX === 'number' ? opts.autoCameraMobileShiftX : 0;
        this._autoCameraShiftMaxX = mobileShiftX;
        const travelShiftX = typeof opts.autoCameraTravelMobileShiftX === 'number' ? opts.autoCameraTravelMobileShiftX : 0;
        this._autoCameraShiftMaxTravelX = travelShiftX;
        const mhsaShiftX = typeof opts.autoCameraMhsaMobileShiftX === 'number' ? opts.autoCameraMhsaMobileShiftX : 0;
        this._autoCameraShiftMaxMhsaX = mhsaShiftX;
        this._autoCameraTravelMobileOverrideCameraOffset = null;
        this._autoCameraTravelMobileOverrideTargetOffset = null;
        this._autoCameraTravelMobileOverrideEnabled = false;
        this._autoCameraScaleMinWidth = Number.isFinite(opts.autoCameraScaleMinWidth)
            ? Math.max(200, opts.autoCameraScaleMinWidth)
            : 360;
        this._autoCameraScaleMaxWidth = Number.isFinite(opts.autoCameraScaleMaxWidth)
            ? Math.max(this._autoCameraScaleMinWidth + 10, opts.autoCameraScaleMaxWidth)
            : 880;
        this._autoCameraScaleLast = 1.0;
        this._autoCameraShiftLastX = 0;
        this._autoCameraCenter = new THREE.Vector3();
        this._autoCameraOffsetScratch = new THREE.Vector3();
        this._autoCameraDesiredCameraOffset = new THREE.Vector3();
        this._autoCameraDesiredTargetOffset = new THREE.Vector3();
        this._autoCameraCurrentCameraOffset = new THREE.Vector3();
        this._autoCameraCurrentTargetOffset = new THREE.Vector3();
        this._autoCameraSmoothedRef = new THREE.Vector3();
        this._autoCameraSmoothValid = false;
        this._autoCameraViewKey = 'default';
        this._autoCameraViewBlendT = 1;
        this._autoCameraViewFromCameraOffset = new THREE.Vector3();
        this._autoCameraViewFromTargetOffset = new THREE.Vector3();
        this._autoCameraViewToCameraOffset = new THREE.Vector3();
        this._autoCameraViewToTargetOffset = new THREE.Vector3();
        this._autoCameraViewBlendCameraOffset = new THREE.Vector3();
        this._autoCameraViewBlendTargetOffset = new THREE.Vector3();
        this._autoCameraViewContext = null;
        this._autoCameraDefaultCameraOffset = new THREE.Vector3();
        this._autoCameraDefaultTargetOffset = new THREE.Vector3();
        this._autoCameraMhsaCameraOffset = new THREE.Vector3();
        this._autoCameraMhsaTargetOffset = new THREE.Vector3();
        this._autoCameraConcatCameraOffset = new THREE.Vector3();
        this._autoCameraConcatTargetOffset = new THREE.Vector3();
        this._autoCameraMhsaOffsetsEnabled = false;
        this._autoCameraConcatOffsetsEnabled = false;
        this._autoCameraLnCameraOffset = new THREE.Vector3();
        this._autoCameraLnTargetOffset = new THREE.Vector3();
        this._autoCameraTravelCameraOffset = new THREE.Vector3();
        this._autoCameraTravelTargetOffset = new THREE.Vector3();
        this._autoCameraLnOffsetsEnabled = false;
        this._autoCameraTravelOffsetsEnabled = false;
        this._autoCameraDefaultCameraOffsetBase = new THREE.Vector3();
        this._autoCameraDefaultTargetOffsetBase = new THREE.Vector3();
        this._autoCameraMhsaCameraOffsetBase = new THREE.Vector3();
        this._autoCameraMhsaTargetOffsetBase = new THREE.Vector3();
        this._autoCameraConcatCameraOffsetBase = new THREE.Vector3();
        this._autoCameraConcatTargetOffsetBase = new THREE.Vector3();
        this._autoCameraLnCameraOffsetBase = new THREE.Vector3();
        this._autoCameraLnTargetOffsetBase = new THREE.Vector3();
        this._autoCameraTravelCameraOffsetBase = new THREE.Vector3();
        this._autoCameraTravelTargetOffsetBase = new THREE.Vector3();
        this._autoCameraInspectorRef = new THREE.Vector3();
        this._hasAutoCameraOffsets = false;
        this._suppressControlsChange = false;
        this._devMode = !!opts.devMode;
        this._cameraOffsetDiv = (typeof document !== 'undefined')
            ? document.getElementById('cameraOffsetOverlay')
            : null;
        this._controlsChangeHandler = null;
        this._onAutoCameraProgress = () => { this._maybeAutoCameraFocus(); };
        this._cameraOverlayRaf = null;
        this.addEventListener('progress', this._onAutoCameraProgress);

        // ------------------------------------------------------------------
        // Pre-create *all* layers so their static visuals are visible upfront.
        // Only the first layer is active immediately; higher layers remain
        // dormant until their turn, at which point we inject residual lanes.
        // ------------------------------------------------------------------

        const engineOpts = { ...opts };
        delete engineOpts.laneCount;
        if (typeof engineOpts.cameraFarMargin !== 'number') {
            const DEFAULT_CAMERA_FAR_MARGIN = 40000;
            // Provide additional depth so tall transformer stacks remain visible
            // when the user zooms far away from the tower. The allowance scales
            // with layer count but never drops below the default margin.
            const approxTowerAllowance = Math.max(DEFAULT_CAMERA_FAR_MARGIN, this._numLayers * 1800);
            engineOpts.cameraFarMargin = approxTowerAllowance;
        }
        this._engine = new CoreEngine(canvas, [], engineOpts);

        const parseVec3 = (value, fallback = null) => {
            if (value instanceof THREE.Vector3) return value.clone();
            if (value && typeof value === 'object' && 'x' in value && 'y' in value && 'z' in value) {
                return new THREE.Vector3(value.x, value.y, value.z);
            }
            return fallback ? fallback.clone() : null;
        };

        this._overviewCameraPosition = parseVec3(
            (opts.skipToEndCameraPosition ?? opts.cameraPosition) ?? null,
            null
        );
        this._overviewCameraTarget = parseVec3(
            (opts.skipToEndCameraTarget ?? opts.cameraTarget) ?? null,
            null
        );

        const defaultTargetOffset = parseVec3(opts.autoCameraDefaultTargetOffset, new THREE.Vector3(0, 0, 0));
        const cameraTarget = this._engine?.controls?.target || new THREE.Vector3();
        const defaultCameraOffset = parseVec3(
            opts.autoCameraDefaultCameraOffset,
            this._engine?.camera?.position ? this._engine.camera.position.clone().sub(cameraTarget) : new THREE.Vector3(0, 2000, 16000)
        );
        if (defaultTargetOffset) this._autoCameraDefaultTargetOffsetBase.copy(defaultTargetOffset);
        if (defaultCameraOffset) this._autoCameraDefaultCameraOffsetBase.copy(defaultCameraOffset);

        const mhsaCamOffset = parseVec3(opts.autoCameraMhsaCameraOffset, null);
        const mhsaTargetOffset = parseVec3(opts.autoCameraMhsaTargetOffset, null);
        if (mhsaCamOffset && mhsaTargetOffset) {
            this._autoCameraMhsaCameraOffsetBase.copy(mhsaCamOffset);
            this._autoCameraMhsaTargetOffsetBase.copy(mhsaTargetOffset);
            this._autoCameraMhsaOffsetsEnabled = true;
        }

        const concatCamOffset = parseVec3(opts.autoCameraConcatCameraOffset, null);
        const concatTargetOffset = parseVec3(opts.autoCameraConcatTargetOffset, null);
        if (concatCamOffset && concatTargetOffset) {
            this._autoCameraConcatCameraOffsetBase.copy(concatCamOffset);
            this._autoCameraConcatTargetOffsetBase.copy(concatTargetOffset);
            this._autoCameraConcatOffsetsEnabled = true;
        }

        const lnCamOffset = parseVec3(opts.autoCameraLnCameraOffset, null);
        const lnTargetOffset = parseVec3(opts.autoCameraLnTargetOffset, null);
        if (lnCamOffset && lnTargetOffset) {
            this._autoCameraLnCameraOffsetBase.copy(lnCamOffset);
            this._autoCameraLnTargetOffsetBase.copy(lnTargetOffset);
            this._autoCameraLnOffsetsEnabled = true;
        }

        const travelCamOffset = parseVec3(opts.autoCameraTravelCameraOffset, null);
        const travelTargetOffset = parseVec3(opts.autoCameraTravelTargetOffset, null);
        if (travelCamOffset && travelTargetOffset) {
            this._autoCameraTravelCameraOffsetBase.copy(travelCamOffset);
            this._autoCameraTravelTargetOffsetBase.copy(travelTargetOffset);
            this._autoCameraTravelOffsetsEnabled = true;
        }
        const travelMobileCamOffset = parseVec3(opts.autoCameraTravelMobileCameraOffset, null);
        const travelMobileTargetOffset = parseVec3(opts.autoCameraTravelMobileTargetOffset, null);
        if (travelMobileCamOffset && travelMobileTargetOffset) {
            this._autoCameraTravelMobileOverrideCameraOffset = travelMobileCamOffset;
            this._autoCameraTravelMobileOverrideTargetOffset = travelMobileTargetOffset;
            this._autoCameraTravelMobileOverrideEnabled = true;
        }

        this._updateAutoCameraScaledOffsets(true);

        if (this._engine?.controls) {
            this._controlsChangeHandler = () => {
                if (!this._autoCameraFollow || this._suppressControlsChange) {
                    return;
                }
                this._captureAutoCameraOffsets();
                this._updateCameraOffsetOverlay();
            };
            this._engine.controls.addEventListener('change', this._controlsChangeHandler);
        }

        for (let i = 0; i < this._numLayers; i++) {
            const rand = this._randFactory();
            const isActive = i === 0; // only first layer active initially
            const layer = new Gpt2Layer(
                i,
                rand,
                0,
                /*externalLanes*/ null,
                /*onFinished*/ null,
                isActive,
                this._activationSource,
                this._laneCount
            );

            // Assign onFinished callback for chaining once layer becomes active
            layer.setOnFinished(() => this._advanceToNextLayer());
            layer.setProgressEmitter(this);

            layer.init(this._engine.scene);
            if (typeof this._engine.registerRaycastRoot === 'function') {
                this._engine.registerRaycastRoot(layer.raycastRoot || layer.root);
            }
            this._layers.push(layer);
            this._engine._layers.push(layer); // add to engine update list
        }

        // Ensure first layer has active callback wired before start
        this._layers[0].setOnFinished(() => this._advanceToNextLayer());
        this._layers[0].setProgressEmitter(this);

        this._maybeAutoCameraFocus({ immediate: true });
        if (this._autoCameraFollow) {
            this._startCameraOverlayLoop();
        }
    }

    /** Dispose and tear down Three resources */
    dispose() {
        if (this._onAutoCameraProgress) {
            this.removeEventListener('progress', this._onAutoCameraProgress);
            this._onAutoCameraProgress = null;
        }
        if (this._engine?.controls && this._controlsChangeHandler) {
            this._engine.controls.removeEventListener('change', this._controlsChangeHandler);
        }
        this._controlsChangeHandler = null;
        if (this._cameraOffsetDiv) {
            this._cameraOffsetDiv.style.display = 'none';
        }
        if (this._skipToEndRaf) {
            if (typeof window !== 'undefined' && typeof window.cancelAnimationFrame === 'function') {
                window.cancelAnimationFrame(this._skipToEndRaf);
            } else {
                clearTimeout(this._skipToEndRaf);
            }
            this._skipToEndRaf = null;
        }
        this._stopCameraOverlayLoop();
        if (this._engine) {
            this._engine.dispose();
        }
    }

    /** Return reference to internal CoreEngine (for advanced use-cases). */
    get engine() { return this._engine; }

    /** Enable or disable automatic camera tracking of the active layer. */
    setAutoCameraFollow(enabled, { immediate = false, resetView = false, smoothReset = false } = {}) {
        const nextValue = !!enabled;
        if (nextValue === this._autoCameraFollow) {
            if (nextValue) {
                if (immediate) {
                    this._updateAutoCameraFollow();
                    this._updateCameraOffsetOverlay();
                }
            } else {
                this._updateCameraOffsetOverlay();
            }
            return;
        }
        this._autoCameraFollow = nextValue;
        if (this._autoCameraFollow) {
            this._autoCameraSmoothValid = false;
            if (resetView) {
                const canSmoothReset = smoothReset && this._captureAutoCameraOffsets();
                this._setAutoCameraOffsets(
                    this._autoCameraDefaultCameraOffset,
                    this._autoCameraDefaultTargetOffset,
                    { snap: !canSmoothReset }
                );
            }
        }
        if (this._autoCameraFollow) {
            if (immediate) {
                this._updateAutoCameraFollow();
            }
            this._startCameraOverlayLoop();
        } else {
            this._clearAutoCameraOffsets();
            this._stopCameraOverlayLoop();
        }
        this._updateCameraOffsetOverlay();
    }

    /** Check whether automatic camera tracking is enabled. */
    isAutoCameraFollowEnabled() {
        return !!this._autoCameraFollow;
    }

    /** Snap the camera to the overview framing (typically the initial tower view). */
    focusOverview() {
        const engine = this._engine;
        const camera = engine?.camera;
        const controls = engine?.controls;
        if (!camera || !controls || !controls.target) return;
        if (!this._overviewCameraPosition || !this._overviewCameraTarget) return;
        camera.position.copy(this._overviewCameraPosition);
        controls.target.copy(this._overviewCameraTarget);
        if (typeof controls.update === 'function') controls.update();
        engine?.notifyCameraUpdated?.();
    }

    /** Get current follow reference position (residual stream center). */
    getAutoCameraReference() {
        const ref = this._autoCameraInspectorRef;
        const info = this._resolveActiveLanePosition(ref);
        if (!info || info.laneIndex < 0) return null;
        return {
            laneIndex: info.laneIndex,
            position: { x: ref.x, y: ref.y, z: ref.z }
        };
    }

    isSkipToEndActive() {
        return !!this._skipToEndActive;
    }

    isSkipLayerActive() {
        return !!this._skipLayerActive;
    }

    isForwardPassComplete() {
        return this._checkForwardPassComplete();
    }

    skipCurrentLayer(opts = {}) {
        if (this._skipLayerActive || this._skipToEndActive) return;
        if (this._checkForwardPassComplete()) return;

        const layer = this._layers[this._currentLayerIdx];
        if (!layer || layer._completed) return;

        const engine = this._engine;
        this._skipLayerActive = true;
        this._skipLayerLast = this._currentLayerIdx >= this._numLayers - 1;
        this._skipLayerRestore = {
            engineSpeed: engine && typeof engine._speed === 'number' ? engine._speed : 1,
            globalSpeed: GLOBAL_ANIM_SPEED_MULT,
            prismAddSpeed: PRISM_ADD_ANIM_SPEED_MULT,
            selfAttentionSpeed: SELF_ATTENTION_TIME_MULT
        };

        const {
            engineSpeed = 6,
            globalSpeed = 800,
            prismAddSpeed = 200,
            selfAttentionSpeed = 0.03
        } = opts || {};

        if (engine && typeof engine.setSpeed === 'function') {
            engine.setSpeed(engineSpeed);
        }
        setGlobalAnimSpeedMult(globalSpeed);
        setPrismAddAnimSpeedMult(prismAddSpeed);
        setSelfAttentionTimeMult(selfAttentionSpeed);
    }

    skipToEndForwardPass(opts = {}) {
        if (this._skipToEndActive || this._checkForwardPassComplete()) return;
        if (this._skipLayerActive) {
            this._restoreSkipLayerSpeeds();
        }
        this._skipToEndActive = true;

        const engine = this._engine;
        if (engine && typeof engine.resume === 'function') {
            engine.resume('manual');
        }

        this._skipToEndRestore = {
            engineSpeed: engine && typeof engine._speed === 'number' ? engine._speed : 1,
            globalSpeed: GLOBAL_ANIM_SPEED_MULT,
            prismAddSpeed: PRISM_ADD_ANIM_SPEED_MULT,
            selfAttentionSpeed: SELF_ATTENTION_TIME_MULT
        };

        const {
            engineSpeed = 6,
            globalSpeed = 800,
            prismAddSpeed = 200,
            selfAttentionSpeed = 0.03
        } = opts || {};

        if (engine && typeof engine.setSpeed === 'function') {
            engine.setSpeed(engineSpeed);
        }
        setGlobalAnimSpeedMult(globalSpeed);
        setPrismAddAnimSpeedMult(prismAddSpeed);
        setSelfAttentionTimeMult(selfAttentionSpeed);

        this._layers.forEach(layer => {
            if (layer && typeof layer.setSkipToEndMode === 'function') {
                layer.setSkipToEndMode(true);
            }
        });

        this._startSkipCompletionWatch();
    }

    // ----------------------------------------------------------------------
    // Private helpers
    // ----------------------------------------------------------------------

    _checkForwardPassComplete() {
        if (this._forwardPassComplete) return true;
        if (!Array.isArray(this._layers) || !this._layers.length) return false;
        const allLayersDone = this._layers.every(layer => layer && layer._completed);
        if (!allLayersDone) return false;

        const lastLayer = this._layers[this._numLayers - 1];
        const stopY = lastLayer && lastLayer.__topEmbedStopYLocal;
        if (Number.isFinite(stopY)) {
            const lanes = Array.isArray(lastLayer.lanes) ? lastLayer.lanes : [];
            if (!lanes.length) return false;
            const allAtTop = lanes.every(lane => {
                const y = lane && lane.originalVec && lane.originalVec.group && lane.originalVec.group.position
                    ? lane.originalVec.group.position.y
                    : NaN;
                return Number.isFinite(y) && y >= stopY - 0.5;
            });
            if (!allAtTop) return false;
        }

        this._forwardPassComplete = true;
        return true;
    }

    _startSkipCompletionWatch() {
        if (this._skipToEndRaf) {
            if (typeof window !== 'undefined' && typeof window.cancelAnimationFrame === 'function') {
                window.cancelAnimationFrame(this._skipToEndRaf);
            } else {
                clearTimeout(this._skipToEndRaf);
            }
            this._skipToEndRaf = null;
        }

        const schedule = (typeof window !== 'undefined' && typeof window.requestAnimationFrame === 'function')
            ? window.requestAnimationFrame.bind(window)
            : (cb) => setTimeout(cb, 50);
        const tick = () => {
            if (!this._skipToEndActive) return;
            if (this._checkForwardPassComplete()) {
                this._finalizeSkipToEnd();
                return;
            }
            this._skipToEndRaf = schedule(tick);
        };
        this._skipToEndRaf = schedule(tick);
    }

    _finalizeSkipToEnd() {
        this._skipToEndActive = false;
        if (this._skipToEndRaf) {
            if (typeof window !== 'undefined' && typeof window.cancelAnimationFrame === 'function') {
                window.cancelAnimationFrame(this._skipToEndRaf);
            } else {
                clearTimeout(this._skipToEndRaf);
            }
            this._skipToEndRaf = null;
        }

        const restore = this._skipToEndRestore;
        if (restore) {
            if (this._engine && typeof this._engine.setSpeed === 'function') {
                this._engine.setSpeed(restore.engineSpeed);
            }
            setGlobalAnimSpeedMult(restore.globalSpeed);
            setPrismAddAnimSpeedMult(restore.prismAddSpeed);
            setSelfAttentionTimeMult(restore.selfAttentionSpeed);
            this._skipToEndRestore = null;
        }
    }

    _restoreSkipLayerSpeeds() {
        if (!this._skipLayerActive) return;
        const restore = this._skipLayerRestore;
        if (restore) {
            if (this._engine && typeof this._engine.setSpeed === 'function') {
                this._engine.setSpeed(restore.engineSpeed);
            }
            setGlobalAnimSpeedMult(restore.globalSpeed);
            setPrismAddAnimSpeedMult(restore.prismAddSpeed);
            setSelfAttentionTimeMult(restore.selfAttentionSpeed);
        }
        this._skipLayerActive = false;
        this._skipLayerLast = false;
        this._skipLayerRestore = null;
    }

    /**
     * Called when the currently active layer reports completion via its
     * `onFinished` callback.  This creates the next Gpt2Layer, injects the
     * existing lane bundle and registers its own completion hook.
     */
    _advanceToNextLayer() {
        this._currentLayerIdx += 1;
        this.dispatchEvent(new Event('progress'));
        if (this._currentLayerIdx >= this._numLayers) {
            // All layers processed – trigger final rise into top embedding
            try { this._animateRiseIntoTopEmbedding(); } catch (_) { /* optional */ }
            return;
        }

        // Grab lanes from the previous (just-completed) layer
        const prevLayer = this._layers[this._currentLayerIdx - 1];
        // Deactivate the previous layer to stop it updating the shared vectors
        if (prevLayer) {
            prevLayer.isActive = false;
        }
        const externalLanes = prevLayer.lanes;
        // Ensure residual trails remain continuous by reparenting any
        // world-space trails to the new engine scene (safety no-op if same).
        if (externalLanes && externalLanes.length) {
            externalLanes.forEach(lane => {
                if (!lane) return;

                const originalVec = lane.originalVec;
                const originalGroup = originalVec && originalVec.group;

                // Prefer the dedicated world-space residual trail carried across lanes
                const trailRef = lane.originalTrail
                    || (originalVec && originalVec.userData && originalVec.userData.trail);

                if (originalGroup && typeof originalGroup.updateMatrixWorld === 'function') {
                    originalGroup.updateMatrixWorld(true);
                }

                if (trailRef && typeof trailRef.snapLastPointTo === 'function' && originalGroup) {
                    originalGroup.getWorldPosition(TMP_WORLD_POS);
                    trailRef.snapLastPointTo(TMP_WORLD_POS);
                }

                if (trailRef && typeof trailRef.reparent === 'function') {
                    trailRef.reparent(this._engine.scene);
                }
            });
        }

        const nextLayer = this._layers[this._currentLayerIdx];
        if (!nextLayer) return;

        nextLayer.activateWithLanes(externalLanes);

        this._maybeAutoCameraFocus({ immediate: true });

        if (this._skipLayerActive && !this._skipLayerLast) {
            this._restoreSkipLayerSpeeds();
        }

        // Now that the original residual vectors have been transferred, we can safely
        // hide the remaining heavy geometry in the previous layer to save GPU work.
        if (prevLayer && typeof prevLayer.hideDynamicGeometry === 'function') {
            prevLayer.hideDynamicGeometry();
        }
    }

    /**
     * After the last layer completes, raise residual vectors up into the top
     * vocabulary embedding position using the same placement logic as the test page.
     */
    _animateRiseIntoTopEmbedding() {
        const lastLayer = this._layers[this._numLayers - 1];
        if (!lastLayer || !Array.isArray(lastLayer.lanes) || !lastLayer.lanes.length) return;

        if (!lastLayer.mlpDown || !lastLayer.mlpDown.group) return;

        if (this._skipLayerActive && this._skipLayerLast) {
            this._restoreSkipLayerSpeeds();
        }

        const targetYLocal = this._calculateTopEmbeddingTargetY(lastLayer);

        const lnInfo = this._findTopLayerNorm(lastLayer);

        this._animateResidualVectors(lastLayer, targetYLocal, lnInfo);
    }

    /**
     * Determine the target local Y position for residual vectors entering the
     * top vocabulary embedding and update MHSA animation boundaries.
     * @param {Gpt2Layer} lastLayer
     * @returns {number} Local-space Y coordinate where vectors should stop.
     */
    _calculateTopEmbeddingTargetY(lastLayer) {
        let targetYLocal = null;
        try {
            const scene = this._engine && this._engine.scene;
            let topEmbedObj = null;
            if (scene && typeof scene.traverse === 'function') {
                scene.traverse((obj) => {
                    if (topEmbedObj) return;
                    if (obj && obj.userData && obj.userData.label === 'Vocab Embedding (Top)') {
                        topEmbedObj = obj;
                    }
                });
            }
            if (topEmbedObj) {
                const centerWorld = new THREE.Vector3();
                topEmbedObj.getWorldPosition(centerWorld);
                const stopWorldY = centerWorld.y - EMBEDDING_MATRIX_PARAMS_VOCAB.height / 2 + 5;
                const localVec = new THREE.Vector3(0, stopWorldY, 0);
                lastLayer.root.worldToLocal(localVec);
                targetYLocal = localVec.y;
            }
        } catch (_) { /* fallback to formula below */ }

        if (targetYLocal == null) {
            const towerTopYLocal = lastLayer.mlpDown.group.position.y + MLP_MATRIX_PARAMS_DOWN.height / 2;
            const topVocabCenterYLocal = towerTopYLocal + TOP_EMBED_Y_GAP_ABOVE_TOWER + EMBEDDING_MATRIX_PARAMS_VOCAB.height / 2 + TOP_EMBED_Y_ADJUST;
            targetYLocal = topVocabCenterYLocal - EMBEDDING_MATRIX_PARAMS_VOCAB.height / 2 + 5;
        }

        try {
            if (lastLayer.mhsaAnimation) {
                lastLayer.mhsaAnimation.finalOriginalY = targetYLocal;
                lastLayer.mhsaAnimation.topEmbeddingStopY = targetYLocal;
                lastLayer.mhsaAnimation.postSplitRiseSpeed = ANIM_RISE_SPEED_ORIGINAL;
            }
            lastLayer.__topEmbedStopYLocal = targetYLocal;
        } catch (_) { /* no-op */ }

        return targetYLocal;
    }

    /**
     * Locate the optional top LayerNorm in the scene and compute useful
     * positional data for animation.
     * @param {Gpt2Layer} lastLayer
     * @returns {{lnTopGroup: THREE.Object3D, lnCenterY: number, lnBottomY: number}|null}
     *          LayerNorm group and position info if found.
     */
    _findTopLayerNorm(lastLayer) {
        let lnTopGroup = null;
        try {
            const scene = this._engine && this._engine.scene;
            if (scene && typeof scene.traverse === 'function') {
                scene.traverse(obj => {
                    if (!lnTopGroup && obj && obj.userData && obj.userData.label === 'LayerNorm (Top)') {
                        lnTopGroup = obj;
                    }
                });
            }
        } catch (_) { /* optional */ }

        if (!lnTopGroup) return null;

        const lnCenterWorld = new THREE.Vector3();
        lnTopGroup.getWorldPosition(lnCenterWorld);
        const lnCenterLocal = lnCenterWorld.clone();
        lastLayer.root.worldToLocal(lnCenterLocal);
        const lnCenterY = lnCenterLocal.y;
        const lnBottomY = lnCenterY - LN_PARAMS.height / 2;

        return { lnTopGroup, lnCenterY, lnBottomY };
    }

    /**
     * Apply the bright activated appearance to a LayerNorm group.
     * @param {THREE.Object3D} lnTopGroup
     */
    _activateLayerNormColor(lnTopGroup) {
        const white = new THREE.Color(0xffffff);
        lnTopGroup.traverse(obj => {
            if (obj.isMesh && obj.material) {
                const apply = mat => { mat.color.copy(white); mat.emissive.copy(white); mat.emissiveIntensity = 0.5; mat.transparent = false; mat.opacity = 1.0; };
                if (Array.isArray(obj.material)) obj.material.forEach(apply); else apply(obj.material);
            }
        });
    }

    /**
     * Animate residual vectors toward the vocab embedding, optionally passing
     * through the top LayerNorm pipeline when present.
     * @param {Gpt2Layer} lastLayer
     * @param {number} targetYLocal
     * @param {{lnTopGroup: THREE.Object3D, lnCenterY: number, lnBottomY: number}|null} lnInfo
     */
    _animateResidualVectors(lastLayer, targetYLocal, lnInfo) {
        if (lnInfo && lnInfo.lnTopGroup) {
            if (lastLayer.mhsaAnimation) {
                lastLayer.mhsaAnimation.suppressResidualRise = true;
            }
            const { lnTopGroup, lnCenterY, lnBottomY } = lnInfo;
            const lnMeshes = [];
            lnTopGroup.traverse(obj => {
                if (obj && obj.isMesh && obj.material) {
                    lnMeshes.push(obj);
                }
            });

            const lnColorState = {
                highestY: -Infinity,
                locked: false,
                lockedColor: new THREE.Color(COLOR_BRIGHT_YELLOW),
                currentColor: new THREE.Color(COLOR_DARK_GRAY),
                currentOpacity: 1.0
            };
            const tempColor = new THREE.Color();
            const applyTopLnColor = () => {
                lnMeshes.forEach(mesh => {
                    const applyMaterial = mat => {
                        if (!mat) return;
                        if (mat.color) mat.color.copy(lnColorState.currentColor);
                        if (mat.emissive) mat.emissive.copy(lnColorState.currentColor);
                        mat.transparent = lnColorState.currentOpacity < 1.0;
                        mat.opacity = lnColorState.currentOpacity;
                        mat.needsUpdate = true;
                    };
                    if (Array.isArray(mesh.material)) {
                        mesh.material.forEach(applyMaterial);
                    } else {
                        applyMaterial(mesh.material);
                    }
                });
            };
            applyTopLnColor();

            const lnHeight = LN_PARAMS.height || 0;
            const lnTopY = lnCenterY + lnHeight / 2;
            const normStartY = lnBottomY + lnHeight * LN_NORM_START_FRACTION_FROM_BOTTOM;
            const exitTransitionRange = 5;
            const tmpWorldPos = new THREE.Vector3();
            const tmpLocalPos = new THREE.Vector3();

            const updateTopLnColor = (y) => {
                if (!Number.isFinite(y)) return;
                if (y > lnColorState.highestY) {
                    lnColorState.highestY = y;
                }

                const highest = lnColorState.highestY;
                if (lnColorState.locked) {
                    lnColorState.currentColor.copy(lnColorState.lockedColor);
                    lnColorState.currentOpacity = 1.0;
                } else {
                    if (highest >= lnBottomY && highest < lnCenterY) {
                        const denom = Math.max(lnCenterY - lnBottomY, 1e-6);
                        const t = (highest - lnBottomY) / denom;
                        tempColor.copy(COLOR_DARK_GRAY).lerp(COLOR_LIGHT_YELLOW, t);
                        lnColorState.currentColor.copy(tempColor);
                        lnColorState.currentOpacity = THREE.MathUtils.lerp(1.0, 0.6, t);
                    } else if (highest >= lnCenterY && highest < lnTopY) {
                        lnColorState.currentColor.copy(COLOR_LIGHT_YELLOW);
                        lnColorState.currentOpacity = 0.6;
                    } else if (highest >= lnTopY) {
                        const tRaw = (highest - lnTopY) / exitTransitionRange;
                        const t = Math.min(1, Math.max(0, tRaw));
                        tempColor.copy(COLOR_LIGHT_YELLOW).lerp(COLOR_BRIGHT_YELLOW, t);
                        lnColorState.currentColor.copy(tempColor);
                        lnColorState.currentOpacity = THREE.MathUtils.lerp(0.6, 1.0, t);
                    } else {
                        lnColorState.currentColor.copy(COLOR_DARK_GRAY);
                        lnColorState.currentOpacity = 1.0;
                    }

                    if (highest >= lnTopY + exitTransitionRange) {
                        lnColorState.locked = true;
                        lnColorState.lockedColor.copy(COLOR_BRIGHT_YELLOW);
                        lnColorState.currentColor.copy(lnColorState.lockedColor);
                        lnColorState.currentOpacity = 1.0;
                    }
                }

                applyTopLnColor();
            };

            const additionDurationFor = (vec) => {
                const length = vec?.instanceCount || VECTOR_LENGTH_PRISM;
                return (PRISM_ADD_ANIM_BASE_DURATION + PRISM_ADD_ANIM_BASE_FLASH_DURATION + length * PRISM_ADD_ANIM_BASE_DELAY_BETWEEN_PRISMS) / PRISM_ADD_ANIM_SPEED_MULT;
            };

            const updateTrailPosition = (vector) => {
                if (!vector || !vector.userData || !vector.userData.trail) return;
                const trail = vector.userData.trail;
                if (typeof trail.update !== 'function') return;
                vector.group.getWorldPosition(tmpWorldPos);
                if (vector.userData.trailWorld) {
                    tmpLocalPos.copy(tmpWorldPos);
                    trail.update(tmpLocalPos);
                } else {
                    tmpLocalPos.copy(tmpWorldPos);
                    try {
                        const parentObject = (trail._line && trail._line.parent) || trail._scene || null;
                        if (parentObject && typeof parentObject.worldToLocal === 'function') {
                            parentObject.worldToLocal(tmpLocalPos);
                        }
                    } catch (_) {
                        // fall back to world position already copied into tmpLocalPos
                    }
                    trail.update(tmpLocalPos);
                }
            };

            lastLayer.lanes.forEach(lane => {
                const vec = lane && lane.originalVec;
                if (!vec || !vec.group) return;
                if (lane && lane.originalTrail) {
                    vec.userData = vec.userData || {};
                    if (!vec.userData.trail) {
                        vec.userData.trail = lane.originalTrail;
                        vec.userData.trailWorld = true;
                    }
                }
                if (lane && !lane.__topLnStopRise) {
                    lane.__topLnStopRise = true;
                    lane.stopRise = true;
                    delete lane.stopRiseTarget;
                }
                const startY = vec.group.position.y;
                if (!Number.isFinite(startY)) return;

                const zPos = lane.zPos || 0;
                const multVec = new VectorVisualizationInstancedPrism(
                    vec.rawData.slice(),
                    new THREE.Vector3(0, lnCenterY, zPos),
                    30,
                    vec.instanceCount
                );
                lastLayer.root.add(multVec.group);
                multVec.group.visible = false;

                const addVec = new VectorVisualizationInstancedPrism(
                    vec.rawData.slice(),
                    new THREE.Vector3(0, lnCenterY + LN_PARAMS.height / 4, zPos),
                    30,
                    vec.instanceCount
                );
                lastLayer.root.add(addVec.group);
                addVec.group.visible = false;

                const normAnim = new PrismLayerNormAnimation(vec);
                let normLoopActive = false;

                if (startY >= lnBottomY) {
                    multVec.group.visible = true;
                    addVec.group.visible = true;
                    lane.__topLnEntered = true;
                }

                const startFinalRise = (resVec) => {
                    const riseDist = Math.max(0, targetYLocal - resVec.group.position.y);
                    const durMs = (riseDist / (ANIM_RISE_SPEED_ORIGINAL * GLOBAL_ANIM_SPEED_MULT)) * 1000;
                    new TWEEN.Tween(resVec.group.position)
                        .to({ y: targetYLocal }, Math.max(100, durMs))
                        .easing(TWEEN.Easing.Quadratic.InOut)
                        .onUpdate(() => {
                            updateTopLnColor(resVec.group.position.y);
                            updateTrailPosition(resVec);
                            this.dispatchEvent(new Event('progress'));
                        })
                        .onComplete(() => {
                            updateTopLnColor(targetYLocal + exitTransitionRange);
                            updateTrailPosition(resVec);
                            if (lane && lane.__topLnStopRise) {
                                delete lane.stopRise;
                                delete lane.stopRiseTarget;
                                delete lane.__topLnStopRise;
                            }
                            this.dispatchEvent(new Event('progress'));
                        })
                        .start();
                };

                const beginMultiply = () => {
                    if (lane.__topLnMultStarted) return;
                    lane.__topLnMultStarted = true;
                    multVec.group.visible = true;
                    addVec.group.visible = true;

                    simplePrismMultiply(vec, multVec, () => {
                        updateTopLnColor(multVec.group.position.y);
                        vec.group.visible = false;
                        multVec.group.visible = false;

                        const resVec = new VectorVisualizationInstancedPrism(
                            multVec.rawData.slice(),
                            multVec.group.position.clone(),
                            30,
                            multVec.instanceCount
                        );
                        lastLayer.root.add(resVec.group);

                        if (multVec.group && multVec.group.parent) {
                            multVec.group.parent.remove(multVec.group);
                        }

                        resVec.userData = resVec.userData || {};
                        if (vec.userData && vec.userData.trail) {
                            resVec.userData.trail = vec.userData.trail;
                            resVec.userData.trailWorld = vec.userData.trailWorld;
                        }
                        lane.originalVec = resVec;

                        updateTopLnColor(resVec.group.position.y);
                        this.dispatchEvent(new Event('progress'));

                        lane.__topLnShiftStarted = true;
                        startPrismAdditionAnimation(resVec, addVec, null, () => {
                            lane.__topLnShiftComplete = true;
                            const additionTrail = resVec.userData && resVec.userData.trail;
                            if (additionTrail) {
                                addVec.userData = addVec.userData || {};
                                const additionTrailIsWorld = Boolean(resVec.userData && resVec.userData.trailWorld);
                                const prevTrail = addVec.userData.trail;
                                if (prevTrail && prevTrail !== additionTrail) {
                                    try {
                                        if (typeof prevTrail.dispose === 'function') {
                                            prevTrail.dispose();
                                        } else if (prevTrail._line && prevTrail._line.parent) {
                                            prevTrail._line.parent.remove(prevTrail._line);
                                        }
                                    } catch (_) { /* optional cleanup */ }
                                }
                                addVec.userData.trail = additionTrail;
                                addVec.userData.trailWorld = additionTrailIsWorld;
                                if (additionTrailIsWorld) {
                                    addVec.group.getWorldPosition(TMP_WORLD_POS);
                                    if (typeof additionTrail.snapLastPointTo === 'function') {
                                        additionTrail.snapLastPointTo(TMP_WORLD_POS);
                                    } else {
                                        additionTrail.update(TMP_WORLD_POS);
                                    }
                                } else {
                                    if (typeof additionTrail.snapLastPointTo === 'function') {
                                        additionTrail.snapLastPointTo(addVec.group.position);
                                    } else {
                                        additionTrail.update(addVec.group.position);
                                    }
                                }
                                delete resVec.userData.trail;
                                delete resVec.userData.trailWorld;
                            }
                            if (resVec.group && resVec.group.parent) {
                                resVec.group.parent.remove(resVec.group);
                            }
                            lane.originalVec = addVec;
                            startFinalRise(addVec);
                        }, { suppressResidualTrailUpdates: true });

                        const additionDuration = additionDurationFor(resVec);
                        new TWEEN.Tween({ t: 0 })
                            .to({ t: 1 }, additionDuration)
                            .onUpdate(() => {
                                updateTopLnColor(resVec.group.position.y);
                                this.dispatchEvent(new Event('progress'));
                            })
                            .start();
                    });
                };

                const riseToCenter = () => {
                    const targetY = Math.max(vec.group.position.y, lnCenterY);
                    if (vec.group.position.y >= targetY - 0.01) {
                        vec.group.position.y = targetY;
                        updateTopLnColor(vec.group.position.y);
                        updateTrailPosition(vec);
                        this.dispatchEvent(new Event('progress'));
                        beginMultiply();
                        return;
                    }

                    const distance = Math.max(0, targetY - vec.group.position.y);
                    const duration = (distance / (ANIM_RISE_SPEED_ORIGINAL * GLOBAL_ANIM_SPEED_MULT)) * 1000;
                    new TWEEN.Tween(vec.group.position)
                        .to({ y: targetY }, Math.max(100, duration))
                        .easing(TWEEN.Easing.Quadratic.InOut)
                        .onUpdate(() => {
                            updateTopLnColor(vec.group.position.y);
                            updateTrailPosition(vec);
                            this.dispatchEvent(new Event('progress'));
                        })
                        .onComplete(() => {
                            updateTopLnColor(vec.group.position.y);
                            updateTrailPosition(vec);
                            this.dispatchEvent(new Event('progress'));
                            beginMultiply();
                        })
                        .start();
                };

                const startNormalization = () => {
                    if (normLoopActive) return;
                    if (this._skipToEndActive) {
                        normLoopActive = true;
                        riseToCenter();
                        return;
                    }
                    normLoopActive = true;
                    try {
                        normAnim.start(vec.rawData.slice());
                    } catch (_) {
                        normLoopActive = false;
                        riseToCenter();
                        return;
                    }

                    const runLoop = () => {
                        if (this._skipToEndActive) {
                            normAnim.isAnimating = false;
                            normLoopActive = false;
                            riseToCenter();
                            return;
                        }
                        normAnim.update(0);
                        updateTopLnColor(vec.group.position.y);
                        this.dispatchEvent(new Event('progress'));
                        if (normAnim.isAnimating) {
                            requestAnimationFrame(runLoop);
                        } else {
                            normLoopActive = false;
                            riseToCenter();
                        }
                    };
                    runLoop();
                };

                const moveToNormStart = () => {
                    updateTopLnColor(vec.group.position.y);
                    const stageTarget = Math.max(vec.group.position.y, normStartY);

                    if (vec.group.position.y >= stageTarget - 0.01) {
                        vec.group.position.y = stageTarget;
                        updateTopLnColor(vec.group.position.y);
                        updateTrailPosition(vec);
                        this.dispatchEvent(new Event('progress'));
                        startNormalization();
                        return;
                    }

                    const distance = Math.max(0, stageTarget - vec.group.position.y);
                    const duration = (distance / (ANIM_RISE_SPEED_ORIGINAL * GLOBAL_ANIM_SPEED_MULT)) * 1000;
                    new TWEEN.Tween(vec.group.position)
                        .to({ y: stageTarget }, Math.max(100, duration))
                        .easing(TWEEN.Easing.Quadratic.InOut)
                        .onUpdate(() => {
                            updateTopLnColor(vec.group.position.y);
                            updateTrailPosition(vec);
                            if (!lane.__topLnEntered && vec.group.position.y >= lnBottomY) {
                                lane.__topLnEntered = true;
                                multVec.group.visible = true;
                                addVec.group.visible = true;
                            }
                            this.dispatchEvent(new Event('progress'));
                        })
                        .onComplete(() => {
                            updateTopLnColor(vec.group.position.y);
                            updateTrailPosition(vec);
                            this.dispatchEvent(new Event('progress'));
                            startNormalization();
                        })
                        .start();
                };

                moveToNormStart();
            });

            return;
        }

        lastLayer.lanes.forEach(lane => {
            const vec = lane && lane.originalVec;
            if (!vec || !vec.group) return;
            const startY = vec.group.position.y;
            if (typeof startY !== 'number' || !isFinite(startY)) return;
            if (startY >= targetYLocal - 0.01) return;

            const riseDist = Math.max(0, targetYLocal - startY);
            const durMs = (riseDist / (ANIM_RISE_SPEED_ORIGINAL * GLOBAL_ANIM_SPEED_MULT)) * 1000;

            new TWEEN.Tween(vec.group.position)
                .to({ y: targetYLocal }, Math.max(100, durMs))
                .easing(TWEEN.Easing.Quadratic.InOut)
                .onUpdate(() => this.dispatchEvent(new Event('progress')))
                .onComplete(() => this.dispatchEvent(new Event('progress')))
                .start();
        });
    }

    _resolveActiveLanePosition(targetVec = null) {
        const layers = this._layers;
        if (!Array.isArray(layers) || layers.length === 0) {
            return { laneIndex: -1, laneCount: 0 };
        }

        const layerIndex = Math.min(this._currentLayerIdx, layers.length - 1);
        const layer = layers[layerIndex];
        const lanes = Array.isArray(layer?.lanes) ? layer.lanes : [];
        const laneCount = lanes.length;
        if (!laneCount) {
            return { laneIndex: -1, laneCount };
        }

        const laneIndex = Math.min(laneCount - 1, Math.floor(laneCount / 2));
        const lane = lanes[laneIndex];
        const vec = lane?.originalVec;
        const vecGroup = vec?.group;
        if (!vecGroup || typeof vecGroup.getWorldPosition !== 'function') {
            return { laneIndex: -1, laneCount };
        }

        if (targetVec) {
            if (lane?.stopRise && lane?.stopRiseTarget) {
                const sourceCenter = TMP_CENTER_A;
                const targetCenter = TMP_CENTER_B;
                const sourceOk = this._getVectorWorldCenter(vec, sourceCenter);
                const targetVecRef = this._findVectorByGroup(lane, lane.stopRiseTarget);
                const targetOk = targetVecRef
                    ? this._getVectorWorldCenter(targetVecRef, targetCenter)
                    : this._getGroupWorldPosition(lane.stopRiseTarget, targetCenter);

                if (sourceOk && targetOk) {
                    const hiddenThreshold = HIDE_INSTANCE_Y_OFFSET * 0.2;
                    if (sourceCenter.y < hiddenThreshold) {
                        const lastSource = lane.__followLastSourceVec || (lane.__followLastSourceVec = new THREE.Vector3());
                        if (!lane.__followLastSourceValid) {
                            lastSource.copy(targetCenter);
                            lane.__followLastSourceValid = true;
                        }
                        const progress = Math.min(1, (lane.__followHiddenProgress || 0) + 0.12);
                        lane.__followHiddenProgress = progress;
                        targetVec.lerpVectors(lastSource, targetCenter, progress);
                    } else {
                        lane.__followHiddenProgress = 0;
                        const lastSource = lane.__followLastSourceVec || (lane.__followLastSourceVec = new THREE.Vector3());
                        lastSource.copy(sourceCenter);
                        lane.__followLastSourceValid = true;
                        targetVec.copy(sourceCenter);
                    }
                } else if (sourceOk) {
                    lane.__followHiddenProgress = 0;
                    targetVec.copy(sourceCenter);
                } else if (targetOk) {
                    targetVec.copy(targetCenter);
                } else {
                    vecGroup.getWorldPosition(targetVec);
                }
            } else if (!this._getVectorWorldCenter(vec, targetVec)) {
                if (lane) {
                    lane.__followHiddenProgress = 0;
                }
                vecGroup.getWorldPosition(targetVec);
            }
        }

        return { laneIndex, laneCount };
    }

    _getGroupWorldPosition(group, out) {
        if (!group || typeof group.getWorldPosition !== 'function') return false;
        group.getWorldPosition(out);
        return Number.isFinite(out.x) && Number.isFinite(out.y) && Number.isFinite(out.z);
    }

    _getVectorWorldCenter(vec, out) {
        const vecGroup = vec?.group;
        const mesh = vec?.mesh;
        const canSample = mesh && typeof mesh.getMatrixAt === 'function' && vecGroup?.matrixWorld;
        if (!canSample) {
            return this._getGroupWorldPosition(vecGroup, out);
        }
        const hideThreshold = HIDE_INSTANCE_Y_OFFSET * 0.5;
        const rawLen = Array.isArray(vec.rawData) ? vec.rawData.length : Infinity;
        const count = Number.isFinite(vec.instanceCount) ? vec.instanceCount : Infinity;
        const length = Math.max(1, Math.min(rawLen, count));
        const firstIndex = Math.max(0, Math.floor((length - 1) / 2));
        const secondIndex = length % 2 === 0 ? Math.min(length - 1, firstIndex + 1) : firstIndex;
        mesh.getMatrixAt(firstIndex, TMP_CENTER_MAT_A);
        out.setFromMatrixPosition(TMP_CENTER_MAT_A).applyMatrix4(vecGroup.matrixWorld);
        if (!Number.isFinite(out.y) || out.y <= hideThreshold) {
            return this._getGroupWorldPosition(vecGroup, out);
        }
        if (secondIndex !== firstIndex) {
            mesh.getMatrixAt(secondIndex, TMP_CENTER_MAT_B);
            TMP_CENTER_B.setFromMatrixPosition(TMP_CENTER_MAT_B).applyMatrix4(vecGroup.matrixWorld);
            if (!Number.isFinite(TMP_CENTER_B.y) || TMP_CENTER_B.y <= hideThreshold) {
                return this._getGroupWorldPosition(vecGroup, out);
            }
            out.add(TMP_CENTER_B).multiplyScalar(0.5);
        }
        return Number.isFinite(out.x) && Number.isFinite(out.y) && Number.isFinite(out.z);
    }

    _findVectorByGroup(lane, group) {
        if (!lane || !group) return null;
        const candidates = [
            lane.originalVec,
            lane.dupVec,
            lane.travellingVec,
            lane.resultVec,
            lane.postAdditionVec,
            lane.movingVecLN2,
            lane.resultVecLN2,
            lane.finalVecAfterMlp,
            lane.addTarget,
            lane.addTargetLN2,
            lane.multTarget,
            lane.multTargetLN2,
        ];
        for (let i = 0; i < candidates.length; i++) {
            const candidate = candidates[i];
            if (candidate && candidate.group === group) return candidate;
        }
        return null;
    }

    _clearAutoCameraOffsets() {
        this._hasAutoCameraOffsets = false;
        this._autoCameraDesiredCameraOffset.set(0, 0, 0);
        this._autoCameraDesiredTargetOffset.set(0, 0, 0);
        this._autoCameraCurrentCameraOffset.set(0, 0, 0);
        this._autoCameraCurrentTargetOffset.set(0, 0, 0);
        this._autoCameraSmoothValid = false;
    }

    _setAutoCameraOffsets(cameraOffset, targetOffset, { snap = false } = {}) {
        if (cameraOffset) this._autoCameraDesiredCameraOffset.copy(cameraOffset);
        if (targetOffset) this._autoCameraDesiredTargetOffset.copy(targetOffset);
        if (snap || !this._hasAutoCameraOffsets) {
            this._autoCameraCurrentCameraOffset.copy(this._autoCameraDesiredCameraOffset);
            this._autoCameraCurrentTargetOffset.copy(this._autoCameraDesiredTargetOffset);
        }
        this._hasAutoCameraOffsets = true;
    }

    _captureAutoCameraOffsets(existingReference = null) {
        const engine = this._engine;
        const camera = engine?.camera;
        if (!camera) return false;

        const reference = existingReference || this._autoCameraCenter;
        if (!existingReference) {
            const laneInfo = this._resolveActiveLanePosition(reference);
            if (laneInfo.laneIndex < 0) {
                this._clearAutoCameraOffsets();
                return false;
            }
        }

        this._autoCameraDesiredCameraOffset.copy(camera.position).sub(reference);
        if (!Number.isFinite(this._autoCameraDesiredCameraOffset.x)
            || !Number.isFinite(this._autoCameraDesiredCameraOffset.y)
            || !Number.isFinite(this._autoCameraDesiredCameraOffset.z)) {
            this._clearAutoCameraOffsets();
            return false;
        }

        const controls = engine?.controls;
        if (controls && controls.target) {
            this._autoCameraDesiredTargetOffset.copy(controls.target).sub(reference);
        } else {
            this._autoCameraDesiredTargetOffset.set(0, 0, 0);
        }

        this._setAutoCameraOffsets(this._autoCameraDesiredCameraOffset, this._autoCameraDesiredTargetOffset, { snap: true });
        return true;
    }

    _applyAutoCamera(reference) {
        if (!this._autoCameraFollow || !this._hasAutoCameraOffsets) {
            return;
        }

        const engine = this._engine;
        const camera = engine?.camera;
        if (!engine || !camera) return;

        if (!Number.isFinite(reference?.x) || !Number.isFinite(reference?.y) || !Number.isFinite(reference?.z)) {
            return;
        }

        if (!Number.isFinite(this._autoCameraDesiredCameraOffset.x)
            || !Number.isFinite(this._autoCameraDesiredCameraOffset.y)
            || !Number.isFinite(this._autoCameraDesiredCameraOffset.z)) {
            return;
        }

        this._suppressControlsChange = true;
        try {
            const camOffset = this._autoCameraCurrentCameraOffset;
            if (!Number.isFinite(camOffset.x) || !Number.isFinite(camOffset.y) || !Number.isFinite(camOffset.z)) {
                camOffset.copy(this._autoCameraDesiredCameraOffset);
            } else if (this._autoCameraOffsetLerpAlpha > 0) {
                camOffset.lerp(this._autoCameraDesiredCameraOffset, this._autoCameraOffsetLerpAlpha);
            } else {
                camOffset.copy(this._autoCameraDesiredCameraOffset);
            }
            camera.position.copy(reference).add(camOffset);

            const controls = engine.controls;
            if (controls && controls.target) {
                const targetOffset = this._autoCameraCurrentTargetOffset;
                if (!Number.isFinite(targetOffset.x) || !Number.isFinite(targetOffset.y) || !Number.isFinite(targetOffset.z)) {
                    targetOffset.copy(this._autoCameraDesiredTargetOffset);
                } else if (this._autoCameraOffsetLerpAlpha > 0) {
                    targetOffset.lerp(this._autoCameraDesiredTargetOffset, this._autoCameraOffsetLerpAlpha);
                } else {
                    targetOffset.copy(this._autoCameraDesiredTargetOffset);
                }
                controls.target.copy(reference).add(targetOffset);
                if (typeof controls.update === 'function') {
                    controls.update();
                }
            }

            if (typeof engine.notifyCameraUpdated === 'function') {
                engine.notifyCameraUpdated();
            }
        } finally {
            this._suppressControlsChange = false;
        }
    }

    _updateCameraOffsetOverlay() {
        const overlay = this._cameraOffsetDiv;
        if (!overlay || !this._devMode) {
            if (overlay) overlay.style.display = 'none';
            return;
        }

        if (!this._autoCameraFollow) {
            overlay.style.display = 'none';
            return;
        }

        const engine = this._engine;
        const camera = engine?.camera;
        if (!camera) {
            overlay.style.display = 'block';
            overlay.textContent = 'Offset vs Residual Lane —\nΔx: —\nΔy: —\nΔz: —';
            this._clearAutoCameraOffsets();
            return;
        }

        const reference = this._autoCameraCenter;
        const { laneIndex } = this._resolveActiveLanePosition(reference);
        if (laneIndex < 0 || !Number.isFinite(reference.x) || !Number.isFinite(reference.y) || !Number.isFinite(reference.z)) {
            overlay.style.display = 'block';
            overlay.textContent = 'Offset vs Residual Lane —\nΔx: —\nΔy: —\nΔz: —';
            this._clearAutoCameraOffsets();
            return;
        }

        const offset = this._autoCameraOffsetScratch;
        offset.copy(camera.position).sub(reference);
        const format = (value) => (Number.isFinite(value) ? value.toFixed(2) : '—');
        const laneLabel = Number.isInteger(laneIndex) && laneIndex >= 0 ? (laneIndex + 1) : '—';

        overlay.style.display = 'block';
        overlay.textContent = `Offset vs Residual Lane ${laneLabel}\nΔx: ${format(offset.x)}\nΔy: ${format(offset.y)}\nΔz: ${format(offset.z)}`;
    }

    _resolveAutoCameraViewKey() {
        const layers = this._layers;
        if (!Array.isArray(layers) || !layers.length) return 'default';
        const layerIndex = Math.min(this._currentLayerIdx, layers.length - 1);
        const layer = layers[layerIndex];
        const mhsa = layer?.mhsaAnimation;
        const lanes = Array.isArray(layer?.lanes) ? layer.lanes : [];
        const laneCount = lanes.length;
        const laneIndex = laneCount ? Math.min(laneCount - 1, Math.floor(laneCount / 2)) : -1;
        const lane = laneIndex >= 0 ? lanes[laneIndex] : null;
        const inLn = !!(lane && (lane.horizPhase === 'insideLN' || lane.ln2Phase === 'insideLN'));
        this._autoCameraViewContext = { lane, laneIndex, laneCount, inLn };
        const passPhase = mhsa?.mhaPassThroughPhase || 'positioning_mha_vectors';
        const inTravel = !!(lane && lane.horizPhase === 'travelMHSA'
            && passPhase === 'positioning_mha_vectors');
        const inCopyStage = !!(lane && lane.horizPhase === 'finishedHeads'
            && (passPhase === 'positioning_mha_vectors' || passPhase === 'ready_for_parallel_pass_through'));
        if (!mhsa) {
            if (inLn) return 'ln';
            return 'default';
        }

        if (inLn) {
            return 'ln';
        }
        if (inTravel || inCopyStage) {
            return 'travel';
        }

        const outputPhase = mhsa.outputProjMatrixAnimationPhase || 'waiting';
        const rowPhase = mhsa.rowMergePhase || 'not_started';
        const outputReturnComplete = mhsa.outputProjMatrixReturnComplete === true;
        const concatActive = (rowPhase === 'merging' || rowPhase === 'merged')
            && outputPhase === 'waiting'
            && !outputReturnComplete;
        if (concatActive) {
            return 'concat';
        }

        const mhsaGate = rowPhase === 'not_started' && outputPhase === 'waiting';
        const mhsaActive = mhsaGate && (passPhase === 'parallel_pass_through_active'
            || passPhase === 'mha_pass_through_complete'
            || (passPhase === 'ready_for_parallel_pass_through' && !inCopyStage));
        if (mhsaActive) {
            return 'mhsa';
        }
        return 'default';
    }

    _applyAutoCameraViewOffsets() {
        this._updateAutoCameraScaledOffsets();
        const key = this._resolveAutoCameraViewKey();
        const viewContext = this._autoCameraViewContext;
        let camOffset = this._autoCameraDefaultCameraOffset;
        let targetOffset = this._autoCameraDefaultTargetOffset;

        if (key === 'ln' && this._autoCameraLnOffsetsEnabled) {
            camOffset = this._autoCameraLnCameraOffset;
            targetOffset = this._autoCameraLnTargetOffset;
        } else if (key === 'travel' && this._autoCameraTravelOffsetsEnabled) {
            camOffset = this._autoCameraTravelCameraOffset;
            targetOffset = this._autoCameraTravelTargetOffset;
        } else if (key === 'mhsa' && this._autoCameraMhsaOffsetsEnabled) {
            camOffset = this._autoCameraMhsaCameraOffset;
            targetOffset = this._autoCameraMhsaTargetOffset;
        } else if (key === 'concat' && this._autoCameraConcatOffsetsEnabled) {
            camOffset = this._autoCameraConcatCameraOffset;
            targetOffset = this._autoCameraConcatTargetOffset;
        }

        if (key !== this._autoCameraViewKey) {
            const priorKey = this._autoCameraViewKey;
            const isMlpTransition = priorKey === 'ln'
                && key === 'default'
                && viewContext
                && viewContext.lane
                && viewContext.lane.ln2Phase
                && viewContext.lane.ln2Phase !== 'insideLN';
            this._autoCameraViewBlendAlphaActive = isMlpTransition
                ? this._autoCameraViewBlendAlphaMlpReturn
                : this._autoCameraViewBlendAlpha;
            this._autoCameraViewKey = key;
            if (!this._autoCameraSmoothValid) {
                this._autoCameraViewBlendT = 1;
                this._autoCameraViewFromCameraOffset.copy(camOffset);
                this._autoCameraViewFromTargetOffset.copy(targetOffset);
            } else {
                this._autoCameraViewBlendT = 0;
                this._autoCameraViewFromCameraOffset.copy(this._autoCameraCurrentCameraOffset);
                this._autoCameraViewFromTargetOffset.copy(this._autoCameraCurrentTargetOffset);
            }
        }

        this._autoCameraViewToCameraOffset.copy(camOffset);
        this._autoCameraViewToTargetOffset.copy(targetOffset);

        if (this._autoCameraViewBlendT < 1) {
            this._autoCameraViewBlendT = Math.min(1, this._autoCameraViewBlendT + this._autoCameraViewBlendAlphaActive);
            const t = this._autoCameraViewBlendT;
            this._autoCameraViewBlendCameraOffset.copy(this._autoCameraViewFromCameraOffset).lerp(this._autoCameraViewToCameraOffset, t);
            this._autoCameraViewBlendTargetOffset.copy(this._autoCameraViewFromTargetOffset).lerp(this._autoCameraViewToTargetOffset, t);
            camOffset = this._autoCameraViewBlendCameraOffset;
            targetOffset = this._autoCameraViewBlendTargetOffset;
        }

        this._setAutoCameraOffsets(camOffset, targetOffset, { snap: false });
    }

    _computeAutoCameraScale() {
        if (typeof window === 'undefined') return 1.0;
        const width = window.innerWidth || 0;
        if (!Number.isFinite(width) || width <= 0) return 1.0;
        if (width >= this._autoCameraScaleMaxWidth) return 1.0;
        if (width <= this._autoCameraScaleMinWidth) return this._autoCameraScaleMax;
        const t = (this._autoCameraScaleMaxWidth - width)
            / (this._autoCameraScaleMaxWidth - this._autoCameraScaleMinWidth);
        return 1.0 + t * (this._autoCameraScaleMax - 1.0);
    }

    _updateAutoCameraScaledOffsets(force = false) {
        const scale = this._computeAutoCameraScale();
        const shiftFactor = (this._autoCameraScaleMax > 1.0)
            ? (scale - 1.0) / Math.max(0.0001, this._autoCameraScaleMax - 1.0)
            : 0;
        const shiftX = Math.abs(this._autoCameraShiftMaxX) > 0.0001
            ? this._autoCameraShiftMaxX * shiftFactor
            : 0;
        const shiftTravelX = Math.abs(this._autoCameraShiftMaxTravelX) > 0.0001
            ? this._autoCameraShiftMaxTravelX * shiftFactor
            : 0;
        const shiftMhsaX = Math.abs(this._autoCameraShiftMaxMhsaX) > 0.0001
            ? this._autoCameraShiftMaxMhsaX * shiftFactor
            : 0;
        if (!force
            && Math.abs(scale - this._autoCameraScaleLast) < 0.001
            && Math.abs(shiftX - this._autoCameraShiftLastX) < 0.5) {
            return;
        }
        this._autoCameraScaleLast = scale;
        this._autoCameraShiftLastX = shiftX;

        const applyScaleShift = (dest, base, extraShiftX = 0) => {
            dest.copy(base).multiplyScalar(scale);
            if (shiftX || extraShiftX) dest.x += shiftX + extraShiftX;
        };

        applyScaleShift(this._autoCameraDefaultCameraOffset, this._autoCameraDefaultCameraOffsetBase);
        applyScaleShift(this._autoCameraDefaultTargetOffset, this._autoCameraDefaultTargetOffsetBase);

        if (this._autoCameraMhsaOffsetsEnabled) {
            applyScaleShift(this._autoCameraMhsaCameraOffset, this._autoCameraMhsaCameraOffsetBase, shiftMhsaX);
            applyScaleShift(this._autoCameraMhsaTargetOffset, this._autoCameraMhsaTargetOffsetBase, shiftMhsaX);
        }
        if (this._autoCameraConcatOffsetsEnabled) {
            applyScaleShift(this._autoCameraConcatCameraOffset, this._autoCameraConcatCameraOffsetBase);
            applyScaleShift(this._autoCameraConcatTargetOffset, this._autoCameraConcatTargetOffsetBase);
        }
        if (this._autoCameraLnOffsetsEnabled) {
            applyScaleShift(this._autoCameraLnCameraOffset, this._autoCameraLnCameraOffsetBase);
            applyScaleShift(this._autoCameraLnTargetOffset, this._autoCameraLnTargetOffsetBase);
        }
        if (this._autoCameraTravelOffsetsEnabled) {
            applyScaleShift(this._autoCameraTravelCameraOffset, this._autoCameraTravelCameraOffsetBase, shiftTravelX);
            applyScaleShift(this._autoCameraTravelTargetOffset, this._autoCameraTravelTargetOffsetBase, shiftTravelX);
            if (this._autoCameraTravelMobileOverrideEnabled && shiftFactor > 0.0001) {
                this._autoCameraTravelCameraOffset.lerp(
                    this._autoCameraTravelMobileOverrideCameraOffset,
                    shiftFactor
                );
                this._autoCameraTravelTargetOffset.lerp(
                    this._autoCameraTravelMobileOverrideTargetOffset,
                    shiftFactor
                );
            }
        }
    }

    _updateAutoCameraFollow() {
        if (!this._autoCameraFollow) return false;
        const engine = this._engine;
        const camera = engine?.camera;
        if (!camera) return false;

        this._applyAutoCameraViewOffsets();

        const reference = this._autoCameraCenter;
        const { laneIndex } = this._resolveActiveLanePosition(reference);
        if (laneIndex < 0 || !Number.isFinite(reference.x) || !Number.isFinite(reference.y) || !Number.isFinite(reference.z)) {
            this._clearAutoCameraOffsets();
            return false;
        }

        let followRef = reference;
        if (this._autoCameraSmoothAlpha > 0) {
            if (!this._autoCameraSmoothValid) {
                this._autoCameraSmoothedRef.copy(reference);
                this._autoCameraSmoothValid = true;
            } else {
                this._autoCameraSmoothedRef.lerp(reference, this._autoCameraSmoothAlpha);
            }
            followRef = this._autoCameraSmoothedRef;
        }

        if (!this._hasAutoCameraOffsets) {
            this._captureAutoCameraOffsets(followRef);
        }

        this._applyAutoCamera(followRef);
        return true;
    }

    _maybeAutoCameraFocus({ immediate = false } = {}) {
        if (!this._autoCameraFollow && !immediate) {
            this._updateCameraOffsetOverlay();
            return;
        }
        this._updateAutoCameraFollow();
        this._updateCameraOffsetOverlay();
    }

    _startCameraOverlayLoop() {
        if (this._cameraOverlayRaf !== null) return;
        if (typeof requestAnimationFrame !== 'function') return;

        const tick = () => {
            if (!this._autoCameraFollow) {
                this._stopCameraOverlayLoop();
                return;
            }
            this._updateAutoCameraFollow();
            this._updateCameraOffsetOverlay();
            this._cameraOverlayRaf = requestAnimationFrame(tick);
        };

        this._cameraOverlayRaf = requestAnimationFrame(tick);
    }

    _stopCameraOverlayLoop() {
        if (this._cameraOverlayRaf !== null && typeof cancelAnimationFrame === 'function') {
            cancelAnimationFrame(this._cameraOverlayRaf);
        }
        this._cameraOverlayRaf = null;
    }

    setDevMode(enabled) {
        this._devMode = !!enabled;
        if (!this._devMode && this._cameraOffsetDiv) {
            this._cameraOffsetDiv.style.display = 'none';
        } else {
            this._updateCameraOffsetOverlay();
        }
    }
}

/** Convenience helper mirroring CoreEngine.startEngine signature */
export function startPipeline(canvas, numLayers = 12, opts = {}) {
    const pipeline = new LayerPipeline(canvas, numLayers, opts);
    return () => pipeline.dispose();
} 
