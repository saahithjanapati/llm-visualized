import * as THREE from 'three';
import { CoreEngine } from './CoreEngine.js';
import { AutoCameraController } from './AutoCameraController.js';
import Gpt2Layer from './layers/Gpt2Layer.js';
import { createRandomSource } from '../data/RandomActivationSource.js';
import { getLayerNormParamData } from '../data/layerNormParams.js';
import {
    MLP_MATRIX_PARAMS_DOWN,
    EMBEDDING_MATRIX_PARAMS_VOCAB,
    TOP_EMBED_Y_GAP_ABOVE_TOWER,
    TOP_EMBED_Y_ADJUST,
    TOP_EMBED_MAX_RISE_FRACTION,
    GLOBAL_ANIM_SPEED_MULT,
    SELF_ATTENTION_TIME_MULT,
    ANIM_RISE_SPEED_ORIGINAL,
    LN_PARAMS,
    LN_NORM_START_FRACTION_FROM_BOTTOM,
    PRISM_ADD_ANIM_BASE_DURATION,
    PRISM_ADD_ANIM_BASE_FLASH_DURATION,
    PRISM_ADD_ANIM_BASE_DELAY_BETWEEN_PRISMS,
    PRISM_ADD_ANIM_SPEED_MULT,
    setGlobalAnimSpeedMult,
    setPrismAddAnimSpeedMult,
    setSelfAttentionTimeMult,
    SKIP_TRAIL_MAX_STEP_DISTANCE,
    SKIP_COMPONENT_COLOR_LERP_ALPHA,
    VECTOR_LENGTH_PRISM,
    NUM_VECTOR_LANES,
    LAYER_STACK_SPACING_Y
} from '../utils/constants.js';
import { VectorVisualizationInstancedPrism } from '../components/VectorVisualizationInstancedPrism.js';
import { startPrismAdditionAnimation } from '../utils/additionUtils.js';
import { PrismLayerNormAnimation } from '../animations/PrismLayerNormAnimation.js';
import { setGlobalTrailMaxStepDistance, clearTrailsFromScene } from '../utils/trailUtils.js';

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
const LN_PARAM_MONOCHROME = {
    type: 'monochromatic',
    baseHue: 0,
    saturation: 0,
    minLightness: 0.03,
    maxLightness: 0.88,
    useData: true,
    valueMin: -1.8,
    valueMax: 1.8
};
const FINAL_LN_INPUT_LIGHT_BLUE = {
    type: 'hueRange',
    baseHue: 0.56,
    hueSpread: 0.12,
    saturation: 0.92,
    minLightness: 0.42,
    maxLightness: 0.84,
    valueMin: -2,
    valueMax: 2,
    valueClampMin: -2,
    valueClampMax: 2
};

const SKIP_SPEED_RAMP_IN_MS = 60;
const SKIP_SPEED_RAMP_OUT_MS = 95;
const DEFAULT_SKIP_SPEED_PROFILE = Object.freeze({
    engineSpeed: 6.4,
    globalSpeed: 900,
    prismAddSpeed: 230,
    selfAttentionSpeed: 0.03
});

const TMP_WORLD_POS = new THREE.Vector3();

function isArrayLike(values) {
    return Array.isArray(values) || ArrayBuffer.isView(values);
}

function toMutableArray(values, targetLength, fallbackValue = 0) {
    const length = Math.max(1, Math.floor(targetLength || 1));
    if (!isArrayLike(values) || values.length === 0) {
        return new Array(length).fill(fallbackValue);
    }
    const source = Array.isArray(values) ? values : Array.from(values);
    if (source.length === length) return source.slice();
    if (source.length > length) return source.slice(0, length);
    const out = source.slice();
    const padValue = Number.isFinite(source[source.length - 1]) ? source[source.length - 1] : fallbackValue;
    while (out.length < length) out.push(padValue);
    return out;
}

function recolorVectorFromData(vec, values, colorOptions = null) {
    if (!vec || !isArrayLike(values) || values.length === 0) return;
    vec.rawData = Array.isArray(values) ? values.slice() : Array.from(values);
    const numKeyColors = Math.min(30, Math.max(1, vec.rawData.length || 1));
    vec.updateKeyColorsFromData(vec.rawData, numKeyColors, colorOptions, values);
}

function buildElementwiseSum(lhs, rhs, targetLength) {
    const length = Math.max(1, Math.floor(targetLength || 1));
    const out = new Array(length);
    for (let i = 0; i < length; i++) {
        const left = isArrayLike(lhs) ? (lhs[i] || 0) : 0;
        const right = isArrayLike(rhs) ? (rhs[i] || 0) : 0;
        out[i] = left + right;
    }
    return out;
}

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
        this._skipSpeedRampRaf = null;
        this._postResetTrailPurgeRaf = null;
        this._postResetTrailPurgeUntil = 0;
        this._trailPassId = 1;
        this._topLnParamPlaceholders = null;

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
        if (this._engine && this._engine.scene && this._engine.scene.userData) {
            this._engine.scene.userData.trailPassId = this._trailPassId;
        }
        this._autoCamera = new AutoCameraController({
            pipeline: this,
            engine: this._engine,
            opts
        });
        // Keep layer spacing consistent across devices; auto-camera scaling is
        // for view offsets only and shouldn't stretch the stack.
        const layerStackSpacing = LAYER_STACK_SPACING_Y;
        this._initLayers(layerStackSpacing);
        this._initAutoCameraDriver();

        this._autoCamera?.maybeFocus({ immediate: true });
    }


    _initLayers(layerStackSpacing) {
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
                this._laneCount,
                layerStackSpacing
            );

            // Assign onFinished callback for chaining once layer becomes active
            layer.setOnFinished(() => this._advanceToNextLayer());
            layer.setProgressEmitter(this);

            layer.init(this._engine.scene);
            if (layer.root && layer.root.userData) {
                layer.root.userData.trailPassId = this._trailPassId;
            }
            if (typeof this._engine.registerRaycastRoot === 'function') {
                this._engine.registerRaycastRoot(layer.raycastRoot || layer.root);
            }
            this._layers.push(layer);
            this._engine._layers.push(layer); // add to engine update list
        }
    }

    _initAutoCameraDriver() {
        // Drive auto-camera follow from the CoreEngine RAF instead of a
        // secondary requestAnimationFrame loop.
        this._autoCameraDriver = {
            isActive: true,
            update: () => {
                this._autoCamera?.update?.();
            },
            dispose: () => {}
        };
        this._engine._layers.push(this._autoCameraDriver);
    }

    /** Dispose and tear down Three resources */
    dispose() {
        if (this._autoCamera) {
            this._autoCamera.dispose?.();
            this._autoCamera = null;
        }
        if (this._skipToEndRaf) {
            if (typeof window !== 'undefined' && typeof window.cancelAnimationFrame === 'function') {
                window.cancelAnimationFrame(this._skipToEndRaf);
            } else {
                clearTimeout(this._skipToEndRaf);
            }
            this._skipToEndRaf = null;
        }
        if (this._postResetTrailPurgeRaf) {
            if (typeof window !== 'undefined' && typeof window.cancelAnimationFrame === 'function') {
                window.cancelAnimationFrame(this._postResetTrailPurgeRaf);
            } else {
                clearTimeout(this._postResetTrailPurgeRaf);
            }
            this._postResetTrailPurgeRaf = null;
        }
        this._cancelSkipSpeedRamp();

        if (typeof TWEEN !== 'undefined' && typeof TWEEN.removeAll === 'function') {
            try { TWEEN.removeAll(); } catch (_) { /* no-op */ }
        }

        if (this._engine && this._engine.scene) {
            clearTrailsFromScene(this._engine.scene, { includeAllLines: true });
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
        this._autoCamera?.setEnabled?.(enabled, { immediate, resetView, smoothReset });
    }

    /** Check whether automatic camera tracking is enabled. */
    isAutoCameraFollowEnabled() {
        return !!this._autoCamera?.isEnabled?.();
    }

    /** Move the camera to the overview framing (typically the initial tower view). */
    focusOverview({ immediate = true, durationMs = 1400 } = {}) {
        this._autoCamera?.focusOverview?.({ immediate, durationMs });
    }

    /** Apply a horizontal screen-space shift (in pixels) to re-center the view. */
    setScreenShiftPixels(shiftPx, { immediate = false, durationMs = 520 } = {}) {
        this._autoCamera?.setScreenShiftPixels?.(shiftPx, { immediate, durationMs });
    }

    /** Get current follow reference position (residual stream center). */
    getAutoCameraReference() {
        return this._autoCamera?.getReference?.() ?? null;
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

    resetForNewPass({ activationSource = this._activationSource, laneCount = this._laneCount } = {}) {
        const engine = this._engine;

        if (this._skipToEndActive) {
            this._finalizeSkipToEnd({ immediate: true });
        }
        if (this._skipLayerActive) {
            this._restoreSkipLayerSpeeds({ immediate: true });
        }
        this._cancelSkipSpeedRamp();
        setGlobalTrailMaxStepDistance(0);

        this._forwardPassComplete = false;
        this._skipToEndActive = false;
        this._skipLayerActive = false;
        this._skipLayerLast = false;
        this._skipLayerRestore = null;
        this._skipToEndRestore = null;

        if (this._skipToEndRaf) {
            if (typeof window !== 'undefined' && typeof window.cancelAnimationFrame === 'function') {
                window.cancelAnimationFrame(this._skipToEndRaf);
            } else {
                clearTimeout(this._skipToEndRaf);
            }
            this._skipToEndRaf = null;
        }

        if (this._postResetTrailPurgeRaf) {
            if (typeof window !== 'undefined' && typeof window.cancelAnimationFrame === 'function') {
                window.cancelAnimationFrame(this._postResetTrailPurgeRaf);
            } else {
                clearTimeout(this._postResetTrailPurgeRaf);
            }
            this._postResetTrailPurgeRaf = null;
        }

        if (typeof TWEEN !== 'undefined' && typeof TWEEN.removeAll === 'function') {
            try { TWEEN.removeAll(); } catch (_) { /* no-op */ }
        }

        this._trailPassId = (Number.isFinite(this._trailPassId) ? this._trailPassId : 0) + 1;
        this._topLnParamPlaceholders = null;
        if (engine && engine.scene && engine.scene.userData) {
            engine.scene.userData.trailPassId = this._trailPassId;
        }
        if (engine && engine.scene) {
            clearTrailsFromScene(engine.scene, { passId: this._trailPassId });
        }

        const oldLayers = Array.isArray(this._layers) ? this._layers : [];
        this._layers = [];
        this._currentLayerIdx = 0;

        const autoDriver = this._autoCameraDriver;
        if (engine && Array.isArray(engine._layers)) {
            engine._layers = engine._layers.filter(layer => !oldLayers.includes(layer) && layer !== autoDriver);
        }

        oldLayers.forEach((layer) => {
            if (!layer) return;
            const root = layer.raycastRoot || layer.root;
            if (root && engine && typeof engine.removeRaycastRoot === 'function') {
                engine.removeRaycastRoot(root);
            }
            if (layer.root && layer.root.parent) {
                layer.root.parent.remove(layer.root);
            }
            if (typeof layer.dispose === 'function') {
                layer.dispose();
            }
        });

        this._activationSource = activationSource || null;
        this._laneCount = Math.max(1, Math.floor(laneCount || 1));

        const layerStackSpacing = LAYER_STACK_SPACING_Y;
        this._initLayers(layerStackSpacing);
        if (engine && Array.isArray(engine._layers) && autoDriver && !engine._layers.includes(autoDriver)) {
            engine._layers.push(autoDriver);
        }
        this._schedulePostResetTrailPurge(1500);
        this.dispatchEvent(new Event('progress'));
    }

    _schedulePostResetTrailPurge(durationMs = 1500) {
        const engine = this._engine;
        if (!engine || !engine.scene) return;
        const duration = Math.max(0, Number.isFinite(durationMs) ? durationMs : 0);
        if (duration <= 0) return;

        const schedule = (typeof window !== 'undefined' && typeof window.requestAnimationFrame === 'function')
            ? window.requestAnimationFrame.bind(window)
            : (cb) => setTimeout(cb, 16);
        const cancel = (typeof window !== 'undefined' && typeof window.cancelAnimationFrame === 'function')
            ? window.cancelAnimationFrame.bind(window)
            : (id) => clearTimeout(id);

        if (this._postResetTrailPurgeRaf) {
            cancel(this._postResetTrailPurgeRaf);
            this._postResetTrailPurgeRaf = null;
        }

        const start = (typeof performance !== 'undefined' && performance.now)
            ? performance.now()
            : Date.now();
        this._postResetTrailPurgeUntil = start + duration;

        const tick = () => {
            if (!engine.scene) return;
            clearTrailsFromScene(engine.scene, { passId: this._trailPassId });
            const now = (typeof performance !== 'undefined' && performance.now)
                ? performance.now()
                : Date.now();
            if (now >= this._postResetTrailPurgeUntil) {
                this._postResetTrailPurgeRaf = null;
                return;
            }
            this._postResetTrailPurgeRaf = schedule(tick);
        };

        this._postResetTrailPurgeRaf = schedule(tick);
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
            engineSpeed = DEFAULT_SKIP_SPEED_PROFILE.engineSpeed,
            globalSpeed = DEFAULT_SKIP_SPEED_PROFILE.globalSpeed,
            prismAddSpeed = DEFAULT_SKIP_SPEED_PROFILE.prismAddSpeed,
            selfAttentionSpeed = DEFAULT_SKIP_SPEED_PROFILE.selfAttentionSpeed
        } = opts || {};
        this._rampSpeedProfile({
            engineSpeed,
            globalSpeed,
            prismAddSpeed,
            selfAttentionSpeed
        }, { durationMs: SKIP_SPEED_RAMP_IN_MS });
        setGlobalTrailMaxStepDistance(SKIP_TRAIL_MAX_STEP_DISTANCE);

        if (layer && typeof layer.setSkipToEndMode === 'function') {
            layer.setSkipToEndMode(true);
        }
    }

    skipToEndForwardPass(opts = {}) {
        if (this._skipToEndActive || this._checkForwardPassComplete()) return;
        if (this._skipLayerActive) {
            this._restoreSkipLayerSpeeds({ immediate: true });
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
            engineSpeed = DEFAULT_SKIP_SPEED_PROFILE.engineSpeed,
            globalSpeed = DEFAULT_SKIP_SPEED_PROFILE.globalSpeed,
            prismAddSpeed = DEFAULT_SKIP_SPEED_PROFILE.prismAddSpeed,
            selfAttentionSpeed = DEFAULT_SKIP_SPEED_PROFILE.selfAttentionSpeed
        } = opts || {};
        this._rampSpeedProfile({
            engineSpeed,
            globalSpeed,
            prismAddSpeed,
            selfAttentionSpeed
        }, { durationMs: SKIP_SPEED_RAMP_IN_MS });
        setGlobalTrailMaxStepDistance(SKIP_TRAIL_MAX_STEP_DISTANCE);

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

    _getSpeedProfileSnapshot() {
        const engine = this._engine;
        return {
            engineSpeed: engine && typeof engine._speed === 'number' ? engine._speed : 1,
            globalSpeed: GLOBAL_ANIM_SPEED_MULT,
            prismAddSpeed: PRISM_ADD_ANIM_SPEED_MULT,
            selfAttentionSpeed: SELF_ATTENTION_TIME_MULT
        };
    }

    _applySpeedProfile(profile) {
        if (!profile) return;
        if (this._engine && typeof this._engine.setSpeed === 'function' && Number.isFinite(profile.engineSpeed) && profile.engineSpeed > 0) {
            this._engine.setSpeed(profile.engineSpeed);
        }
        if (Number.isFinite(profile.globalSpeed) && profile.globalSpeed > 0) {
            setGlobalAnimSpeedMult(profile.globalSpeed);
        }
        if (Number.isFinite(profile.prismAddSpeed) && profile.prismAddSpeed > 0) {
            setPrismAddAnimSpeedMult(profile.prismAddSpeed);
        }
        if (Number.isFinite(profile.selfAttentionSpeed) && profile.selfAttentionSpeed > 0) {
            setSelfAttentionTimeMult(profile.selfAttentionSpeed);
        }
    }

    _cancelSkipSpeedRamp() {
        if (!this._skipSpeedRampRaf) return;
        if (typeof window !== 'undefined' && typeof window.cancelAnimationFrame === 'function') {
            window.cancelAnimationFrame(this._skipSpeedRampRaf);
        } else {
            clearTimeout(this._skipSpeedRampRaf);
        }
        this._skipSpeedRampRaf = null;
    }

    _rampSpeedProfile(targetProfile, { durationMs = SKIP_SPEED_RAMP_IN_MS, immediate = false } = {}) {
        if (!targetProfile) return;
        const target = {
            engineSpeed: Number.isFinite(targetProfile.engineSpeed) ? targetProfile.engineSpeed : 1,
            globalSpeed: Number.isFinite(targetProfile.globalSpeed) ? targetProfile.globalSpeed : GLOBAL_ANIM_SPEED_MULT,
            prismAddSpeed: Number.isFinite(targetProfile.prismAddSpeed) ? targetProfile.prismAddSpeed : PRISM_ADD_ANIM_SPEED_MULT,
            selfAttentionSpeed: Number.isFinite(targetProfile.selfAttentionSpeed) ? targetProfile.selfAttentionSpeed : SELF_ATTENTION_TIME_MULT
        };
        const duration = Math.max(0, Number(durationMs) || 0);
        this._cancelSkipSpeedRamp();
        if (immediate || duration <= 0) {
            this._applySpeedProfile(target);
            return;
        }

        const start = this._getSpeedProfileSnapshot();
        const nowFn = (typeof performance !== 'undefined' && typeof performance.now === 'function')
            ? performance.now.bind(performance)
            : Date.now;
        const startedAt = nowFn();
        const schedule = (typeof window !== 'undefined' && typeof window.requestAnimationFrame === 'function')
            ? window.requestAnimationFrame.bind(window)
            : (cb) => setTimeout(cb, 16);
        const lerp = (a, b, t) => a + (b - a) * t;
        const easeOutCubic = (t) => {
            const inv = 1 - t;
            return 1 - inv * inv * inv;
        };

        const tick = () => {
            const elapsed = nowFn() - startedAt;
            const t = Math.min(1, elapsed / duration);
            const eased = easeOutCubic(t);
            this._applySpeedProfile({
                engineSpeed: lerp(start.engineSpeed, target.engineSpeed, eased),
                globalSpeed: lerp(start.globalSpeed, target.globalSpeed, eased),
                prismAddSpeed: lerp(start.prismAddSpeed, target.prismAddSpeed, eased),
                selfAttentionSpeed: lerp(start.selfAttentionSpeed, target.selfAttentionSpeed, eased)
            });
            if (t < 1) {
                this._skipSpeedRampRaf = schedule(tick);
            } else {
                this._skipSpeedRampRaf = null;
            }
        };

        this._skipSpeedRampRaf = schedule(tick);
    }

    _checkForwardPassComplete() {
        if (this._forwardPassComplete) return true;
        if (!Array.isArray(this._layers) || !this._layers.length) return false;
        const allLayersDone = this._layers.every(layer => layer && layer._completed);
        if (!allLayersDone) return false;

        const lastLayer = this._layers[this._numLayers - 1];
        const stopY = lastLayer && (Number.isFinite(lastLayer.__topEmbedExitYLocal)
            ? lastLayer.__topEmbedExitYLocal
            : lastLayer.__topEmbedStopYLocal);
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

    _finalizeSkipToEnd({ immediate = false } = {}) {
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
            this._rampSpeedProfile(restore, { durationMs: SKIP_SPEED_RAMP_OUT_MS, immediate });
            this._skipToEndRestore = null;
        }
        setGlobalTrailMaxStepDistance(0);
    }

    _restoreSkipLayerSpeeds({ immediate = false } = {}) {
        if (!this._skipLayerActive) return;
        const restore = this._skipLayerRestore;
        if (Array.isArray(this._layers)) {
            this._layers.forEach(layer => {
                if (layer && typeof layer.setSkipToEndMode === 'function') {
                    layer.setSkipToEndMode(false);
                }
            });
        }
        if (restore) {
            this._rampSpeedProfile(restore, { durationMs: SKIP_SPEED_RAMP_OUT_MS, immediate });
        }
        setGlobalTrailMaxStepDistance(0);
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
        if (this._skipLayerActive && prevLayer && typeof prevLayer.restoreResidualVectorVisibility === 'function') {
            prevLayer.restoreResidualVectorVisibility(externalLanes);
        }
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

        this._autoCamera?.maybeFocus?.({ immediate: true });

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
        let exitYLocal = null;
        const embedHeight = EMBEDDING_MATRIX_PARAMS_VOCAB.height;
        const embedInset = 5;
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
                const entryWorldY = centerWorld.y - embedHeight / 2 + embedInset;
                const exitWorldY = centerWorld.y + embedHeight / 2 - embedInset;
                const entryLocalVec = new THREE.Vector3(0, entryWorldY, 0);
                lastLayer.root.worldToLocal(entryLocalVec);
                targetYLocal = entryLocalVec.y;
                const exitLocalVec = new THREE.Vector3(0, exitWorldY, 0);
                lastLayer.root.worldToLocal(exitLocalVec);
                exitYLocal = exitLocalVec.y;
            }
        } catch (_) { /* fallback to formula below */ }

        if (targetYLocal == null) {
            const towerTopYLocal = lastLayer.mlpDown.group.position.y + MLP_MATRIX_PARAMS_DOWN.height / 2;
            const topVocabCenterYLocal = towerTopYLocal + TOP_EMBED_Y_GAP_ABOVE_TOWER + embedHeight / 2 + TOP_EMBED_Y_ADJUST;
            targetYLocal = topVocabCenterYLocal - embedHeight / 2 + embedInset;
            exitYLocal = topVocabCenterYLocal + embedHeight / 2 - embedInset;
        }

        // Cap how far vectors rise within the top vocab embedding so they
        // don't reach the logit bars above.
        const riseFracRaw = Number.isFinite(TOP_EMBED_MAX_RISE_FRACTION) ? TOP_EMBED_MAX_RISE_FRACTION : 1;
        const riseFrac = Math.max(0, Math.min(1, riseFracRaw));
        if (Number.isFinite(targetYLocal) && Number.isFinite(exitYLocal) && riseFrac < 1) {
            const maxRise = embedHeight * riseFrac;
            const cappedExit = targetYLocal + maxRise;
            if (exitYLocal > cappedExit) exitYLocal = cappedExit;
        }

        try {
            if (Number.isFinite(exitYLocal) && Number.isFinite(targetYLocal) && exitYLocal < targetYLocal) {
                exitYLocal = targetYLocal;
            }
            const finalStopY = Number.isFinite(exitYLocal) ? exitYLocal : targetYLocal;
            if (lastLayer.mhsaAnimation) {
                lastLayer.mhsaAnimation.finalOriginalY = finalStopY;
                lastLayer.mhsaAnimation.topEmbeddingStopY = finalStopY;
                lastLayer.mhsaAnimation.postSplitRiseSpeed = ANIM_RISE_SPEED_ORIGINAL;
            }
            lastLayer.__topEmbedEntryYLocal = targetYLocal;
            lastLayer.__topEmbedExitYLocal = Number.isFinite(exitYLocal) ? exitYLocal : targetYLocal;
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
        const entryYLocal = targetYLocal;
        const exitYLocal = Number.isFinite(lastLayer?.__topEmbedExitYLocal)
            ? lastLayer.__topEmbedExitYLocal
            : targetYLocal;
        const allowSkipVisible = this._skipToEndActive;
        let needsSkipRefresh = false;
        const markSkipVisible = (vec) => {
            if (!allowSkipVisible || !vec || !vec.group) return;
            vec.group.userData = vec.group.userData || {};
            vec.group.userData.skipVisible = true;
            if (vec.mesh) {
                vec.mesh.userData = vec.mesh.userData || {};
                vec.mesh.userData.skipVisible = true;
            }
            needsSkipRefresh = true;
        };
        const refreshSkipVisibility = () => {
            if (!needsSkipRefresh || !allowSkipVisible) return;
            if (lastLayer && typeof lastLayer.refreshSkipVisibility === 'function') {
                lastLayer.refreshSkipVisibility();
            }
            needsSkipRefresh = false;
        };
        const startEmbedTraverse = (resVec, updateTrailFn = null) => {
            if (!resVec || !resVec.group) return;
            if (!Number.isFinite(exitYLocal) || exitYLocal <= entryYLocal + 0.01) return;
            const dist = Math.max(0, exitYLocal - resVec.group.position.y);
            if (dist <= 0.01) return;
            const durMs = (dist / (ANIM_RISE_SPEED_ORIGINAL * GLOBAL_ANIM_SPEED_MULT)) * 1000;
            new TWEEN.Tween(resVec.group.position)
                .to({ y: exitYLocal }, Math.max(100, durMs))
                .easing(TWEEN.Easing.Quadratic.InOut)
                .onUpdate(() => {
                    if (typeof updateTrailFn === 'function') updateTrailFn(resVec);
                    this.dispatchEvent(new Event('progress'));
                })
                .onComplete(() => {
                    if (typeof updateTrailFn === 'function') updateTrailFn(resVec);
                    this.dispatchEvent(new Event('progress'));
                })
                .start();
        };

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
                let desiredOpacity = 1.0;
                if (lnColorState.locked) {
                    tempColor.copy(lnColorState.lockedColor);
                    desiredOpacity = 1.0;
                } else if (highest >= lnBottomY && highest < lnCenterY) {
                    const denom = Math.max(lnCenterY - lnBottomY, 1e-6);
                    const t = (highest - lnBottomY) / denom;
                    tempColor.copy(COLOR_DARK_GRAY).lerp(COLOR_LIGHT_YELLOW, t);
                    desiredOpacity = THREE.MathUtils.lerp(1.0, 0.6, t);
                } else if (highest >= lnCenterY && highest < lnTopY) {
                    tempColor.copy(COLOR_LIGHT_YELLOW);
                    desiredOpacity = 0.6;
                } else if (highest >= lnTopY) {
                    const tRaw = (highest - lnTopY) / exitTransitionRange;
                    const t = Math.min(1, Math.max(0, tRaw));
                    tempColor.copy(COLOR_LIGHT_YELLOW).lerp(COLOR_BRIGHT_YELLOW, t);
                    desiredOpacity = THREE.MathUtils.lerp(0.6, 1.0, t);
                } else {
                    tempColor.copy(COLOR_DARK_GRAY);
                    desiredOpacity = 1.0;
                }

                if (highest >= lnTopY + exitTransitionRange) {
                    lnColorState.locked = true;
                    lnColorState.lockedColor.copy(COLOR_BRIGHT_YELLOW);
                    tempColor.copy(lnColorState.lockedColor);
                    desiredOpacity = 1.0;
                }

                const smoothAlpha = this._skipToEndActive ? SKIP_COMPONENT_COLOR_LERP_ALPHA : 1;
                if (smoothAlpha >= 1) {
                    lnColorState.currentColor.copy(tempColor);
                    lnColorState.currentOpacity = desiredOpacity;
                } else {
                    lnColorState.currentColor.lerp(tempColor, smoothAlpha);
                    lnColorState.currentOpacity = THREE.MathUtils.lerp(lnColorState.currentOpacity, desiredOpacity, smoothAlpha);
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

            const getFinalLnParamData = (param, targetLength) => {
                const fallbackValue = param === 'scale' ? 1 : 0;
                const fromParams = getLayerNormParamData(lastLayer.index, 'final', param, targetLength);
                return toMutableArray(fromParams, targetLength, fallbackValue);
            };

            const getFinalLnStageData = (stage, lane, targetLength) => {
                const tokenIndex = (lane && Number.isFinite(lane.tokenIndex))
                    ? lane.tokenIndex
                    : null;
                if (!Number.isFinite(tokenIndex)) return null;
                const source = this._activationSource;
                if (!source || typeof source.getFinalLayerNorm !== 'function') return null;
                try {
                    return source.getFinalLayerNorm(stage, tokenIndex, targetLength);
                } catch (_) {
                    return null;
                }
            };

            const topLnPlaceholders = (
                this._topLnParamPlaceholders
                && this._topLnParamPlaceholders.layerIndex === lastLayer.index
            ) ? this._topLnParamPlaceholders : null;
            const hideTopLnPlaceholder = (laneIdx, kind) => {
                if (!topLnPlaceholders || !Number.isFinite(laneIdx)) return;
                const list = kind === 'scale' ? topLnPlaceholders.scale : topLnPlaceholders.shift;
                if (!Array.isArray(list)) return;
                const vec = list[laneIdx];
                if (vec && vec.group) vec.group.visible = false;
            };

            lastLayer.lanes.forEach(lane => {
                const vec = lane && lane.originalVec;
                if (!vec || !vec.group) return;
                markSkipVisible(vec);
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
                const targetLength = Math.max(1, Math.floor(vec.instanceCount || VECTOR_LENGTH_PRISM));
                const finalScaleParams = getFinalLnParamData('scale', targetLength);
                const finalShiftParams = getFinalLnParamData('shift', targetLength);
                const laneIdx = Number.isFinite(lane?.laneIndex) ? lane.laneIndex : -1;
                const scalePlaceholder = (laneIdx >= 0 && topLnPlaceholders && Array.isArray(topLnPlaceholders.scale))
                    ? topLnPlaceholders.scale[laneIdx]
                    : null;
                const shiftPlaceholder = (laneIdx >= 0 && topLnPlaceholders && Array.isArray(topLnPlaceholders.shift))
                    ? topLnPlaceholders.shift[laneIdx]
                    : null;
                const usingScalePlaceholder = !!(scalePlaceholder && scalePlaceholder.group);
                const usingShiftPlaceholder = !!(shiftPlaceholder && shiftPlaceholder.group);

                let multVec = usingScalePlaceholder ? scalePlaceholder : null;
                if (!multVec) {
                    multVec = new VectorVisualizationInstancedPrism(
                        finalScaleParams,
                        new THREE.Vector3(0, lnCenterY, zPos),
                        30,
                        vec.instanceCount
                    );
                    lastLayer.root.add(multVec.group);
                    multVec.group.visible = false;
                }
                markSkipVisible(multVec);
                recolorVectorFromData(multVec, finalScaleParams, LN_PARAM_MONOCHROME);

                let addVec = usingShiftPlaceholder ? shiftPlaceholder : null;
                if (!addVec) {
                    addVec = new VectorVisualizationInstancedPrism(
                        finalShiftParams,
                        new THREE.Vector3(0, lnCenterY + LN_PARAMS.height / 4, zPos),
                        30,
                        vec.instanceCount
                    );
                    lastLayer.root.add(addVec.group);
                    addVec.group.visible = false;
                }
                markSkipVisible(addVec);
                recolorVectorFromData(addVec, finalShiftParams, LN_PARAM_MONOCHROME);

                const applyFinalLnInputColor = () => {
                    if (lane.__topLnInputColored) return;
                    recolorVectorFromData(vec, vec.rawData, FINAL_LN_INPUT_LIGHT_BLUE);
                    lane.__topLnInputColored = true;
                };
                const activateFinalLnParamColors = () => {
                    if (lane.__topLnParamsColored) return;
                    recolorVectorFromData(multVec, multVec.rawData, null);
                    recolorVectorFromData(addVec, addVec.rawData, null);
                    lane.__topLnParamsColored = true;
                };
                const markTopLnEntered = () => {
                    if (!lane.__topLnEntered) lane.__topLnEntered = true;
                    if (multVec && multVec.group) multVec.group.visible = true;
                    if (addVec && addVec.group) addVec.group.visible = true;
                    applyFinalLnInputColor();
                    activateFinalLnParamColors();
                };

                const normAnim = new PrismLayerNormAnimation(vec);
                let normLoopActive = false;

                if (startY >= lnBottomY) {
                    markTopLnEntered();
                }

                const startFinalRise = (resVec) => {
                    const riseDist = Math.max(0, entryYLocal - resVec.group.position.y);
                    const durMs = (riseDist / (ANIM_RISE_SPEED_ORIGINAL * GLOBAL_ANIM_SPEED_MULT)) * 1000;
                    new TWEEN.Tween(resVec.group.position)
                        .to({ y: entryYLocal }, Math.max(100, durMs))
                        .easing(TWEEN.Easing.Quadratic.InOut)
                        .onUpdate(() => {
                            updateTopLnColor(resVec.group.position.y);
                            updateTrailPosition(resVec);
                            this.dispatchEvent(new Event('progress'));
                        })
                        .onComplete(() => {
                            updateTopLnColor(entryYLocal + exitTransitionRange);
                            updateTrailPosition(resVec);
                            if (lane && lane.__topLnStopRise) {
                                delete lane.stopRise;
                                delete lane.stopRiseTarget;
                                delete lane.__topLnStopRise;
                            }
                            this.dispatchEvent(new Event('progress'));
                            startEmbedTraverse(resVec, updateTrailPosition);
                        })
                        .start();
                };

                const beginMultiply = () => {
                    if (lane.__topLnMultStarted) return;
                    lane.__topLnMultStarted = true;
                    if (!usingScalePlaceholder) hideTopLnPlaceholder(lane && lane.laneIndex, 'scale');
                    if (!usingShiftPlaceholder) hideTopLnPlaceholder(lane && lane.laneIndex, 'shift');
                    multVec.group.visible = true;
                    addVec.group.visible = true;
                    activateFinalLnParamColors();

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
                        markSkipVisible(resVec);

                        if (!usingScalePlaceholder && multVec.group && multVec.group.parent) {
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

                        const finalLnShiftedData = getFinalLnStageData('shift', lane, resVec.instanceCount);
                        const finalAdditionData = (isArrayLike(finalLnShiftedData) && finalLnShiftedData.length)
                            ? finalLnShiftedData
                            : buildElementwiseSum(resVec.rawData, addVec.rawData, resVec.instanceCount);

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
                            markSkipVisible(addVec);
                            startFinalRise(addVec);
                        }, {
                            suppressResidualTrailUpdates: true,
                            finalData: finalAdditionData
                        });

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
                        if (vec.group.position.y >= lnBottomY) {
                            markTopLnEntered();
                        }
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
                                markTopLnEntered();
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

            refreshSkipVisibility();
            return;
        }

        lastLayer.lanes.forEach(lane => {
            const vec = lane && lane.originalVec;
            if (!vec || !vec.group) return;
            markSkipVisible(vec);
            const startY = vec.group.position.y;
            if (typeof startY !== 'number' || !isFinite(startY)) return;
            if (startY >= entryYLocal - 0.01) {
                startEmbedTraverse(vec);
                return;
            }

            const riseDist = Math.max(0, entryYLocal - startY);
            const durMs = (riseDist / (ANIM_RISE_SPEED_ORIGINAL * GLOBAL_ANIM_SPEED_MULT)) * 1000;

            new TWEEN.Tween(vec.group.position)
                .to({ y: entryYLocal }, Math.max(100, durMs))
                .easing(TWEEN.Easing.Quadratic.InOut)
                .onUpdate(() => this.dispatchEvent(new Event('progress')))
                .onComplete(() => {
                    this.dispatchEvent(new Event('progress'));
                    startEmbedTraverse(vec);
                })
                .start();
        });

        refreshSkipVisibility();
    }

    setDevMode(enabled) {
        this._autoCamera?.setDevMode?.(enabled);
    }
}

/** Convenience helper mirroring CoreEngine.startEngine signature */
export function startPipeline(canvas, numLayers = 12, opts = {}) {
    const pipeline = new LayerPipeline(canvas, numLayers, opts);
    return () => pipeline.dispose();
} 
