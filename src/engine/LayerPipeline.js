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
    SKIP_COMPONENT_COLOR_LERP_ALPHA,
    VECTOR_LENGTH_PRISM,
    NUM_VECTOR_LANES,
    LAYER_STACK_SPACING_Y,
    HIDE_INSTANCE_Y_OFFSET,
    SA_RED_EXTRA_RISE,
    INACTIVE_COMPONENT_COLOR
} from '../utils/constants.js';
import { VectorVisualizationInstancedPrism } from '../components/VectorVisualizationInstancedPrism.js';
import { BatchedPrismVectorSet } from '../components/BatchedPrismVectorSet.js';
import { startPrismAdditionAnimation } from '../utils/additionUtils.js';
import { PrismLayerNormAnimation } from '../animations/PrismLayerNormAnimation.js';
import {
    MHA_FINAL_K_COLOR,
    MHA_VALUE_SPECTRUM_COLOR,
    MHA_VALUE_HUE_SPREAD,
    MHA_VALUE_LIGHTNESS_MIN,
    MHA_VALUE_LIGHTNESS_MAX,
    MHA_VALUE_RANGE_MIN,
    MHA_VALUE_RANGE_MAX,
    MHA_VALUE_CLAMP_MAX
} from '../animations/LayerAnimationConstants.js';
import { buildHueRangeOptions, mapValueToHueRange } from '../utils/colors.js';
import { setGlobalTrailMaxStepDistance, clearTrailsFromScene, refreshTrailDisplayScales } from '../utils/trailUtils.js';
import { applyLayerNormMaterial } from './layers/gpt2LayerUtils.js';
import { scaleGlobalEmissiveIntensity } from '../utils/materialUtils.js';
import {
    buildElementwiseSum,
    isArrayLike,
    recolorVectorFromData,
    simplePrismMultiply,
    toMutableArray
} from './layerPipelineMath.js';
import {
    activateLayerNormColor as activateTopLayerNormColor,
    calculateTopEmbeddingTargets,
    findTopLayerNormInfo
} from './layerPipelineTopEmbedding.js';

const COLOR_DARK_GRAY = new THREE.Color(INACTIVE_COMPONENT_COLOR);
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

const SKIP_SPEED_RAMP_IN_MS = 40;
const SKIP_SPEED_RAMP_OUT_MS = 70;
const POST_RESET_TRAIL_PURGE_INTERVAL_MS = 120;
const DEFAULT_SKIP_SPEED_PROFILE = Object.freeze({
    engineSpeed: 13.5,
    globalSpeed: 2600,
    prismAddSpeed: 780,
    selfAttentionSpeed: 0.0075
});

const TMP_WORLD_POS = new THREE.Vector3();
const TMP_KV_PROXY_CENTER = new THREE.Vector3();
const TMP_KV_PROXY_SIZE = new THREE.Vector3();
const TMP_KV_PROXY_LOCAL_CENTER = new THREE.Vector3();
const TMP_KV_PROXY_INST_MATRIX = new THREE.Matrix4();
const TMP_KV_PROXY_INST_POS = new THREE.Vector3();
const TMP_KV_PROXY_INST_QUAT = new THREE.Quaternion();
const TMP_KV_PROXY_INST_SCALE = new THREE.Vector3();
const TMP_KV_PROXY_CORNER = new THREE.Vector3();
const TMP_KV_CAPTURE_WORLD = new THREE.Vector3();
const TMP_KV_CAPTURE_LOCAL = new THREE.Vector3();
const TMP_KV_CAPTURE_INV = new THREE.Matrix4();
const TMP_KV_BOOTSTRAP_LOCAL = new THREE.Vector3();
const TMP_KV_BOOTSTRAP_WORLD = new THREE.Vector3();
const KV_HIDDEN_SCALE_EPS = 0.01;
const KV_COLLAPSED_SCALE = 0.001;
const KV_CACHE_BATCH_CHUNK_SIZE = 24;
const KV_RAYCAST_PROXY_GEOMETRY = new THREE.BoxGeometry(1, 1, 1);
const KV_RAYCAST_PROXY_MATERIAL = (() => {
    const mat = new THREE.MeshBasicMaterial({
        transparent: true,
        opacity: 0,
        depthWrite: false,
        depthTest: false
    });
    // Keep this proxy non-rendering while still raycastable.
    mat.visible = false;
    return mat;
})();


function computeVisibleInstancedBoundsLocal(vec, outBounds) {
    if (!vec || !vec.mesh || !vec.mesh.isInstancedMesh || !outBounds) return false;
    const mesh = vec.mesh;
    const geom = mesh.geometry;
    if (!geom) return false;
    if (!geom.boundingBox && typeof geom.computeBoundingBox === 'function') {
        geom.computeBoundingBox();
    }
    const bb = geom.boundingBox;
    if (!bb) return false;

    const countFromVec = Number.isFinite(vec.instanceCount) ? Math.max(1, Math.floor(vec.instanceCount)) : null;
    const countFromMesh = Number.isFinite(mesh.count) ? Math.max(1, Math.floor(mesh.count)) : null;
    const count = Math.min(
        countFromVec ?? countFromMesh ?? 0,
        countFromMesh ?? countFromVec ?? 0
    );
    if (!Number.isFinite(count) || count <= 0) return false;

    const hiddenThreshold = HIDE_INSTANCE_Y_OFFSET * 0.5;
    outBounds.makeEmpty();
    let hasVisible = false;

    const meshMatrix = mesh.matrix;
    const minX = bb.min.x;
    const maxX = bb.max.x;
    const minY = bb.min.y;
    const maxY = bb.max.y;
    const minZ = bb.min.z;
    const maxZ = bb.max.z;
    for (let i = 0; i < count; i++) {
        mesh.getMatrixAt(i, TMP_KV_PROXY_INST_MATRIX);
        TMP_KV_PROXY_INST_MATRIX.decompose(TMP_KV_PROXY_INST_POS, TMP_KV_PROXY_INST_QUAT, TMP_KV_PROXY_INST_SCALE);
        if (!Number.isFinite(TMP_KV_PROXY_INST_POS.y) || TMP_KV_PROXY_INST_POS.y <= hiddenThreshold) continue;
        if (!Number.isFinite(TMP_KV_PROXY_INST_SCALE.y) || Math.abs(TMP_KV_PROXY_INST_SCALE.y) < KV_HIDDEN_SCALE_EPS) continue;

        hasVisible = true;
        for (let xi = 0; xi < 2; xi++) {
            const x = xi === 0 ? minX : maxX;
            for (let yi = 0; yi < 2; yi++) {
                const y = yi === 0 ? minY : maxY;
                for (let zi = 0; zi < 2; zi++) {
                    const z = zi === 0 ? minZ : maxZ;
                    TMP_KV_PROXY_CORNER.set(x, y, z);
                    TMP_KV_PROXY_CORNER.applyMatrix4(TMP_KV_PROXY_INST_MATRIX);
                    TMP_KV_PROXY_CORNER.applyMatrix4(meshMatrix);
                    outBounds.expandByPoint(TMP_KV_PROXY_CORNER);
                }
            }
        }
    }

    return hasVisible && !outBounds.isEmpty();
}

function normalizeHiddenInstancesForPersistentKv(vec) {
    if (!vec || !vec.mesh || !vec.mesh.isInstancedMesh) return false;
    const mesh = vec.mesh;
    const countFromVec = Number.isFinite(vec.instanceCount) ? Math.max(1, Math.floor(vec.instanceCount)) : null;
    const countFromMesh = Number.isFinite(mesh.count) ? Math.max(1, Math.floor(mesh.count)) : null;
    const count = Math.min(
        countFromVec ?? countFromMesh ?? 0,
        countFromMesh ?? countFromVec ?? 0
    );
    if (!Number.isFinite(count) || count <= 0) return false;

    const hiddenThreshold = HIDE_INSTANCE_Y_OFFSET * 0.5;
    const baseY = Number.isFinite(vec._basePrismCenterY) ? vec._basePrismCenterY : 0;
    let changed = false;

    for (let i = 0; i < count; i++) {
        mesh.getMatrixAt(i, TMP_KV_PROXY_INST_MATRIX);
        TMP_KV_PROXY_INST_MATRIX.decompose(TMP_KV_PROXY_INST_POS, TMP_KV_PROXY_INST_QUAT, TMP_KV_PROXY_INST_SCALE);

        const hiddenByY = !Number.isFinite(TMP_KV_PROXY_INST_POS.y) || TMP_KV_PROXY_INST_POS.y <= hiddenThreshold;
        const hiddenByScale = !Number.isFinite(TMP_KV_PROXY_INST_SCALE.y) || Math.abs(TMP_KV_PROXY_INST_SCALE.y) < KV_HIDDEN_SCALE_EPS;
        if (!hiddenByY && !hiddenByScale) continue;

        const needsRelayout = hiddenByY
            || Math.abs(TMP_KV_PROXY_INST_POS.y - baseY) > 1e-4
            || Math.abs(TMP_KV_PROXY_INST_SCALE.x - KV_COLLAPSED_SCALE) > 1e-4
            || Math.abs(TMP_KV_PROXY_INST_SCALE.y - KV_COLLAPSED_SCALE) > 1e-4
            || Math.abs(TMP_KV_PROXY_INST_SCALE.z - KV_COLLAPSED_SCALE) > 1e-4;
        if (!needsRelayout) continue;

        TMP_KV_PROXY_INST_POS.y = baseY;
        TMP_KV_PROXY_INST_SCALE.set(KV_COLLAPSED_SCALE, KV_COLLAPSED_SCALE, KV_COLLAPSED_SCALE);
        TMP_KV_PROXY_INST_MATRIX.compose(TMP_KV_PROXY_INST_POS, TMP_KV_PROXY_INST_QUAT, TMP_KV_PROXY_INST_SCALE);
        mesh.setMatrixAt(i, TMP_KV_PROXY_INST_MATRIX);
        changed = true;
    }

    if (changed && mesh.instanceMatrix) {
        mesh.instanceMatrix.needsUpdate = true;
    }
    return changed;
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
        this._laneLayoutCount = this._laneCount;
        this._activeLaneLayoutIndices = Array.from({ length: this._laneCount }, (_, idx) => idx);
        this._passLaneTokenIndices = null;
        this._kvCacheModeEnabled = false;
        this._kvCacheDecodeActive = false;
        this._reuseKvCacheForPass = false;
        this._kvCacheEntriesByLayer = new Map();
        this._kvCacheBatchStores = new Map();
        this._kvCachePersistentRoot = null;

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
        this._skipKvCaptureReadyLayers = new Set();
        this._postResetTrailPurgeRaf = null;
        this._postResetTrailPurgeUntil = 0;
        this._trailPassId = 1;
        this._topLnParamPlaceholders = null;
        this._awaitingTopLogitReveal = false;
        this._topLogitRevealComplete = true;
        this._topLogitRevealGateId = 0;
        this._mhsaRuntimePrewarmHandles = new Map();

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
        if (this._engine && this._engine.scene) {
            this._kvCachePersistentRoot = new THREE.Group();
            this._kvCachePersistentRoot.name = 'KvCachePersistentRoot';
            this._kvCachePersistentRoot.userData.skipVisible = true;
            this._engine.scene.add(this._kvCachePersistentRoot);
            if (typeof this._engine.registerRaycastRoot === 'function') {
                this._engine.registerRaycastRoot(this._kvCachePersistentRoot);
            }
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
            const cachedKvEntries = this._reuseKvCacheForPass
                ? this._getCachedKvEntriesForLayer(i)
                : [];
            const layer = new Gpt2Layer(
                i,
                rand,
                0,
                /*externalLanes*/ null,
                /*onFinished*/ null,
                isActive,
                this._activationSource,
                this._laneCount,
                layerStackSpacing,
                {
                    laneLayoutCount: this._laneLayoutCount,
                    activeLaneLayoutIndices: this._activeLaneLayoutIndices,
                    laneTokenIndices: this._passLaneTokenIndices,
                    kvCacheModeEnabled: this._kvCacheModeEnabled,
                    kvCacheDecodeActive: this._kvCacheDecodeActive,
                    cachedKvEntries
                }
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
        this._scheduleMhsaRuntimePrewarm(1);
    }

    _scheduleMhsaRuntimePrewarm(layerIndex) {
        const idx = Number.isFinite(layerIndex) ? Math.floor(layerIndex) : -1;
        if (idx < 0 || idx >= this._layers.length) return;
        if (this._mhsaRuntimePrewarmHandles.has(idx)) return;
        const layer = this._layers[idx];
        if (!layer || layer.isActive) return;

        const run = () => {
            this._mhsaRuntimePrewarmHandles.delete(idx);
            try {
                layer.ensureMhsaRuntimeResources?.();
            } catch (_) { /* best-effort prewarm */ }
        };

        if (typeof window !== 'undefined' && typeof window.requestIdleCallback === 'function') {
            const handle = window.requestIdleCallback(run, { timeout: 1200 });
            this._mhsaRuntimePrewarmHandles.set(idx, { type: 'idle', handle });
            return;
        }

        const handle = setTimeout(run, 120);
        this._mhsaRuntimePrewarmHandles.set(idx, { type: 'timeout', handle });
    }

    _clearMhsaRuntimePrewarmHandles() {
        this._mhsaRuntimePrewarmHandles.forEach((entry) => {
            try {
                if (!entry) return;
                if (entry.type === 'idle' && typeof window !== 'undefined' && typeof window.cancelIdleCallback === 'function') {
                    window.cancelIdleCallback(entry.handle);
                    return;
                }
                clearTimeout(entry.handle);
            } catch (_) { /* best-effort cleanup */ }
        });
        this._mhsaRuntimePrewarmHandles.clear();
    }

    _initAutoCameraDriver() {
        // Drive auto-camera follow from the CoreEngine RAF instead of a
        // secondary requestAnimationFrame loop.
        this._autoCameraDriver = {
            isActive: true,
            updateWhenPaused: true,
            update: () => {
                this._autoCamera?.update?.();
            },
            dispose: () => {}
        };
        this._engine._layers.push(this._autoCameraDriver);
    }

    /** Dispose and tear down Three resources */
    dispose() {
        this._clearMhsaRuntimePrewarmHandles();
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

        this._clearKvCacheVisuals();
        if (this._kvCachePersistentRoot && this._kvCachePersistentRoot.parent) {
            this._kvCachePersistentRoot.parent.remove(this._kvCachePersistentRoot);
        }
        if (this._engine && typeof this._engine.removeRaycastRoot === 'function' && this._kvCachePersistentRoot) {
            this._engine.removeRaycastRoot(this._kvCachePersistentRoot);
        }
        this._kvCachePersistentRoot = null;

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

    /** Get current auto-camera semantic view key (e.g., embed-vocab/embed-position). */
    getAutoCameraViewKey() {
        return this._autoCamera?.getViewKey?.() ?? 'default';
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

    setTopLogitRevealPending(pending = true) {
        const shouldWait = !!pending;
        if (!Number.isFinite(this._topLogitRevealGateId)) {
            this._topLogitRevealGateId = 0;
        }
        if (shouldWait) {
            this._topLogitRevealGateId += 1;
        }
        this._awaitingTopLogitReveal = shouldWait;
        this._topLogitRevealComplete = !shouldWait;
        if (shouldWait) {
            this._forwardPassComplete = false;
        }
        return shouldWait ? this._topLogitRevealGateId : null;
    }

    setTopLogitRevealComplete(gateId = null) {
        if (!this._awaitingTopLogitReveal) return;
        if (Number.isFinite(gateId) && gateId !== this._topLogitRevealGateId) return;
        this._topLogitRevealComplete = true;
        this.dispatchEvent(new Event('progress'));
    }

    _normalizeActiveLaneLayoutIndices(indices, laneCount, layoutCount) {
        const activeCount = Math.max(1, Math.floor(laneCount || 1));
        const maxLaneIdx = Math.max(0, Math.floor(layoutCount || activeCount) - 1);
        const out = [];
        if (Array.isArray(indices) && indices.length) {
            for (let i = 0; i < indices.length && out.length < activeCount; i++) {
                const laneIdx = Number.isFinite(indices[i]) ? Math.floor(indices[i]) : 0;
                out.push(Math.max(0, Math.min(maxLaneIdx, laneIdx)));
            }
        }
        while (out.length < activeCount) {
            out.push(Math.max(0, Math.min(maxLaneIdx, out.length)));
        }
        return out;
    }

    _getCachedKvEntriesForLayer(layerIndex) {
        const entries = this._kvCacheEntriesByLayer.get(layerIndex);
        if (!Array.isArray(entries) || !entries.length) return [];
        return entries.map((entry) => ({
            ...entry,
            upwardCopies: Array.isArray(entry.upwardCopies) ? entry.upwardCopies.slice() : [],
            sideCopies: Array.isArray(entry.sideCopies)
                ? entry.sideCopies.map((side) => ({ ...side }))
                : []
        }));
    }

    _getKvBatchStoreKey({ layerIndex = null, headIndex = null, category = 'K', prismCount = VECTOR_LENGTH_PRISM } = {}) {
        const layerPart = Number.isFinite(layerIndex) ? Math.floor(layerIndex) : 'x';
        const headPart = Number.isFinite(headIndex) ? Math.floor(headIndex) : 'x';
        const catPart = String(category || 'K').toUpperCase() === 'V' ? 'V' : 'K';
        const prismPart = Number.isFinite(prismCount) ? Math.max(1, Math.floor(prismCount)) : VECTOR_LENGTH_PRISM;
        return `${layerPart}|${headPart}|${catPart}|${prismPart}`;
    }

    _getOrCreateKvBatchStore({ layerIndex = null, headIndex = null, category = 'K', prismCount = VECTOR_LENGTH_PRISM } = {}) {
        if (!this._kvCachePersistentRoot) return null;
        if (!this._kvCacheBatchStores) this._kvCacheBatchStores = new Map();
        const resolvedCategory = String(category || 'K').toUpperCase() === 'V' ? 'V' : 'K';
        const resolvedPrismCount = Number.isFinite(prismCount)
            ? Math.max(1, Math.floor(prismCount))
            : VECTOR_LENGTH_PRISM;
        const key = this._getKvBatchStoreKey({
            layerIndex,
            headIndex,
            category: resolvedCategory,
            prismCount: resolvedPrismCount
        });
        const existing = this._kvCacheBatchStores.get(key);
        if (existing) return existing;

        const store = {
            key,
            layerIndex: Number.isFinite(layerIndex) ? Math.floor(layerIndex) : null,
            headIndex: Number.isFinite(headIndex) ? Math.floor(headIndex) : null,
            category: resolvedCategory,
            prismCount: resolvedPrismCount,
            chunks: []
        };
        this._kvCacheBatchStores.set(key, store);
        return store;
    }

    _acquireKvBatchSlot(store) {
        if (!store || !this._kvCachePersistentRoot) return null;
        if (!Array.isArray(store.chunks)) store.chunks = [];

        let chunk = store.chunks[store.chunks.length - 1] || null;
        const needsNewChunk = !chunk || !chunk.batch || chunk.used >= chunk.capacity;
        if (needsNewChunk) {
            const capacity = Math.max(1, KV_CACHE_BATCH_CHUNK_SIZE);
            const categoryLabel = store.category === 'V' ? 'Cached Value Vector' : 'Cached Key Vector';
            const batch = new BatchedPrismVectorSet({
                vectorCount: capacity,
                prismCount: store.prismCount,
                parentGroup: this._kvCachePersistentRoot,
                label: categoryLabel,
                raycastMetadataMode: 'perVector'
            });
            if (batch && batch.mesh) {
                batch.mesh.visible = true;
                batch.mesh.userData = batch.mesh.userData || {};
                batch.mesh.userData.skipVisible = true;
                batch.mesh.userData.cachedKv = true;
                batch.mesh.userData.kvCachePersistent = true;
                batch.mesh.userData.vectorCategory = store.category;
                batch.mesh.userData.label = categoryLabel;
                if (Number.isFinite(store.layerIndex)) batch.mesh.userData.layerIndex = store.layerIndex;
                if (Number.isFinite(store.headIndex)) batch.mesh.userData.headIndex = store.headIndex;
                batch.mesh.frustumCulled = false;
                const mats = Array.isArray(batch.mesh.material) ? batch.mesh.material : [batch.mesh.material];
                mats.forEach((mat) => {
                    if (!mat) return;
                    mat.opacity = 1;
                    mat.transparent = false;
                    if ('depthWrite' in mat) mat.depthWrite = true;
                    if ('depthTest' in mat) mat.depthTest = true;
                    if ('alphaTest' in mat) mat.alphaTest = 0;
                    mat.side = THREE.DoubleSide;
                    mat.needsUpdate = true;
                });
            }
            chunk = {
                batch,
                used: 0,
                capacity
            };
            store.chunks.push(chunk);
        }

        if (!chunk || !chunk.batch) return null;
        const vectorIndex = chunk.used++;
        const vec = chunk.batch.getVectorRef(vectorIndex);
        vec.group.visible = true;
        vec.group.userData = vec.group.userData || {};
        vec.group.userData.skipVisible = true;
        vec.group.userData.cachedKv = true;
        vec.group.userData.kvCachePersistent = true;
        vec.group.userData.vectorCategory = store.category;
        vec.group.userData.label = store.category === 'V' ? 'Cached Value Vector' : 'Cached Key Vector';
        vec.userData = vec.userData || {};
        vec.userData.cachedKv = true;
        vec.userData.kvCachePersistent = true;
        vec.userData.vectorCategory = store.category;
        return { vec, batch: chunk.batch, vectorIndex };
    }

    _getKvSourceWorldPosition(vec, out = TMP_KV_CAPTURE_WORLD) {
        if (!vec || !out) return null;
        if (vec.group && vec.group.parent && typeof vec.group.getWorldPosition === 'function') {
            vec.group.parent.updateMatrixWorld?.(true);
            vec.group.getWorldPosition(out);
            return out;
        }
        if (vec.isBatchedVectorRef && vec._batch?.mesh?.parent && vec.group?.position) {
            vec._batch.mesh.parent.updateMatrixWorld?.(true);
            out.copy(vec.group.position);
            vec._batch.mesh.parent.localToWorld(out);
            return out;
        }
        if (vec.group?.position) {
            out.copy(vec.group.position);
            return out;
        }
        return null;
    }

    _markKvBatchVectorMetadata(vec, laneEntry, { category = 'K', headIndex = null, layerIndex = null, batch = null, vectorIndex = null } = {}) {
        if (!vec) return;
        const resolvedCategory = String(category || 'K').toUpperCase() === 'V' ? 'V' : 'K';
        const categoryLabel = resolvedCategory === 'V' ? 'Cached Value Vector' : 'Cached Key Vector';
        vec.userData = vec.userData || {};
        vec.userData.parentLane = laneEntry || null;
        vec.userData.cachedKv = true;
        vec.userData.kvCachePersistent = true;
        vec.userData.vectorCategory = resolvedCategory;
        if (Number.isFinite(headIndex)) vec.userData.headIndex = Math.floor(headIndex);
        if (Number.isFinite(layerIndex)) vec.userData.layerIndex = Math.floor(layerIndex);
        if (Number.isFinite(laneEntry?.laneLayoutIndex)) vec.userData.laneLayoutIndex = laneEntry.laneLayoutIndex;
        if (Number.isFinite(laneEntry?.tokenIndex)) vec.userData.tokenIndex = laneEntry.tokenIndex;

        if (vec.group) {
            vec.group.userData = vec.group.userData || {};
            vec.group.userData.cachedKv = true;
            vec.group.userData.kvCachePersistent = true;
            vec.group.userData.vectorCategory = resolvedCategory;
            vec.group.userData.label = categoryLabel;
            vec.group.userData.skipVisible = true;
            if (Number.isFinite(headIndex)) vec.group.userData.headIndex = Math.floor(headIndex);
            if (Number.isFinite(layerIndex)) vec.group.userData.layerIndex = Math.floor(layerIndex);
            if (Number.isFinite(laneEntry?.laneLayoutIndex)) vec.group.userData.laneLayoutIndex = laneEntry.laneLayoutIndex;
            if (Number.isFinite(laneEntry?.tokenIndex)) vec.group.userData.tokenIndex = laneEntry.tokenIndex;
        }

        if (vec.mesh) {
            vec.mesh.userData = vec.mesh.userData || {};
            vec.mesh.userData.cachedKv = true;
            vec.mesh.userData.kvCachePersistent = true;
            vec.mesh.userData.vectorCategory = resolvedCategory;
            vec.mesh.userData.label = categoryLabel;
            vec.mesh.userData.skipVisible = true;
            if (Number.isFinite(headIndex)) vec.mesh.userData.headIndex = Math.floor(headIndex);
            if (Number.isFinite(layerIndex)) vec.mesh.userData.layerIndex = Math.floor(layerIndex);
        }

        if (batch && Number.isFinite(vectorIndex) && typeof batch.updateVectorRaycastInfo === 'function') {
            batch.updateVectorRaycastInfo(vectorIndex, vec);
        }
    }

    _disposeKvCacheVector(vec) {
        if (!vec) return;
        if (vec.isBatchedVectorRef) {
            try {
                if (vec.group) vec.group.visible = false;
                if (vec.mesh) vec.mesh.visible = false;
            } catch (_) { /* ignore batched cleanup */ }
            return;
        }
        try {
            if (vec.userData && vec.userData.trail && typeof vec.userData.trail.dispose === 'function') {
                vec.userData.trail.dispose();
            }
        } catch (_) { /* optional cleanup */ }
        try {
            if (vec.userData) {
                delete vec.userData.trail;
                delete vec.userData.trailWorld;
            }
        } catch (_) { /* optional cleanup */ }
        try {
            if (vec.group && vec.group.parent) {
                vec.group.parent.remove(vec.group);
            }
        } catch (_) { /* optional cleanup */ }
        try {
            if (typeof vec.dispose === 'function') vec.dispose();
        } catch (_) { /* optional cleanup */ }
    }

    _markKvCacheVectorPersistent(vec, laneEntry, { category = 'K', headIndex = null, layerIndex = null } = {}) {
        if (!vec || !this._kvCachePersistentRoot) return null;
        const resolvedCategory = String(category || 'K').toUpperCase() === 'V' ? 'V' : 'K';
        const prismCount = Number.isFinite(vec.instanceCount)
            ? Math.max(1, Math.floor(vec.instanceCount))
            : VECTOR_LENGTH_PRISM;
        const store = this._getOrCreateKvBatchStore({
            layerIndex,
            headIndex,
            category: resolvedCategory,
            prismCount
        });
        const slot = this._acquireKvBatchSlot(store);
        if (slot && slot.batch && slot.vec) {
            const worldPos = this._getKvSourceWorldPosition(vec, TMP_KV_CAPTURE_WORLD);
            if (worldPos) {
                TMP_KV_CAPTURE_LOCAL.copy(worldPos);
            } else if (vec.group?.position) {
                TMP_KV_CAPTURE_LOCAL.copy(vec.group.position);
            } else {
                TMP_KV_CAPTURE_LOCAL.set(0, 0, 0);
            }
            this._kvCachePersistentRoot.updateMatrixWorld(true);
            this._kvCachePersistentRoot.worldToLocal(TMP_KV_CAPTURE_LOCAL);

            const sourceMesh = vec.isBatchedVectorRef ? vec._batch?.mesh : vec.mesh;
            if (sourceMesh?.updateMatrixWorld) sourceMesh.updateMatrixWorld(true);
            TMP_KV_CAPTURE_INV.copy(this._kvCachePersistentRoot.matrixWorld).invert();

            slot.batch.copyVectorStateFrom(slot.vectorIndex, vec, {
                targetPosition: TMP_KV_CAPTURE_LOCAL,
                sourceMatrixWorld: sourceMesh?.matrixWorld || null,
                targetParentMatrixWorldInverse: TMP_KV_CAPTURE_INV,
                copyData: true
            });
            this._markKvBatchVectorMetadata(slot.vec, laneEntry, {
                category: resolvedCategory,
                headIndex,
                layerIndex,
                batch: slot.batch,
                vectorIndex: slot.vectorIndex
            });
            return slot.vec;
        }

        if (vec && vec.isBatchedVectorRef) return null;
        if (!vec.group) return null;
        try {
            if (vec.group.parent !== this._kvCachePersistentRoot) {
                vec.group.updateMatrixWorld?.(true);
                this._kvCachePersistentRoot.attach(vec.group);
            }
        } catch (_) { /* fallback handled below */ }
        if (vec.group.parent !== this._kvCachePersistentRoot) {
            try { this._kvCachePersistentRoot.add(vec.group); } catch (_) { /* ignore */ }
        }

        try {
            const trail = vec.userData && vec.userData.trail;
            if (trail && typeof trail.dispose === 'function') {
                trail.dispose();
            }
            if (vec.userData) {
                delete vec.userData.trail;
                delete vec.userData.trailWorld;
            }
        } catch (_) { /* optional cleanup */ }

        vec.group.visible = true;
        vec.group.userData = vec.group.userData || {};
        vec.group.userData.skipVisible = true;
        vec.group.userData.kvCachePersistent = true;
        if (Number.isFinite(layerIndex)) {
            vec.group.userData.layerIndex = layerIndex;
        }

        if (vec.mesh) {
            vec.mesh.visible = true;
            vec.mesh.userData = vec.mesh.userData || {};
            vec.mesh.userData.skipVisible = true;
            vec.mesh.userData.kvCachePersistent = true;
            // Cached K/V vectors keep many hidden prism instances far below the
            // tower; frustum bounds can become unstable and cause angle-based
            // popping in decode passes. Keep them always renderable.
            vec.mesh.frustumCulled = false;
            if (vec.mesh.instanceMatrix && typeof vec.mesh.instanceMatrix.setUsage === 'function') {
                vec.mesh.instanceMatrix.setUsage(THREE.StaticDrawUsage);
            }
            normalizeHiddenInstancesForPersistentKv(vec);
            vec.mesh.matrixAutoUpdate = false;
            vec.mesh.updateMatrix();
            const mats = Array.isArray(vec.mesh.material) ? vec.mesh.material : [vec.mesh.material];
            mats.forEach((mat) => {
                if (!mat) return;
                mat.opacity = 1;
                mat.transparent = false;
                if ('depthWrite' in mat) mat.depthWrite = true;
                if ('depthTest' in mat) mat.depthTest = true;
                if ('alphaTest' in mat) mat.alphaTest = 0;
                mat.side = THREE.DoubleSide;
                mat.needsUpdate = true;
            });
        }

        vec.userData = vec.userData || {};
        vec.userData.parentLane = laneEntry;
        vec.userData.kvCachePersistent = true;
        vec.userData.cachedKv = true;
        vec.userData.vectorCategory = category;
        if (Number.isFinite(headIndex)) {
            vec.userData.headIndex = headIndex;
            vec.group.userData.headIndex = headIndex;
            if (vec.mesh) vec.mesh.userData.headIndex = headIndex;
        }

        const categoryLabel = category === 'V' ? 'Cached Value Vector' : 'Cached Key Vector';
        vec.group.userData.cachedKv = true;
        vec.group.userData.label = categoryLabel;
        if (vec.mesh) {
            vec.mesh.userData.cachedKv = true;
            vec.mesh.userData.label = categoryLabel;
        }
        if (Number.isFinite(layerIndex)) vec.group.userData.layerIndex = layerIndex;
        if (Number.isFinite(laneEntry?.laneLayoutIndex)) vec.group.userData.laneLayoutIndex = laneEntry.laneLayoutIndex;
        if (Number.isFinite(laneEntry?.tokenIndex)) vec.group.userData.tokenIndex = laneEntry.tokenIndex;

        this._installKvRaycastProxy(vec, {
            category,
            headIndex,
            layerIndex,
            laneLayoutIndex: laneEntry?.laneLayoutIndex,
            tokenIndex: laneEntry?.tokenIndex
        });

        return vec;
    }

    _installKvRaycastProxy(vec, {
        category = 'K',
        headIndex = null,
        layerIndex = null,
        laneLayoutIndex = null,
        tokenIndex = null
    } = {}) {
        if (!vec || !vec.group || vec.isBatchedVectorRef) return;
        const group = vec.group;
        group.userData = group.userData || {};
        if (group.userData.kvRaycastProxyInstalled) return;

        // Disable expensive raycast checks on the full instanced prism mesh.
        const disableRaycast = (obj) => {
            if (!obj || typeof obj !== 'object') return;
            obj.userData = obj.userData || {};
            obj.userData.raycastDisabled = true;
            obj.raycast = () => {};
        };
        if (vec.mesh) disableRaycast(vec.mesh);
        group.traverse((child) => {
            if (!child || child === group || child === vec.mesh) return;
            if (child.isMesh || child.isLine || child.isPoints) {
                disableRaycast(child);
            }
        });

        // Add one lightweight proxy mesh for raycast interactivity.
        // IMPORTANT: ignore hidden prisms parked at HIDE_INSTANCE_Y_OFFSET so
        // proxies don't balloon and overlap adjacent K/V vectors.
        const bounds = new THREE.Box3();
        group.updateMatrixWorld(true);
        const hasVisibleInstanceBounds = computeVisibleInstancedBoundsLocal(vec, bounds);
        if (!hasVisibleInstanceBounds) {
            bounds.setFromObject(group);
            if (!bounds.isEmpty()) {
                bounds.getCenter(TMP_KV_PROXY_CENTER);
                bounds.getSize(TMP_KV_PROXY_SIZE);
                TMP_KV_PROXY_LOCAL_CENTER.copy(TMP_KV_PROXY_CENTER);
                group.worldToLocal(TMP_KV_PROXY_LOCAL_CENTER);
            } else {
                TMP_KV_PROXY_LOCAL_CENTER.set(0, 0, 0);
                TMP_KV_PROXY_SIZE.set(12, 32, 12);
            }
        } else {
            bounds.getCenter(TMP_KV_PROXY_LOCAL_CENTER);
            bounds.getSize(TMP_KV_PROXY_SIZE);
        }

        const proxy = new THREE.Mesh(KV_RAYCAST_PROXY_GEOMETRY, KV_RAYCAST_PROXY_MATERIAL);
        proxy.name = 'KvCacheRaycastProxy';
        proxy.position.copy(TMP_KV_PROXY_LOCAL_CENTER);
        proxy.scale.set(
            Math.max(0.5, TMP_KV_PROXY_SIZE.x),
            Math.max(0.5, TMP_KV_PROXY_SIZE.y),
            Math.max(0.5, TMP_KV_PROXY_SIZE.z)
        );
        proxy.matrixAutoUpdate = false;
        proxy.updateMatrix();
        proxy.userData = {
            kvRaycastProxy: true,
            cachedKv: true,
            headIndex: Number.isFinite(headIndex) ? headIndex : undefined,
            layerIndex: Number.isFinite(layerIndex) ? layerIndex : undefined,
            laneLayoutIndex: Number.isFinite(laneLayoutIndex) ? laneLayoutIndex : undefined,
            tokenIndex: Number.isFinite(tokenIndex) ? tokenIndex : undefined,
            vectorCategory: category
        };
        group.add(proxy);
        group.userData.kvRaycastProxyInstalled = true;
        group.userData.kvRaycastProxy = proxy;
    }

    _clearKvCacheVisuals() {
        if (this._kvCacheBatchStores && this._kvCacheBatchStores.size) {
            this._kvCacheBatchStores.forEach((store) => {
                if (!store || !Array.isArray(store.chunks)) return;
                store.chunks.forEach((chunk) => {
                    const batch = chunk && chunk.batch ? chunk.batch : null;
                    if (!batch) return;
                    try {
                        if (typeof batch.dispose === 'function') {
                            batch.dispose({ removeFromParent: true });
                        }
                    } catch (_) { /* optional cleanup */ }
                });
            });
        }
        this._kvCacheBatchStores = new Map();

        if (this._kvCacheEntriesByLayer && this._kvCacheEntriesByLayer.size) {
            const disposed = new Set();
            this._kvCacheEntriesByLayer.forEach((entries) => {
                if (!Array.isArray(entries)) return;
                entries.forEach((entry) => {
                    if (!entry) return;
                    const upward = Array.isArray(entry.upwardCopies) ? entry.upwardCopies : [];
                    upward.forEach((vec) => {
                        if (!vec || disposed.has(vec)) return;
                        disposed.add(vec);
                        if (vec.isBatchedVectorRef) return;
                        this._disposeKvCacheVector(vec);
                    });
                    const sideCopies = Array.isArray(entry.sideCopies) ? entry.sideCopies : [];
                    sideCopies.forEach((side) => {
                        const vec = side && side.vec ? side.vec : null;
                        if (!vec || disposed.has(vec)) return;
                        disposed.add(vec);
                        if (vec.isBatchedVectorRef) return;
                        this._disposeKvCacheVector(vec);
                    });
                });
            });
        }
        this._kvCacheEntriesByLayer = new Map();
        if (this._kvCachePersistentRoot && Array.isArray(this._kvCachePersistentRoot.children)) {
            const leftovers = [...this._kvCachePersistentRoot.children];
            leftovers.forEach((child) => {
                if (!child) return;
                try {
                    child.traverse?.((obj) => {
                        if (!obj) return;
                        if (obj.geometry && typeof obj.geometry.dispose === 'function') {
                            obj.geometry.dispose();
                        }
                        if (obj.material) {
                            const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
                            mats.forEach((mat) => {
                                if (mat && typeof mat.dispose === 'function') mat.dispose();
                            });
                        }
                    });
                } catch (_) { /* optional cleanup */ }
                try {
                    if (child.parent) child.parent.remove(child);
                } catch (_) { /* optional cleanup */ }
            });
        }
    }

    _captureKvCacheFromLayers(layers) {
        if (!Array.isArray(layers) || !layers.length || !this._kvCachePersistentRoot) return;

        layers.forEach((layer) => {
            if (!layer || !Array.isArray(layer.lanes) || !layer.lanes.length) return;
            const layerIndex = Number.isFinite(layer.index) ? layer.index : null;
            if (!Number.isFinite(layerIndex)) return;
            const existing = Array.isArray(this._kvCacheEntriesByLayer.get(layerIndex))
                ? this._kvCacheEntriesByLayer.get(layerIndex).slice()
                : [];

            layer.lanes.forEach((lane) => {
                if (!lane) return;
                const tokenIndex = Number.isFinite(lane.tokenIndex) ? lane.tokenIndex : null;
                const laneLayoutIndex = Number.isFinite(lane.laneLayoutIndex)
                    ? lane.laneLayoutIndex
                    : null;
                if (this._hasMatchingKvLaneEntry(existing, { laneLayoutIndex, tokenIndex })) return;

                const zPos = Number.isFinite(lane.zPos)
                    ? lane.zPos
                    : (lane.originalVec?.group?.position?.z ?? 0);
                const laneEntry = this._createKvLaneEntry({
                    layerIndex,
                    laneIndex: Number.isFinite(lane.laneIndex) ? lane.laneIndex : 0,
                    laneLayoutIndex,
                    tokenIndex,
                    tokenLabel: lane.tokenLabel || null,
                    zPos
                });

                if (Array.isArray(lane.upwardCopies)) {
                    lane.upwardCopies.forEach((kVec, headIndex) => {
                        if (!kVec) return;
                        const persisted = this._markKvCacheVectorPersistent(
                            kVec,
                            laneEntry,
                            { category: 'K', headIndex, layerIndex }
                        );
                        laneEntry.upwardCopies[headIndex] = persisted;
                    });
                }

                if (Array.isArray(lane.sideCopies)) {
                    lane.sideCopies.forEach((sideCopy) => {
                        if (!sideCopy || sideCopy.type !== 'V' || !sideCopy.vec) return;
                        const headIndex = Number.isFinite(sideCopy.headIndex) ? sideCopy.headIndex : null;
                        const persisted = this._markKvCacheVectorPersistent(
                            sideCopy.vec,
                            laneEntry,
                            { category: 'V', headIndex, layerIndex }
                        );
                        if (!persisted) return;
                        laneEntry.sideCopies.push({
                            type: 'V',
                            headIndex,
                            vec: persisted,
                            targetX: Number.isFinite(sideCopy.targetX)
                                ? sideCopy.targetX
                                : persisted.group.position.x,
                            matrixRef: null
                        });
                    });
                }

                const hasK = laneEntry.upwardCopies.some((vec) => !!vec);
                const hasV = laneEntry.sideCopies.length > 0;
                if (!hasK && !hasV) return;
                existing.push(laneEntry);
            });

            this._kvCacheEntriesByLayer.set(layerIndex, this._sortKvEntries(existing));
        });
    }

    _reflowKvCacheLaneDepths(layoutCount = this._laneLayoutCount) {
        const totalSlots = Math.max(1, Math.floor(layoutCount || 1));
        const spacing = LN_PARAMS.depth / (totalSlots + 1);
        const touchedBatches = new Set();
        this._kvCacheEntriesByLayer.forEach((entries) => {
            if (!Array.isArray(entries) || !entries.length) return;
            entries.forEach((entry) => {
                if (!entry || !Number.isFinite(entry.laneLayoutIndex)) return;
                const clampedLayoutIdx = Math.max(0, Math.min(totalSlots - 1, Math.floor(entry.laneLayoutIndex)));
                const targetZ = -LN_PARAMS.depth / 2 + spacing * (clampedLayoutIdx + 1);
                entry.zPos = targetZ;
                const applyZ = (vec) => {
                    if (!vec || !vec.group || !vec.group.position) return;
                    vec.group.position.z = targetZ;
                    if (vec.isBatchedVectorRef && vec._batch) {
                        touchedBatches.add(vec._batch);
                    }
                };
                if (Array.isArray(entry.upwardCopies)) {
                    entry.upwardCopies.forEach((vec) => applyZ(vec));
                }
                if (Array.isArray(entry.sideCopies)) {
                    entry.sideCopies.forEach((side) => applyZ(side?.vec));
                }
            });
        });
        touchedBatches.forEach((batch) => {
            if (!batch || typeof batch.syncAll !== 'function') return;
            try {
                batch.syncAll();
            } catch (_) { /* optional cleanup */ }
        });
    }

    _hasMatchingKvLaneEntry(entries, { laneLayoutIndex = null, tokenIndex = null } = {}) {
        if (!Array.isArray(entries) || !entries.length) return false;
        const hasLayout = Number.isFinite(laneLayoutIndex);
        const hasToken = Number.isFinite(tokenIndex);
        return entries.some((entry) => {
            if (!entry) return false;
            const entryLayout = Number.isFinite(entry.laneLayoutIndex)
                ? Math.floor(entry.laneLayoutIndex)
                : null;
            const entryToken = Number.isFinite(entry.tokenIndex)
                ? Math.floor(entry.tokenIndex)
                : null;
            if (hasLayout && Number.isFinite(entryLayout) && entryLayout === Math.floor(laneLayoutIndex)) return true;
            if (hasToken && Number.isFinite(entryToken) && entryToken === Math.floor(tokenIndex)) return true;
            return false;
        });
    }

    _sortKvEntries(entries) {
        if (!Array.isArray(entries)) return [];
        entries.sort((a, b) => {
            const aLayout = Number.isFinite(a?.laneLayoutIndex) ? a.laneLayoutIndex : Infinity;
            const bLayout = Number.isFinite(b?.laneLayoutIndex) ? b.laneLayoutIndex : Infinity;
            if (aLayout !== bLayout) return aLayout - bLayout;
            const aToken = Number.isFinite(a?.tokenIndex) ? a.tokenIndex : Infinity;
            const bToken = Number.isFinite(b?.tokenIndex) ? b.tokenIndex : Infinity;
            if (aToken !== bToken) return aToken - bToken;
            const aZ = Number.isFinite(a?.zPos) ? a.zPos : 0;
            const bZ = Number.isFinite(b?.zPos) ? b.zPos : 0;
            return aZ - bZ;
        });
        return entries;
    }

    _computeKvLaneDepthZ(laneLayoutIndex, layoutCount = this._laneLayoutCount) {
        const totalSlots = Math.max(1, Math.floor(layoutCount || 1));
        const clampedLayoutIdx = Math.max(0, Math.min(totalSlots - 1, Math.floor(laneLayoutIndex || 0)));
        const spacing = LN_PARAMS.depth / (totalSlots + 1);
        return -LN_PARAMS.depth / 2 + spacing * (clampedLayoutIdx + 1);
    }

    _createKvLaneEntry({
        layerIndex,
        laneIndex = 0,
        laneLayoutIndex = null,
        tokenIndex = null,
        tokenLabel = null,
        zPos = 0,
        bootstrapFromActivation = false
    } = {}) {
        const entry = {
            layer: Number.isFinite(layerIndex) ? { index: layerIndex } : null,
            layerIndex: Number.isFinite(layerIndex) ? Math.floor(layerIndex) : null,
            laneIndex: Number.isFinite(laneIndex) ? Math.floor(laneIndex) : 0,
            laneLayoutIndex: Number.isFinite(laneLayoutIndex) ? Math.floor(laneLayoutIndex) : null,
            tokenIndex: Number.isFinite(tokenIndex) ? Math.floor(tokenIndex) : null,
            tokenLabel: tokenLabel || null,
            zPos: Number.isFinite(zPos) ? zPos : 0,
            upwardCopies: [],
            sideCopies: []
        };
        if (bootstrapFromActivation) {
            entry.bootstrapFromActivation = true;
        }
        return entry;
    }

    _toKvPersistentLocalPosition({ layer, mhsa, x = 0, y = 0, z = 0 } = {}) {
        TMP_KV_BOOTSTRAP_LOCAL.set(x, y, z);
        TMP_KV_BOOTSTRAP_WORLD.copy(TMP_KV_BOOTSTRAP_LOCAL);

        const localParent = mhsa?.parentGroup || layer?.raycastRoot || layer?.root || null;
        if (localParent && typeof localParent.localToWorld === 'function') {
            localParent.updateMatrixWorld?.(true);
            localParent.localToWorld(TMP_KV_BOOTSTRAP_WORLD);
        }

        if (this._kvCachePersistentRoot && typeof this._kvCachePersistentRoot.worldToLocal === 'function') {
            this._kvCachePersistentRoot.updateMatrixWorld?.(true);
            this._kvCachePersistentRoot.worldToLocal(TMP_KV_BOOTSTRAP_WORLD);
        }

        // Return an owned vector so callers can safely keep both K and V
        // positions in the same scope without temp-vector aliasing.
        return TMP_KV_BOOTSTRAP_WORLD.clone();
    }

    _resolveKvBootstrapAnchor(mhsa) {
        const prismCount = Number.isFinite(mhsa?.vectorPrismCount)
            ? Math.max(1, Math.floor(mhsa.vectorPrismCount))
            : VECTOR_LENGTH_PRISM;
        const outputUnits = Number.isFinite(mhsa?.outputVectorLength)
            ? Math.max(1, Math.floor(mhsa.outputVectorLength))
            : 64;
        // Match the settled post-pass-through height so bootstrap vectors start
        // where persistent cached vectors normally reside.
        const postPassY = (Number.isFinite(mhsa?.mhaPassThroughTargetY) && Number.isFinite(mhsa?.mhaResultRiseOffsetY))
            ? (mhsa.mhaPassThroughTargetY + mhsa.mhaResultRiseOffsetY - 30)
            : (Number.isFinite(mhsa?.headStopY) ? mhsa.headStopY : 0);
        const selfAttentionEnabled = !!mhsa?.enableSelfAttentionAnimation;
        const valueRise = selfAttentionEnabled && Number.isFinite(SA_RED_EXTRA_RISE)
            ? SA_RED_EXTRA_RISE
            : 0;
        // Decode cache visuals mirror preserved KV pose:
        // K snapped under V column; V optionally raised in self-attention mode.
        return {
            prismCount,
            outputUnits,
            keyY: postPassY,
            valueY: postPassY + valueRise
        };
    }

    _resolveKvBootstrapHeadPose({ layer, mhsa, head, zPos, anchor }) {
        const targetValueX = Number.isFinite(head?.v) ? head.v : (Number.isFinite(head?.k) ? head.k : 0);
        // Decode rendering snaps cached keys under value columns.
        const targetKeyX = targetValueX;
        return {
            targetValueX,
            keyPosition: this._toKvPersistentLocalPosition({
                layer,
                mhsa,
                x: targetKeyX,
                y: anchor?.keyY,
                z: zPos
            }),
            valuePosition: this._toKvPersistentLocalPosition({
                layer,
                mhsa,
                x: targetValueX,
                y: anchor?.valueY,
                z: zPos
            })
        };
    }

    resetForNewPass({
        activationSource = this._activationSource,
        laneCount = this._laneCount,
        laneLayoutCount = this._laneLayoutCount,
        laneLayoutIndices = null,
        laneTokenIndices = null,
        kvCacheModeEnabled = false,
        kvCacheDecodeActive = false,
        preservePreviousTrails = false,
        captureKvCache = false,
        reuseKvCache = false,
        clearKvCache = false,
        bootstrapKvCacheFromActivation = false
    } = {}) {
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
        this._awaitingTopLogitReveal = false;
        this._topLogitRevealComplete = true;
        this._topLogitRevealGateId = Number.isFinite(this._topLogitRevealGateId)
            ? this._topLogitRevealGateId + 1
            : 1;
        this._skipKvCaptureReadyLayers.clear();
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

        const shouldPreserveTrails = !!preservePreviousTrails;
        const shouldCaptureKv = !!captureKvCache;
        const shouldReuseKv = !!reuseKvCache;
        const shouldClearKv = !!clearKvCache;

        const oldLayers = Array.isArray(this._layers) ? this._layers : [];
        if (shouldClearKv) {
            this._clearKvCacheVisuals();
        }
        if (shouldCaptureKv) {
            this._captureKvCacheFromLayers(oldLayers);
        }

        if (!shouldPreserveTrails) {
            this._trailPassId = (Number.isFinite(this._trailPassId) ? this._trailPassId : 0) + 1;
        }
        this._topLnParamPlaceholders = null;
        if (engine && engine.scene && engine.scene.userData) {
            engine.scene.userData.trailPassId = this._trailPassId;
        }
        if (engine && engine.scene && !shouldPreserveTrails) {
            clearTrailsFromScene(engine.scene, { passId: this._trailPassId });
        }

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
        this._laneLayoutCount = Math.max(this._laneCount, Math.floor(laneLayoutCount || this._laneCount));
        this._activeLaneLayoutIndices = this._normalizeActiveLaneLayoutIndices(
            laneLayoutIndices,
            this._laneCount,
            this._laneLayoutCount
        );
        this._passLaneTokenIndices = Array.isArray(laneTokenIndices)
            ? laneTokenIndices.slice(0, this._laneCount)
            : null;
        this._kvCacheModeEnabled = !!kvCacheModeEnabled;
        this._kvCacheDecodeActive = !!(this._kvCacheModeEnabled && kvCacheDecodeActive);
        this._reuseKvCacheForPass = !!(this._kvCacheModeEnabled && shouldReuseKv);
        const shouldBootstrapKv = !!(this._kvCacheModeEnabled && this._kvCacheDecodeActive && bootstrapKvCacheFromActivation);
        if (this._kvCacheEntriesByLayer.size) {
            this._reflowKvCacheLaneDepths(this._laneLayoutCount);
        }

        const layerStackSpacing = LAYER_STACK_SPACING_Y;
        this._initLayers(layerStackSpacing);
        if (shouldBootstrapKv) {
            this._bootstrapKvCacheEntriesFromActivation();
        }
        if (engine && Array.isArray(engine._layers) && autoDriver && !engine._layers.includes(autoDriver)) {
            engine._layers.push(autoDriver);
        }
        if (!shouldPreserveTrails) {
            this._schedulePostResetTrailPurge(1500);
        }
        this.dispatchEvent(new Event('passreset'));
        this.dispatchEvent(new Event('progress'));
    }

    _bootstrapKvCacheEntriesFromActivation() {
        if (!this._activationSource || !this._reuseKvCacheForPass) return false;
        if (!this._kvCacheModeEnabled || !this._kvCacheDecodeActive) return false;
        if (!this._kvCachePersistentRoot) return false;
        if (!Array.isArray(this._layers) || !this._layers.length) return false;

        const totalLaneCount = Math.max(1, Math.floor(this._laneLayoutCount || this._laneCount || 1));
        // Decode mode renders only the active token lane; all previous lanes are
        // synthetic cached history that we seed from activation data.
        const cachedHistoryLaneCount = Math.max(0, totalLaneCount - 1);
        if (cachedHistoryLaneCount <= 0) return false;

        const source = this._activationSource;
        const laneTokenIndices = (source && typeof source.getLaneTokenIndices === 'function')
            ? source.getLaneTokenIndices(totalLaneCount)
            : Array.from({ length: totalLaneCount }, (_, idx) => idx);
        const touchedBatches = new Set();
        let seededAny = false;
        const keyRangeOptions = buildHueRangeOptions(MHA_FINAL_K_COLOR, {
            hueSpread: MHA_VALUE_HUE_SPREAD,
            minLightness: MHA_VALUE_LIGHTNESS_MIN,
            maxLightness: MHA_VALUE_LIGHTNESS_MAX,
            valueMin: MHA_VALUE_RANGE_MIN,
            valueMax: MHA_VALUE_RANGE_MAX,
            valueClampMax: MHA_VALUE_CLAMP_MAX
        });
        const valueRangeOptions = buildHueRangeOptions(MHA_VALUE_SPECTRUM_COLOR, {
            hueSpread: MHA_VALUE_HUE_SPREAD,
            minLightness: MHA_VALUE_LIGHTNESS_MIN,
            maxLightness: MHA_VALUE_LIGHTNESS_MAX,
            valueMin: MHA_VALUE_RANGE_MIN,
            valueMax: MHA_VALUE_RANGE_MAX,
            valueClampMax: MHA_VALUE_CLAMP_MAX
        });

        const getScalarBootstrapData = (layerIndex, kind, headIndex, tokenIndex, length) => {
            const scalar = (source && typeof source.getLayerQKVScalar === 'function')
                ? source.getLayerQKVScalar(layerIndex, kind, headIndex, tokenIndex)
                : null;
            const outLength = Math.max(1, Math.floor(length || VECTOR_LENGTH_PRISM));
            if (Number.isFinite(scalar)) {
                return { scalar, data: [scalar] };
            }
            return { scalar: null, data: new Array(outLength).fill(0) };
        };

        const seedVector = ({
            layerIndex,
            headIndex,
            category,
            laneEntry,
            position,
            data,
            prismCount = null,
            scalarValue = null,
            visibleOutputUnits = 64,
            colorGenerationOptions = null
        }) => {
            const resolvedPrismCount = Number.isFinite(prismCount)
                ? Math.max(1, Math.floor(prismCount))
                : Math.max(1, Math.floor(data?.length || VECTOR_LENGTH_PRISM));
            const store = this._getOrCreateKvBatchStore({
                layerIndex,
                headIndex,
                category,
                prismCount: resolvedPrismCount
            });
            const slot = this._acquireKvBatchSlot(store);
            if (!slot || !slot.vec) return null;

            const safeVisibleUnits = Math.max(1, Math.min(resolvedPrismCount, Math.floor(visibleOutputUnits || 64)));
            if (typeof slot.vec.applyProcessedVisuals === 'function') {
                const hasScalar = Number.isFinite(scalarValue);
                const numKeyColors = hasScalar ? 1 : Math.min(30, Math.max(1, data?.length || 1));
                slot.vec.applyProcessedVisuals(
                    data,
                    safeVisibleUnits,
                    { numKeyColors, generationOptions: colorGenerationOptions },
                    { setHiddenToBlack: false, hideByScaleOnly: true },
                    data
                );
            } else {
                slot.vec.updateDataInternal(data, { copyData: true });
            }
            if (Number.isFinite(scalarValue)
                && colorGenerationOptions
                && typeof slot.vec.setUniformColor === 'function') {
                slot.vec.setUniformColor(mapValueToHueRange(scalarValue, colorGenerationOptions));
            }
            slot.vec.group.position.copy(position);
            slot.vec.group.visible = true;

            this._markKvBatchVectorMetadata(slot.vec, laneEntry, {
                category,
                headIndex,
                layerIndex,
                batch: slot.batch,
                vectorIndex: slot.vectorIndex
            });

            if (slot.batch && typeof slot.batch.syncAll === 'function') {
                touchedBatches.add(slot.batch);
            }
            return slot.vec;
        };

        this._layers.forEach((layer) => {
            if (!layer) return;
            const layerIndex = Number.isFinite(layer.index) ? Math.floor(layer.index) : null;
            if (!Number.isFinite(layerIndex)) return;
            const mhsa = layer.mhsaAnimation;
            const headCoords = Array.isArray(mhsa?.headCoords) ? mhsa.headCoords : [];
            if (!headCoords.length) return;
            const existingEntries = Array.isArray(this._kvCacheEntriesByLayer.get(layerIndex))
                ? this._kvCacheEntriesByLayer.get(layerIndex).slice()
                : [];
            const bootstrapAnchor = this._resolveKvBootstrapAnchor(mhsa);

            for (let laneLayoutIndex = 0; laneLayoutIndex < cachedHistoryLaneCount; laneLayoutIndex++) {
                const tokenIndex = Number.isFinite(laneTokenIndices?.[laneLayoutIndex])
                    ? Math.max(0, Math.floor(laneTokenIndices[laneLayoutIndex]))
                    : laneLayoutIndex;
                if (this._hasMatchingKvLaneEntry(existingEntries, { laneLayoutIndex, tokenIndex })) continue;
                const tokenLabel = (source && typeof source.getTokenString === 'function')
                    ? source.getTokenString(tokenIndex)
                    : null;
                const zPos = this._computeKvLaneDepthZ(laneLayoutIndex, totalLaneCount);

                const laneEntry = this._createKvLaneEntry({
                    layerIndex,
                    laneIndex: laneLayoutIndex,
                    laneLayoutIndex,
                    tokenIndex,
                    tokenLabel,
                    zPos,
                    bootstrapFromActivation: true
                });

                for (let headIndex = 0; headIndex < headCoords.length; headIndex++) {
                    const head = headCoords[headIndex] || {};
                    const pose = this._resolveKvBootstrapHeadPose({
                        layer,
                        mhsa,
                        head,
                        zPos,
                        anchor: bootstrapAnchor
                    });

                    const kBootstrap = getScalarBootstrapData(
                        layerIndex,
                        'k',
                        headIndex,
                        tokenIndex,
                        bootstrapAnchor.prismCount
                    );
                    const kVec = seedVector({
                        layerIndex,
                        headIndex,
                        category: 'K',
                        laneEntry,
                        position: pose.keyPosition,
                        data: kBootstrap.data,
                        prismCount: bootstrapAnchor.prismCount,
                        scalarValue: kBootstrap.scalar,
                        visibleOutputUnits: bootstrapAnchor.outputUnits,
                        colorGenerationOptions: keyRangeOptions
                    });
                    if (kVec) laneEntry.upwardCopies[headIndex] = kVec;

                    const vBootstrap = getScalarBootstrapData(
                        layerIndex,
                        'v',
                        headIndex,
                        tokenIndex,
                        bootstrapAnchor.prismCount
                    );
                    const vVec = seedVector({
                        layerIndex,
                        headIndex,
                        category: 'V',
                        laneEntry,
                        position: pose.valuePosition,
                        data: vBootstrap.data,
                        prismCount: bootstrapAnchor.prismCount,
                        scalarValue: vBootstrap.scalar,
                        visibleOutputUnits: bootstrapAnchor.outputUnits,
                        colorGenerationOptions: valueRangeOptions
                    });
                    if (vVec) {
                        laneEntry.sideCopies.push({
                            type: 'V',
                            headIndex,
                            vec: vVec,
                            targetX: pose.targetValueX,
                            matrixRef: null
                        });
                    }
                }

                const hasK = laneEntry.upwardCopies.some((vec) => !!vec);
                const hasV = laneEntry.sideCopies.length > 0;
                if (hasK || hasV) {
                    seededAny = true;
                    existingEntries.push(laneEntry);
                }
            }

            if (!existingEntries.length) return;
            this._kvCacheEntriesByLayer.set(layerIndex, this._sortKvEntries(existingEntries));
        });

        touchedBatches.forEach((batch) => {
            try { batch.syncAll(); } catch (_) { /* optional bootstrap sync */ }
        });

        if (!this._kvCacheEntriesByLayer.size) return false;

        this._layers.forEach((layer) => {
            const idx = Number.isFinite(layer?.index) ? Math.floor(layer.index) : null;
            if (!Number.isFinite(idx)) return;
            const entries = this._getCachedKvEntriesForLayer(idx);
            if (layer?.mhsaAnimation && typeof layer.mhsaAnimation.setCachedKvEntries === 'function') {
                layer.mhsaAnimation.setCachedKvEntries(entries);
            }
        });

        return seededAny;
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
        let nextSweepAt = start;
        let zeroSweepCount = 0;

        const tick = () => {
            if (!engine.scene) return;
            const now = (typeof performance !== 'undefined' && performance.now)
                ? performance.now()
                : Date.now();
            if (now >= nextSweepAt) {
                const removedCount = clearTrailsFromScene(engine.scene, { passId: this._trailPassId });
                if (removedCount > 0) {
                    zeroSweepCount = 0;
                } else {
                    zeroSweepCount += 1;
                }
                nextSweepAt = now + POST_RESET_TRAIL_PURGE_INTERVAL_MS;
            }
            if (now >= this._postResetTrailPurgeUntil) {
                this._postResetTrailPurgeRaf = null;
                return;
            }
            // Stop early once multiple sweeps find nothing left to purge.
            if (zeroSweepCount >= 4) {
                this._postResetTrailPurgeRaf = null;
                return;
            }
            this._postResetTrailPurgeRaf = schedule(tick);
        };

        this._postResetTrailPurgeRaf = schedule(tick);
    }

    skipCurrentLayer(opts = {}) {
        if (this._skipLayerActive || this._skipToEndActive) return false;
        if (this._checkForwardPassComplete()) return false;

        const layer = this._layers[this._currentLayerIdx];
        if (!layer || layer._completed) return false;

        const engine = this._engine;
        if (engine && typeof engine.resume === 'function') {
            engine.resume('manual');
        }
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
        // Keep trail geometry behavior consistent with non-skip playback.
        setGlobalTrailMaxStepDistance(0);
        refreshTrailDisplayScales(this._engine?.scene);

        if (layer && typeof layer.setSkipToEndMode === 'function') {
            layer.setSkipToEndMode(true);
        }

        return true;
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
        // Keep trail geometry behavior consistent with non-skip playback.
        setGlobalTrailMaxStepDistance(0);
        refreshTrailDisplayScales(this._engine?.scene);

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

        if (this._awaitingTopLogitReveal && !this._topLogitRevealComplete) return false;

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
            this._captureSkipReadyKvLayers();
            if (this._checkForwardPassComplete()) {
                this._finalizeSkipToEnd();
                return;
            }
            this._skipToEndRaf = schedule(tick);
        };
        this._skipToEndRaf = schedule(tick);
    }

    _captureSkipReadyKvLayers() {
        if (!this._kvCacheModeEnabled || !Array.isArray(this._layers) || !this._layers.length) return;
        this._layers.forEach((layer) => {
            if (!layer) return;
            const layerIndex = Number.isFinite(layer.index) ? Math.floor(layer.index) : null;
            if (!Number.isFinite(layerIndex) || this._skipKvCaptureReadyLayers.has(layerIndex)) return;
            const mhsa = layer.mhsaAnimation;
            const passThroughComplete = !!(mhsa && mhsa.mhaPassThroughPhase === 'mha_pass_through_complete');
            if (!passThroughComplete) return;
            // During skip-to-end we must wait until the layer has transitioned
            // into skip-concat handling; otherwise K vectors can still be parked
            // above the key matrix when we capture KV cache.
            if (this._skipToEndActive && !layer._skipConcatTriggered) return;
            try {
                if (mhsa && typeof mhsa._preserveKVVectorsForCache === 'function') {
                    mhsa._preserveKVVectorsForCache();
                }
            } catch (_) { /* optional pre-capture alignment */ }
            this._captureKvCacheFromLayers([layer]);
            this._skipKvCaptureReadyLayers.add(layerIndex);
        });
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
        if (this._currentLayerIdx >= this._numLayers) {
            this.dispatchEvent(new Event('progress'));
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
        this._scheduleMhsaRuntimePrewarm(this._currentLayerIdx + 1);

        this._autoCamera?.maybeFocus?.({ immediate: true });
        this.dispatchEvent(new Event('progress'));

        if (this._skipLayerActive && !this._skipLayerLast) {
            this._restoreSkipLayerSpeeds();
        }

        // Now that the original residual vectors have been transferred, we can safely
        // hide the remaining heavy geometry in the previous layer to save GPU work.
        if (this._kvCacheModeEnabled && prevLayer) {
            this._captureKvCacheFromLayers([prevLayer]);
        }
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
        const embedHeight = EMBEDDING_MATRIX_PARAMS_VOCAB.height;
        const embedInset = 5;
        const { targetYLocal, exitYLocal } = calculateTopEmbeddingTargets({
            engineScene: this._engine && this._engine.scene,
            lastLayer,
            mlpDownHeight: MLP_MATRIX_PARAMS_DOWN.height,
            embedHeight,
            embedInset,
            topEmbedGap: TOP_EMBED_Y_GAP_ABOVE_TOWER,
            topEmbedAdjust: TOP_EMBED_Y_ADJUST,
            maxRiseFraction: TOP_EMBED_MAX_RISE_FRACTION
        });

        try {
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
        return findTopLayerNormInfo({
            engineScene: this._engine && this._engine.scene,
            lastLayer,
            lnHeight: LN_PARAMS.height
        });
    }

    /**
     * Apply the bright activated appearance to a LayerNorm group.
     * @param {THREE.Object3D} lnTopGroup
     */
    _activateLayerNormColor(lnTopGroup) {
        activateTopLayerNormColor(lnTopGroup, {
            emissiveIntensity: 0.5,
            scaleEmissiveIntensity: scaleGlobalEmissiveIntensity
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
                .easing(TWEEN.Easing.Linear.None)
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

            const lnColorState = {
                highestY: -Infinity,
                locked: false,
                lockedColor: new THREE.Color(COLOR_BRIGHT_YELLOW),
                currentColor: new THREE.Color(COLOR_DARK_GRAY),
                currentOpacity: 1.0
            };
            const lnMaterialState = {
                color: new THREE.Color(COLOR_DARK_GRAY),
                opacity: 1.0,
                transparent: false,
                initialized: false
            };
            const tempColor = new THREE.Color();
            const applyTopLnColor = () => {
                applyLayerNormMaterial(lnTopGroup, lnColorState.currentColor, lnColorState.currentOpacity, lnMaterialState);
            };

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
                // Keep the original LN appearance untouched until vectors
                // actually enter the top-LN volume.
                if (!lnColorState.locked && highest < lnBottomY) return;
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
                    // Top LayerNorm should continue only the canonical residual
                    // world-space trail to avoid extending stale local trails.
                    if (vec.userData.trail && vec.userData.trail !== lane.originalTrail) {
                        delete vec.userData.trail;
                        delete vec.userData.trailWorld;
                    }
                    vec.userData.trail = lane.originalTrail;
                    vec.userData.trailWorld = true;
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
                const finalNormStageDataRaw = getFinalLnStageData('norm', lane, targetLength);
                const finalNormStageData = (isArrayLike(finalNormStageDataRaw) && finalNormStageDataRaw.length)
                    ? finalNormStageDataRaw
                    : null;

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
                    activateFinalLnParamColors();
                };

                const normAnim = new PrismLayerNormAnimation(vec);
                let normLoopActive = false;

                if (startY >= lnBottomY) {
                    markTopLnEntered();
                }

                const applyFinalNormData = () => {
                    if (finalNormStageData) {
                        recolorVectorFromData(vec, finalNormStageData, null);
                        return;
                    }
                    const runtimeNorm = (vec && isArrayLike(vec.normalizedData) && vec.normalizedData.length)
                        ? vec.normalizedData
                        : null;
                    if (runtimeNorm) {
                        recolorVectorFromData(vec, runtimeNorm, null);
                    }
                };

                const startFinalRise = (resVec) => {
                    const riseDist = Math.max(0, entryYLocal - resVec.group.position.y);
                    const durMs = (riseDist / (ANIM_RISE_SPEED_ORIGINAL * GLOBAL_ANIM_SPEED_MULT)) * 1000;
                    new TWEEN.Tween(resVec.group.position)
                        .to({ y: entryYLocal }, Math.max(100, durMs))
                        .easing(TWEEN.Easing.Linear.None)
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
                        const finalLnScaledDataRaw = getFinalLnStageData('scale', lane, multVec.instanceCount);
                        const finalLnScaledData = (isArrayLike(finalLnScaledDataRaw) && finalLnScaledDataRaw.length)
                            ? finalLnScaledDataRaw
                            : null;
                        const productData = finalLnScaledData
                            ? finalLnScaledData
                            : multVec.rawData.slice();

                        const resVec = new VectorVisualizationInstancedPrism(
                            productData,
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
                        .easing(TWEEN.Easing.Linear.None)
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
                        applyFinalNormData();
                        riseToCenter();
                        return;
                    }
                    normLoopActive = true;
                    try {
                        const normInput = finalNormStageData ? finalNormStageData.slice() : vec.rawData.slice();
                        normAnim.start(normInput, {
                            deferDataUpdate: true,
                            sourceAlreadyNormalized: !!finalNormStageData
                        });
                    } catch (_) {
                        normLoopActive = false;
                        riseToCenter();
                        return;
                    }

                    let lastTickMs = null;
                    const runLoop = (frameNow) => {
                        if (this._skipToEndActive) {
                            normAnim.isAnimating = false;
                            normLoopActive = false;
                            riseToCenter();
                            return;
                        }
                        const nowMs = Number.isFinite(frameNow)
                            ? frameNow
                            : ((typeof performance !== 'undefined' && typeof performance.now === 'function')
                                ? performance.now()
                                : Date.now());
                        if (this._engine && this._engine._paused) {
                            // Keep RAF alive while paused, but pin the last tick so
                            // wall-clock pause time does not advance LN animation.
                            lastTickMs = nowMs;
                            requestAnimationFrame(runLoop);
                            return;
                        }
                        const dt = Number.isFinite(lastTickMs)
                            ? Math.max(0, Math.min((nowMs - lastTickMs) / 1000, 0.1))
                            : 0;
                        lastTickMs = nowMs;
                        normAnim.update(dt);
                        updateTopLnColor(vec.group.position.y);
                        this.dispatchEvent(new Event('progress'));
                        if (normAnim.isAnimating) {
                            requestAnimationFrame(runLoop);
                        } else {
                            applyFinalNormData();
                            normLoopActive = false;
                            riseToCenter();
                        }
                    };
                    requestAnimationFrame(runLoop);
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
                        .easing(TWEEN.Easing.Linear.None)
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
