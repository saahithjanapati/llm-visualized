import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { QUALITY_PRESET, resolveRenderPixelRatio } from '../utils/constants.js';
import { perfStats } from '../utils/perfStats.js';
import { refreshTrailDisplayScales } from '../utils/trailUtils.js';
import { TRAIL_LINE_WIDTH, TRAIL_OPACITY, scaleLineWidthForDisplay, scaleOpacityForDisplay } from '../utils/trailConstants.js';
import Gpt2Layer from './layers/Gpt2Layer.js';
import { resolveRaycastLabel as resolveRaycastLabelFromIntersections } from './coreRaycastResolver.js';
import {
    isCachedKvSelection,
    normalizeRaycastLabel,
    simplifyLayerNormParamHoverLabel
} from './coreRaycastLabels.js';

const ZOOM_OUT_SUPERSAMPLE_MIN_DISTANCE_RATIO = 0.72;
const ZOOM_OUT_SUPERSAMPLE_MAX_DISTANCE_RATIO = 0.98;
const ZOOM_OUT_SUPERSAMPLE_MAX_MULTIPLIER = 1.22;
const ZOOM_OUT_SUPERSAMPLE_MAX_DPR = 2.6;
const ZOOM_OUT_SUPERSAMPLE_RATIO_STEP = 0.05;
const ZOOM_OUT_SUPERSAMPLE_DEBOUNCE_MS = 140;

/**
 * CoreEngine is responsible for creating the Three-JS renderer, camera, 
 * post-processing stack and the single requestAnimationFrame loop that drives
 * the entire visualisation.  It receives an ordered list of "layers" that
 * implement the small BaseLayer interface (init / update / dispose).
 */
export class CoreEngine {
    /**
     * @param {HTMLCanvasElement|HTMLElement} container – a canvas element or a div in which a new canvas will be injected.
     * @param {import('./BaseLayer.js').default[]} layers – transformer layers (or embedding / output layers) to render.
     * @param {object} [opts]
     * @param {number} [opts.speed] – global speed multiplier. 1 = realtime.
     */
    constructor(container, layers, opts = {}) {
        this._container = container;
        this._layers = layers;
        this._speed  = typeof opts.speed === 'number' ? opts.speed : 1.0;
        this._devMode = !!opts.devMode;
        this._cameraDebugEnabled = !!opts.cameraDebug;
        this._cameraDebugGroup = null;
        this._cameraDebugLine = null;
        this._cameraDebugCameraMarker = null;
        this._cameraDebugTargetMarker = null;
        this._cameraDebugAxes = null;
        this._cameraDebugCenter = new THREE.Vector3();
        this._cameraDebugForward = new THREE.Vector3();
        this._cameraDebugRight = new THREE.Vector3();
        this._cameraDebugUp = new THREE.Vector3();
        this._cameraDebugDir = new THREE.Vector3();
        this._cameraDebugRadius = 260;
        this._cameraDebugMinRadius = 140;
        this._cameraDebugMaxRadius = 700;
        // Enable/disable expensive post-processing effects (e.g. bloom).
        // Bloom is disabled by default to reduce initial load; set opts.enableBloom=true to re-enable.
        this._enableBloom = typeof opts.enableBloom === 'boolean' ? opts.enableBloom : false;

        // ────────────────────────────────────────────────────────────────────
        // Scene / camera / renderer
        // ────────────────────────────────────────────────────────────────────
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000000);

        this._cameraFarMargin = typeof opts.cameraFarMargin === 'number' ? opts.cameraFarMargin : 0;
        this._cameraBaseMaxDistance = (typeof opts.cameraMaxDistance === 'number' && opts.cameraMaxDistance > 0)
            ? opts.cameraMaxDistance
            : null;
        this._cameraMaxDistance = this._cameraBaseMaxDistance;
        this._mobileZoomOutMultiplier = (typeof opts.mobileZoomOutMultiplier === 'number' && opts.mobileZoomOutMultiplier > 1)
            ? opts.mobileZoomOutMultiplier
            : 1.0;
        this._desktopZoomOutMultiplier = (typeof opts.desktopZoomOutMultiplier === 'number' && opts.desktopZoomOutMultiplier > 1)
            ? opts.desktopZoomOutMultiplier
            : 1.45;
        this._desktopZoomOutMinWidth = (typeof opts.desktopZoomOutMinWidth === 'number' && opts.desktopZoomOutMinWidth > 0)
            ? opts.desktopZoomOutMinWidth
            : 1280;
        this._desktopZoomOutMinHeight = (typeof opts.desktopZoomOutMinHeight === 'number' && opts.desktopZoomOutMinHeight > 0)
            ? opts.desktopZoomOutMinHeight
            : 760;
        this._zoomOutSupersampleEnabled = opts.zoomOutSupersample !== false;
        this._zoomOutSupersampleMaxMultiplier = (typeof opts.zoomOutSupersampleMaxMultiplier === 'number'
            && Number.isFinite(opts.zoomOutSupersampleMaxMultiplier)
            && opts.zoomOutSupersampleMaxMultiplier >= 1)
            ? opts.zoomOutSupersampleMaxMultiplier
            : ZOOM_OUT_SUPERSAMPLE_MAX_MULTIPLIER;
        this._zoomOutSupersampleMaxDpr = (typeof opts.zoomOutSupersampleMaxDpr === 'number'
            && Number.isFinite(opts.zoomOutSupersampleMaxDpr)
            && opts.zoomOutSupersampleMaxDpr > 0)
            ? opts.zoomOutSupersampleMaxDpr
            : ZOOM_OUT_SUPERSAMPLE_MAX_DPR;
        this._zoomOutSupersampleDebounceMs = (typeof opts.zoomOutSupersampleDebounceMs === 'number'
            && Number.isFinite(opts.zoomOutSupersampleDebounceMs)
            && opts.zoomOutSupersampleDebounceMs >= 0)
            ? opts.zoomOutSupersampleDebounceMs
            : ZOOM_OUT_SUPERSAMPLE_DEBOUNCE_MS;
        this._pixelRatioRefreshTimer = null;

        const initialViewport = this._getViewportDimensions();
        this.camera = new THREE.PerspectiveCamera(60, initialViewport.width / initialViewport.height, 5, 10000);
        this.camera.position.set(0, 150, 800);

        // ────────────────────────────────────────────────────────────────────
        // Apply optional camera overrides BEFORE creating OrbitControls so the
        // controls inherit the correct initial target & position.
        // -------------------------------------------------------------------
        let customTarget = null;
        if (opts.cameraPosition instanceof THREE.Vector3) {
            this.camera.position.copy(opts.cameraPosition);
        } else if (opts.cameraPosition && typeof opts.cameraPosition === 'object' &&
                   'x' in opts.cameraPosition && 'y' in opts.cameraPosition && 'z' in opts.cameraPosition) {
            this.camera.position.set(opts.cameraPosition.x, opts.cameraPosition.y, opts.cameraPosition.z);
        }

        if (opts.cameraTarget instanceof THREE.Vector3) {
            customTarget = opts.cameraTarget.clone();
        } else if (opts.cameraTarget && typeof opts.cameraTarget === 'object' &&
                   'x' in opts.cameraTarget && 'y' in opts.cameraTarget && 'z' in opts.cameraTarget) {
            customTarget = new THREE.Vector3(opts.cameraTarget.x, opts.cameraTarget.y, opts.cameraTarget.z);
        }

        // Allow callers to explicitly override the far clipping plane, or
        // auto-expand it when a custom camera position is provided.
        if (typeof opts.cameraFar === 'number' && opts.cameraFar > this.camera.near) {
            this.camera.far = opts.cameraFar;
            this.camera.updateProjectionMatrix();
        } else if (opts.cameraPosition) {
            const distFromOrigin = this.camera.position.length();
            const margin = Math.max(0, this._cameraFarMargin);
            const suggestedFar = distFromOrigin * 2.5 + margin; // generous margin + scene allowance
            if (suggestedFar > this.camera.far) {
                this.camera.far = suggestedFar;
                this.camera.updateProjectionMatrix();
            }
        }

        const antialiasEnabled = QUALITY_PRESET === 'high';
        if (container instanceof HTMLCanvasElement) {
            this.renderer = new THREE.WebGLRenderer({ canvas: container, antialias: antialiasEnabled, logarithmicDepthBuffer: true, powerPreference: 'high-performance' });
        } else {
            this.renderer = new THREE.WebGLRenderer({ antialias: antialiasEnabled, logarithmicDepthBuffer: true, powerPreference: 'high-performance' });
            container.appendChild(this.renderer.domElement);
        }
        this._lastDevicePixelRatio = (typeof window !== 'undefined' && typeof window.devicePixelRatio === 'number' && window.devicePixelRatio > 0)
            ? window.devicePixelRatio
            : 1;
        this._appliedRenderPixelRatio = null;
        this._updateRendererPixelRatio({
            force: true,
            viewportWidth: initialViewport.width,
            viewportHeight: initialViewport.height
        });
        this.renderer.setSize(initialViewport.width, initialViewport.height);
        // Avoid long-press text selection/context menus hijacking touch controls.
        this.renderer.domElement.style.touchAction = 'none';
        this.renderer.domElement.style.userSelect = 'none';
        this.renderer.domElement.style.webkitUserSelect = 'none';
        this.renderer.domElement.style.webkitTouchCallout = 'none';
        this.renderer.domElement.style.msUserSelect = 'none';

        // Cache canvas bounds so pointer events can reuse them without forcing
        // a layout read on every move event. Updated via the resize listener.
        this._canvasRect = this.renderer.domElement.getBoundingClientRect();
        this._logTrailDebugMetrics('init', initialViewport.width, initialViewport.height);

        // ────────────────────────────────────────────────────────────────────
        // Raycasting setup for hover labels
        // ────────────────────────────────────────────────────────────────────
        this._raycaster = new THREE.Raycaster();
        this._pointer   = new THREE.Vector2();
        this._raycastingEnabled = true; // can be toggled via public API
        this._raycastRoots = [];
        this._raycastRaf = null;
        this._lastPointerX = null;
        this._lastPointerY = null;
        this._raycastSelectionHandler = null;
        this._raycastHoverHandler = null;
        this._hoveringClickable = false;
        this._hoverLabelsEnabled = true;
        this._hoverLabelsSuppressed = false;
        this._clickTapData = null;
        this._clickTapMoveThresholdSq = 36; // ~6px movement allowance

        // Hover label DOM element (similar styling to status overlay)
        this._hoverLabelDiv = document.createElement('div');
        Object.assign(this._hoverLabelDiv.style, {
            position: 'fixed',
            top: '0px',
            left: '0px',
            padding: '6px 10px',
            fontFamily: 'monospace',
            fontSize: '14px',
            color: '#fff',
            background: 'rgba(20,20,20,0.35)',
            backdropFilter: 'blur(6px)',
            WebkitBackdropFilter: 'blur(6px)',
            borderRadius: '8px',
            pointerEvents: 'none',
            zIndex: 6,
            whiteSpace: 'pre',
            display: 'none'
        });
        document.body.appendChild(this._hoverLabelDiv);

        // Track primary touch interactions so quick taps can trigger raycasts
        this._touchTapData = null;
        this._touchTapMoveThresholdSq = 64; // ~8px movement allowance
        this._touchActiveCount = 0;
        this._capturedPointerIds = new Set();

        // Bind pointer move handler and add listener
        this._onPointerMove = this._onPointerMove.bind(this);
        this.renderer.domElement.addEventListener('pointermove', this._onPointerMove);

        // Track pointer down / cancel to detect taps on touchscreens
        this._onPointerDown = this._onPointerDown.bind(this);
        this.renderer.domElement.addEventListener('pointerdown', this._onPointerDown);

        this._onPointerCancel = this._onPointerCancel.bind(this);
        this.renderer.domElement.addEventListener('pointercancel', this._onPointerCancel);

        // Track touch counts to keep OrbitControls pointers in sync on mobile.
        this._onTouchStateChange = this._onTouchStateChange.bind(this);
        this.renderer.domElement.addEventListener('touchstart', this._onTouchStateChange, { passive: true });
        this.renderer.domElement.addEventListener('touchmove', this._onTouchStateChange, { passive: true });
        this.renderer.domElement.addEventListener('touchend', this._onTouchStateChange, { passive: true });
        this.renderer.domElement.addEventListener('touchcancel', this._onTouchStateChange, { passive: true });

        // Bind pointer up handler for touch devices so taps trigger labels
        this._onPointerUp = this._onPointerUp.bind(this);
        this.renderer.domElement.addEventListener('pointerup', this._onPointerUp);
        // Prevent long-press selection and context menus on touch devices.
        this._onContextMenu = this._onContextMenu.bind(this);
        this.renderer.domElement.addEventListener('contextmenu', this._onContextMenu);
        this._onSelectStart = this._onSelectStart.bind(this);
        this.renderer.domElement.addEventListener('selectstart', this._onSelectStart);

        // ────────────────────────────────────────────────────────────────────
        // Post-processing (Bloom for emissive flashes)
        // ────────────────────────────────────────────────────────────────────
        // Post-processing is optional. Only create the composer when bloom is enabled.
        this.composer = null;
        if (this._enableBloom && QUALITY_PRESET === 'high') {
            this.composer = new EffectComposer(this.renderer);
            this.composer.addPass(new RenderPass(this.scene, this.camera));
            const bloomPass = new UnrealBloomPass(
                new THREE.Vector2(window.innerWidth, window.innerHeight),
                1.0,   // strength
                0.4,   // radius
                0.85   // threshold
            );
            this.composer.addPass(bloomPass);
        }
        this._updateRendererPixelRatio({ force: true });

        // ────────────────────────────────────────────────────────────────────
        // Controls & basic lighting
        // ────────────────────────────────────────────────────────────────────
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.08;
        this.controls.minPolarAngle = 0.2;
        this.controls.maxPolarAngle = Math.PI - 0.2;
        this.camera.up.set(0, 1, 0);
        this._keyState = new Set();
        this._keyboardPanSpeed = 420;
        this._keyboardRotateSpeed = 1.1;
        this._keyboardZoomSpeed = 0.6;
        this._keyboardZoomVector = new THREE.Vector3();
        this._keyboardCodes = new Set([
            'ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown',
            'KeyW', 'KeyA', 'KeyS', 'KeyD',
            'Equal', 'Minus', 'NumpadAdd', 'NumpadSubtract',
            'PageUp', 'PageDown'
        ]);
        this._keyboardActive = false;
        // Flag to detect when the user is actively orbiting/panning the scene
        this._isUserNavigating = false;
        // OrbitControls dispatches "start"/"end" events when interaction begins/ends
        this.controls.addEventListener('start', () => { this._isUserNavigating = true; });
        this.controls.addEventListener('end',   () => { this._isUserNavigating = false; });
        this._updateCameraFarFromControls = this._updateCameraFarFromControls.bind(this);
        this._onControlsChangePixelRatio = this._onControlsChangePixelRatio.bind(this);
        this.controls.addEventListener('change', this._updateCameraFarFromControls);
        this.controls.addEventListener('change', this._onControlsChangePixelRatio);
        if (customTarget) {
            this.controls.target.copy(customTarget);
        } else {
            this.controls.target.set(0, 66, 0);
        }
        this._applyCameraZoomLimit();
        this._updateCameraFarFromControls();

        this.scene.add(new THREE.AmbientLight(0xffffff, 0.7));
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.9);
        dirLight.position.set(25, 40, 40);
        this.scene.add(dirLight);

        if (this._cameraDebugEnabled) {
            this._initCameraDebugHelpers();
            this._updateCameraDebug();
        }

        // ────────────────────────────────────────────────────────────────────
        // Initialise layers
        // ────────────────────────────────────────────────────────────────────
        this._layers.forEach(layer => {
            layer.init(this.scene);
            if (layer instanceof Gpt2Layer) {
                const raycastRoot = layer?.raycastRoot || layer?.root;
                if (raycastRoot) {
                    this.registerRaycastRoot(raycastRoot);
                }
            }
        });

        // ────────────────────────────────────────────────────────────────────
        // Event listeners
        // ────────────────────────────────────────────────────────────────────
        window.addEventListener('resize', this._onResize);
        this._visualViewport = window.visualViewport || null;
        if (this._visualViewport) {
            this._visualViewport.addEventListener('resize', this._onResize);
            this._visualViewport.addEventListener('scroll', this._onResize);
        }
        document.addEventListener('visibilitychange', this._onVisibility);
        window.addEventListener('blur', this._onWindowBlur);
        window.addEventListener('focus', this._onWindowFocus);
        window.addEventListener('pagehide', this._onWindowBlur);
        window.addEventListener('pageshow', this._onWindowFocus);
        window.addEventListener('keydown', this._onKeyDown);
        window.addEventListener('keyup', this._onKeyUp);

        // Internal state
        this._clock  = new THREE.Clock();
        this._paused = false;
        this._pauseReasons = new Set();
        this._minFrameIntervalMs = 1000 / 60; // Cap render/update loop to 60 FPS
        this._maxUpdateDeltaSec = 0.1; // Clamp large rAF gaps (tab/window switch) to keep animations stable.
        this._lastFrameTime = null;
        this._now = (typeof performance !== 'undefined' && typeof performance.now === 'function')
            ? performance.now.bind(performance)
            : Date.now;
        const initialTweenNow = (typeof TWEEN !== 'undefined' && typeof TWEEN.now === 'function')
            ? TWEEN.now()
            : this._now();
        this._tweenTimelineMs = Number.isFinite(initialTweenNow) ? initialTweenNow : 0;
        this._tweenStartRestore = null;
        if (typeof TWEEN !== 'undefined' && TWEEN?.Tween?.prototype?.start) {
            const originalStart = TWEEN.Tween.prototype.start;
            const engine = this;
            this._tweenStartRestore = originalStart;
            TWEEN.Tween.prototype.start = function patchedStart(time) {
                const hasExplicitTime = typeof time === 'number' && Number.isFinite(time);
                const effectiveTime = hasExplicitTime ? time : engine._tweenTimelineMs;
                return originalStart.call(this, effectiveTime);
            };
        }

        // ────────────────────────────────────────────────────────────────────
        // Performance stats overlay (FPS, MS, MB) – injected if Stats.js is
        // available on the global scope.  This keeps the CoreEngine agnostic
        // of the hosting environment while providing useful runtime metrics
        // during interactive demos.
        // -------------------------------------------------------------------
        this._stats = null;
        if (typeof Stats !== 'undefined') {
            this._stats = new Stats();
            this._stats.showPanel(0); // 0 = FPS, 1 = ms/frame, 2 = MB
            this._stats.dom.style.position = 'fixed';
            this._stats.dom.style.left = '0px';
            this._stats.dom.style.top = 'auto';
            this._stats.dom.style.bottom = '0px';
            this._stats.dom.style.pointerEvents = 'none';
            this._stats.dom.style.display = this._devMode ? 'block' : 'none';
            document.body.appendChild(this._stats.dom);
        }

        // Kick off RAF loop
        requestAnimationFrame(this._animate);
    }

    // ────────────────────────────────────────────────────────────────────────
    // Public API
    // ────────────────────────────────────────────────────────────────────────
    setSpeed(multiplier) {
        this._speed = multiplier;
    }

    /** Enable or disable hover raycasting and label updates at runtime. */
    setRaycastingEnabled(enabled) {
        this._raycastingEnabled = !!enabled;
        if (!this._raycastingEnabled && this._hoverLabelDiv) {
            this._hoverLabelDiv.style.display = 'none';
        }
        if (!this._raycastingEnabled) {
            this._setCanvasCursor(false);
            if (this._raycastHoverHandler) this._raycastHoverHandler(null);
        }
    }

    /**
     * Temporarily suppress hover labels without disabling selection raycasts.
     * Useful when the pointer is over UI overlays.
     */
    setHoverLabelsSuppressed(suppressed) {
        const next = !!suppressed;
        if (this._hoverLabelsSuppressed === next) return;
        this._hoverLabelsSuppressed = next;
        if (next) {
            if (this._hoverLabelDiv) this._hoverLabelDiv.style.display = 'none';
            this._setCanvasCursor(false);
            if (this._raycastHoverHandler) this._raycastHoverHandler(null);
        }
    }

    /**
     * Clear any active pointer/tap tracking and reset OrbitControls state.
     * Useful when the user shifts focus to UI overlays.
     */
    resetInteractionState() {
        this._touchTapData = null;
        this._clickTapData = null;
        this._resetControlsState();
    }

    setDevMode(enabled) {
        this._devMode = !!enabled;
        if (this._stats && this._stats.dom) {
            this._stats.dom.style.display = this._devMode ? 'block' : 'none';
        }
    }

    setCameraDebugEnabled(enabled) {
        const nextValue = !!enabled;
        this._cameraDebugEnabled = nextValue;
        if (nextValue) {
            if (!this._cameraDebugGroup) {
                this._initCameraDebugHelpers();
            }
            if (this._cameraDebugGroup) {
                this._cameraDebugGroup.visible = true;
                this._updateCameraDebug();
            }
        } else if (this._cameraDebugGroup) {
            this._cameraDebugGroup.visible = false;
        }
    }

    isCameraDebugEnabled() {
        return !!this._cameraDebugEnabled;
    }

    /** Register a handler for click/tap selection via raycasting. */
    setRaycastSelectionHandler(handler) {
        this._raycastSelectionHandler = (typeof handler === 'function') ? handler : null;
    }

    /** Register a handler for hover updates via raycasting. */
    setRaycastHoverHandler(handler) {
        this._raycastHoverHandler = (typeof handler === 'function') ? handler : null;
        if (!this._raycastHoverHandler && this._hoverLabelDiv) {
            this._hoverLabelDiv.style.display = 'none';
        }
    }

    /**
     * Register an Object3D as a raycast root so hover labels include it.
     * @param {THREE.Object3D} root
     */
    registerRaycastRoot(root) {
        if (!root || typeof root !== 'object' || !root.isObject3D) return;
        if (!this._raycastRoots.includes(root)) {
            this._raycastRoots.push(root);
        }
    }

    /**
     * Remove an Object3D from the raycast root list.
     * @param {THREE.Object3D} root
     */
    unregisterRaycastRoot(root) {
        const idx = this._raycastRoots.indexOf(root);
        if (idx !== -1) {
            this._raycastRoots.splice(idx, 1);
        }
    }

    // Backwards-compatible alias used by existing call sites.
    removeRaycastRoot(root) {
        this.unregisterRaycastRoot(root);
    }

    /** Return current raycasting state. */
    isRaycastingEnabled() {
        return !!this._raycastingEnabled;
    }

    /** Determine whether the user is actively manipulating the orbit controls. */
    isUserNavigating() {
        return !!this._isUserNavigating;
    }

    /** Inform the engine that the camera or controls were changed programmatically. */
    notifyCameraUpdated() {
        if (!this.controls) return;
        this._updateCameraFarFromControls();
        this._updateRendererPixelRatio();
    }

    pause(reason = 'generic') {
        this._pauseReasons.add(reason);
        if (!this._paused) {
            this._paused = true;
            this._clock.stop();
        }
    }

    resume(reason = 'generic') {
        this._pauseReasons.delete(reason);
        if (this._pauseReasons.size === 0 && this._paused) {
            this._paused = false;
            this._clock.start();
        }
    }

    dispose() {
        this._cancelPendingPixelRatioRefresh();
        window.removeEventListener('resize', this._onResize);
        if (this._visualViewport) {
            this._visualViewport.removeEventListener('resize', this._onResize);
            this._visualViewport.removeEventListener('scroll', this._onResize);
            this._visualViewport = null;
        }
        document.removeEventListener('visibilitychange', this._onVisibility);
        window.removeEventListener('blur', this._onWindowBlur);
        window.removeEventListener('focus', this._onWindowFocus);
        window.removeEventListener('pagehide', this._onWindowBlur);
        window.removeEventListener('pageshow', this._onWindowFocus);
        window.removeEventListener('keydown', this._onKeyDown);
        window.removeEventListener('keyup', this._onKeyUp);
        if (this.controls) {
            this.controls.removeEventListener('change', this._updateCameraFarFromControls);
            this.controls.removeEventListener('change', this._onControlsChangePixelRatio);
            this.controls.dispose();
        }
        this._layers.forEach(l => l.dispose());
        this._raycastRoots.length = 0;

        this.scene.traverse(obj => {
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) {
                if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
                else obj.material.dispose();
            }
        });

        if (this.composer && Array.isArray(this.composer.passes)) {
            this.composer.passes.forEach(p => p.dispose && p.dispose());
        }
        this.renderer.dispose();
        this.renderer.domElement.removeEventListener('pointermove', this._onPointerMove);
        this.renderer.domElement.removeEventListener('pointerdown', this._onPointerDown);
        this.renderer.domElement.removeEventListener('pointercancel', this._onPointerCancel);
        this.renderer.domElement.removeEventListener('pointerup', this._onPointerUp);
        this.renderer.domElement.removeEventListener('touchstart', this._onTouchStateChange);
        this.renderer.domElement.removeEventListener('touchmove', this._onTouchStateChange);
        this.renderer.domElement.removeEventListener('touchend', this._onTouchStateChange);
        this.renderer.domElement.removeEventListener('touchcancel', this._onTouchStateChange);
        this.renderer.domElement.removeEventListener('contextmenu', this._onContextMenu);
        this.renderer.domElement.removeEventListener('selectstart', this._onSelectStart);
        if (this._hoverLabelDiv && this._hoverLabelDiv.parentElement) {
            this._hoverLabelDiv.parentElement.removeChild(this._hoverLabelDiv);
        }
        if (this._tweenStartRestore && typeof TWEEN !== 'undefined' && TWEEN?.Tween?.prototype) {
            try {
                TWEEN.Tween.prototype.start = this._tweenStartRestore;
            } catch (_) { /* restore best-effort */ }
            this._tweenStartRestore = null;
        }
        if (this._raycastRaf !== null && typeof cancelAnimationFrame === 'function') {
            cancelAnimationFrame(this._raycastRaf);
            this._raycastRaf = null;
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // Private helpers
    // ────────────────────────────────────────────────────────────────────────
    _getViewportDimensions = () => {
        const fallbackWidth = (typeof window !== 'undefined' && Number.isFinite(window.innerWidth))
            ? window.innerWidth
            : 1;
        const fallbackHeight = (typeof window !== 'undefined' && Number.isFinite(window.innerHeight))
            ? window.innerHeight
            : 1;

        const visualViewport = (typeof window !== 'undefined') ? window.visualViewport : null;
        if (!visualViewport) {
            return {
                width: Math.max(1, Math.round(fallbackWidth)),
                height: Math.max(1, Math.round(fallbackHeight))
            };
        }

        const vvWidth = Number.isFinite(visualViewport.width) ? visualViewport.width : fallbackWidth;
        const vvHeight = Number.isFinite(visualViewport.height) ? visualViewport.height : fallbackHeight;

        return {
            width: Math.max(1, Math.round(vvWidth)),
            height: Math.max(1, Math.round(vvHeight))
        };
    };

    _onResize = () => {
        const { width, height } = this._getViewportDimensions();
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this._updateRendererPixelRatio({ force: true, viewportWidth: width, viewportHeight: height });
        this.renderer.setSize(width, height);
        if (this.composer) this.composer.setSize(width, height);
        refreshTrailDisplayScales(this.scene);
        this._logTrailDebugMetrics('resize', width, height);
        this._canvasRect = this.renderer.domElement.getBoundingClientRect();
        this._applyCameraZoomLimit();
        this._updateCameraFarFromControls();
    };

    _logTrailDebugMetrics = (reason, viewportWidth = null, viewportHeight = null) => {
        if (typeof window === 'undefined' || window.__TRAIL_DEBUG !== true) return;
        const renderer = this.renderer;
        const canvas = renderer?.domElement || null;
        const screen = window.screen || null;
        const visualViewport = window.visualViewport || null;
        const gl = (renderer && typeof renderer.getContext === 'function') ? renderer.getContext() : null;
        const dpr = (typeof window.devicePixelRatio === 'number' && window.devicePixelRatio > 0)
            ? window.devicePixelRatio
            : 1;
        const renderRatio = (renderer && typeof renderer.getPixelRatio === 'function')
            ? renderer.getPixelRatio()
            : null;
        const width = Number.isFinite(viewportWidth) ? viewportWidth : (Number.isFinite(window.innerWidth) ? window.innerWidth : null);
        const height = Number.isFinite(viewportHeight) ? viewportHeight : (Number.isFinite(window.innerHeight) ? window.innerHeight : null);
        let glAliasedLineWidthRange = null;
        let glSamples = null;
        let glSampleBuffers = null;
        let glAntialias = null;
        let glVersion = null;
        let glShadingLanguageVersion = null;
        if (gl) {
            try {
                const range = gl.getParameter(gl.ALIASED_LINE_WIDTH_RANGE);
                if (range && typeof range.length === 'number') {
                    glAliasedLineWidthRange = [Number(range[0]), Number(range[1])];
                }
                glSamples = Number(gl.getParameter(gl.SAMPLES));
                glSampleBuffers = Number(gl.getParameter(gl.SAMPLE_BUFFERS));
                const attrs = (typeof gl.getContextAttributes === 'function') ? gl.getContextAttributes() : null;
                glAntialias = attrs ? !!attrs.antialias : null;
                glVersion = String(gl.getParameter(gl.VERSION) || '');
                glShadingLanguageVersion = String(gl.getParameter(gl.SHADING_LANGUAGE_VERSION) || '');
            } catch (_) {
                // Best-effort debug logging only.
            }
        }
        let trailCount = 0;
        let visibleTrailCount = 0;
        if (this.scene && typeof this.scene.traverse === 'function') {
            this.scene.traverse((obj) => {
                if (!obj || !obj.userData) return;
                const isTrail = !!(obj.userData.isTrail || obj.userData.trailMerged || obj.userData.trailBatch);
                if (!isTrail) return;
                trailCount += 1;
                if (obj.visible !== false) {
                    visibleTrailCount += 1;
                }
            });
        }
        console.info('[trail-debug]', {
            reason,
            dpr,
            rendererPixelRatio: renderRatio,
            resolvedPixelRatio: resolveRenderPixelRatio({ viewportWidth: width, viewportHeight: height }),
            viewport: {
                width,
                height,
                innerWidth: Number.isFinite(window.innerWidth) ? window.innerWidth : null,
                innerHeight: Number.isFinite(window.innerHeight) ? window.innerHeight : null,
                outerWidth: Number.isFinite(window.outerWidth) ? window.outerWidth : null,
                outerHeight: Number.isFinite(window.outerHeight) ? window.outerHeight : null,
                visualViewport: visualViewport ? {
                    width: Number.isFinite(visualViewport.width) ? visualViewport.width : null,
                    height: Number.isFinite(visualViewport.height) ? visualViewport.height : null,
                    scale: Number.isFinite(visualViewport.scale) ? visualViewport.scale : null,
                    offsetLeft: Number.isFinite(visualViewport.offsetLeft) ? visualViewport.offsetLeft : null,
                    offsetTop: Number.isFinite(visualViewport.offsetTop) ? visualViewport.offsetTop : null
                } : null
            },
            display: {
                screenWidth: screen ? screen.width : null,
                screenHeight: screen ? screen.height : null,
                availWidth: screen ? screen.availWidth : null,
                availHeight: screen ? screen.availHeight : null,
                colorDepth: screen ? screen.colorDepth : null,
                pixelDepth: screen ? screen.pixelDepth : null
            },
            canvas: {
                clientWidth: canvas?.clientWidth ?? null,
                clientHeight: canvas?.clientHeight ?? null,
                width: canvas?.width ?? null,
                height: canvas?.height ?? null
            },
            webgl: {
                aliasedLineWidthRange: glAliasedLineWidthRange,
                antialias: glAntialias,
                samples: Number.isFinite(glSamples) ? glSamples : null,
                sampleBuffers: Number.isFinite(glSampleBuffers) ? glSampleBuffers : null,
                version: glVersion,
                shadingLanguageVersion: glShadingLanguageVersion,
                precision: renderer?.capabilities?.precision ?? null,
                maxSamples: Number.isFinite(renderer?.capabilities?.maxSamples) ? renderer.capabilities.maxSamples : null,
                isWebGL2: !!renderer?.capabilities?.isWebGL2
            },
            trailStyle: {
                baseLineWidth: TRAIL_LINE_WIDTH,
                scaledLineWidth: scaleLineWidthForDisplay(TRAIL_LINE_WIDTH),
                lineWidthScaleFactor: scaleLineWidthForDisplay(TRAIL_LINE_WIDTH) / Math.max(0.0001, TRAIL_LINE_WIDTH),
                baseOpacity: TRAIL_OPACITY,
                scaledOpacity: scaleOpacityForDisplay(TRAIL_OPACITY),
                opacityScaleFactor: scaleOpacityForDisplay(TRAIL_OPACITY) / Math.max(0.0001, TRAIL_OPACITY)
            },
            trailObjects: {
                total: trailCount,
                visible: visibleTrailCount
            }
        });
    };

    _updateRendererPixelRatio = ({ force = false, viewportWidth = null, viewportHeight = null } = {}) => {
        if (!this.renderer) return;
        let nextRatio = resolveRenderPixelRatio({
            viewportWidth,
            viewportHeight
        });
        nextRatio = this._applyZoomOutSupersample(nextRatio);
        if (!force && Number.isFinite(this._appliedRenderPixelRatio)
            && Math.abs(this._appliedRenderPixelRatio - nextRatio) < 0.001) {
            return;
        }
        this._appliedRenderPixelRatio = nextRatio;
        this.renderer.setPixelRatio(nextRatio);
        if (this.composer && typeof this.composer.setPixelRatio === 'function') {
            this.composer.setPixelRatio(nextRatio);
        }
        // Keep LineMaterial-based trails in sync with the active render size/DPR.
        refreshTrailDisplayScales(this.scene);
    };

    _onControlsChangePixelRatio() {
        this._schedulePixelRatioRefresh();
    }

    _schedulePixelRatioRefresh() {
        const debounceMs = this._zoomOutSupersampleEnabled
            ? Math.max(0, Math.round(this._zoomOutSupersampleDebounceMs))
            : 0;
        if (debounceMs <= 0 || typeof setTimeout !== 'function') {
            this._updateRendererPixelRatio();
            return;
        }
        this._cancelPendingPixelRatioRefresh();
        this._pixelRatioRefreshTimer = setTimeout(() => {
            this._pixelRatioRefreshTimer = null;
            this._updateRendererPixelRatio();
        }, debounceMs);
    }

    _cancelPendingPixelRatioRefresh() {
        if (this._pixelRatioRefreshTimer === null) return;
        if (typeof clearTimeout === 'function') {
            clearTimeout(this._pixelRatioRefreshTimer);
        }
        this._pixelRatioRefreshTimer = null;
    }

    _applyZoomOutSupersample(baseRatio) {
        if (!this._zoomOutSupersampleEnabled) return baseRatio;
        if (!(Number.isFinite(baseRatio) && baseRatio > 0)) return baseRatio;
        if (!this.camera || !this.controls || !this.controls.target) return baseRatio;
        if (this._isTouchPrimaryDevice()) return baseRatio;

        const maxDistance = (typeof this.controls.maxDistance === 'number' && Number.isFinite(this.controls.maxDistance) && this.controls.maxDistance > 0)
            ? this.controls.maxDistance
            : null;
        if (!maxDistance) return baseRatio;

        const distance = this.camera.position.distanceTo(this.controls.target);
        if (!(Number.isFinite(distance) && distance > 0)) return baseRatio;

        const distanceRatio = distance / maxDistance;
        const denom = Math.max(0.0001, ZOOM_OUT_SUPERSAMPLE_MAX_DISTANCE_RATIO - ZOOM_OUT_SUPERSAMPLE_MIN_DISTANCE_RATIO);
        const t = THREE.MathUtils.clamp((distanceRatio - ZOOM_OUT_SUPERSAMPLE_MIN_DISTANCE_RATIO) / denom, 0, 1);
        if (t <= 0) return baseRatio;

        const maxMultiplier = Math.max(1, this._zoomOutSupersampleMaxMultiplier);
        const multiplier = 1 + (maxMultiplier - 1) * t;
        const maxDpr = Math.max(baseRatio, this._zoomOutSupersampleMaxDpr);
        const boosted = Math.min(maxDpr, baseRatio * multiplier);
        const quantized = Math.round(boosted / ZOOM_OUT_SUPERSAMPLE_RATIO_STEP) * ZOOM_OUT_SUPERSAMPLE_RATIO_STEP;
        return Math.max(baseRatio, quantized);
    }

    _onVisibility = () => {
        if (document.hidden) {
            this.pause('visibility');
            this._resetControlsState();
        } else {
            this.resume('visibility');
        }
    };

    _onWindowBlur = () => {
        this._resetControlsState();
    };

    _onWindowFocus = () => {
        // Some environments occasionally miss visibilitychange on return.
        // Explicitly clear visibility pause when focus is back.
        if (typeof document !== 'undefined' && document.hidden) return;
        this.resume('visibility');
    };

    _releasePointerCapture = (pointerId = null) => {
        const domElement = this.renderer?.domElement;
        if (!domElement || typeof domElement.releasePointerCapture !== 'function') return;
        if (Number.isFinite(pointerId)) {
            try {
                domElement.releasePointerCapture(pointerId);
            } catch (_) { /* best-effort */ }
            if (this._capturedPointerIds) {
                this._capturedPointerIds.delete(pointerId);
            }
            return;
        }
        if (!this._capturedPointerIds || this._capturedPointerIds.size === 0) return;
        for (const id of this._capturedPointerIds) {
            try {
                domElement.releasePointerCapture(id);
            } catch (_) { /* best-effort */ }
        }
        this._capturedPointerIds.clear();
    };

    _resetControlsState = () => {
        const controls = this.controls;
        if (!controls) return;
        const domElement = controls.domElement;
        const pointers = Array.isArray(controls._pointers) ? controls._pointers : null;
        if (domElement && typeof domElement.releasePointerCapture === 'function' && pointers) {
            for (const pointerId of pointers) {
                try {
                    domElement.releasePointerCapture(pointerId);
                } catch (_) { /* best-effort */ }
            }
        }
        if (pointers) pointers.length = 0;
        if (controls._pointerPositions && typeof controls._pointerPositions === 'object') {
            controls._pointerPositions = {};
        }
        if (typeof controls._controlActive === 'boolean') {
            controls._controlActive = false;
        }
        if (typeof controls.state === 'number') {
            controls.state = -1;
        }
        this._isUserNavigating = false;
        if (this._hoverLabelDiv) {
            this._hoverLabelDiv.style.display = 'none';
        }
        if (typeof controls.dispatchEvent === 'function') {
            controls.dispatchEvent({ type: 'end' });
        }
        this._releasePointerCapture();
        this._touchTapData = null;
        this._clickTapData = null;
        if (this._keyState) {
            this._keyState.clear();
        }
        this._keyboardActive = false;
    };

    _onKeyDown = (event) => {
        if (!this._shouldHandleKeyboardEvent(event)) return;
        const code = event.code;
        if (!this._keyboardCodes.has(code)) return;
        this._keyState.add(code);
        event.preventDefault();
    };

    _onKeyUp = (event) => {
        const code = event.code;
        if (this._keyState && this._keyState.has(code)) {
            this._keyState.delete(code);
            event.preventDefault();
        }
    };

    _shouldHandleKeyboardEvent = (event) => {
        if (!event || event.defaultPrevented) return false;
        if (event.metaKey || event.ctrlKey) return false;
        const target = event.target;
        if (!target) return true;
        if (target.isContentEditable) return false;
        const tagName = target.tagName;
        return !(tagName === 'INPUT' || tagName === 'TEXTAREA' || tagName === 'SELECT');
    };

    _applyKeyboardNavigation = (deltaSeconds) => {
        const controls = this.controls;
        if (!controls || !this._keyState || this._keyState.size === 0) {
            if (this._keyboardActive) {
                this._keyboardActive = false;
                controls?.dispatchEvent?.({ type: 'end' });
            }
            return;
        }

        let didMove = false;
        const rotateSpeed = this._keyboardRotateSpeed * deltaSeconds;
        const panSpeed = this._keyboardPanSpeed * deltaSeconds;

        if (controls.enableRotate) {
            if (this._keyState.has('ArrowLeft')) {
                controls._rotateLeft(rotateSpeed);
                didMove = true;
            }
            if (this._keyState.has('ArrowRight')) {
                controls._rotateLeft(-rotateSpeed);
                didMove = true;
            }
            if (this._keyState.has('ArrowUp')) {
                controls._rotateUp(rotateSpeed);
                didMove = true;
            }
            if (this._keyState.has('ArrowDown')) {
                controls._rotateUp(-rotateSpeed);
                didMove = true;
            }
        }

        if (controls.enablePan) {
            if (this._keyState.has('KeyW')) {
                controls._pan(0, panSpeed);
                didMove = true;
            }
            if (this._keyState.has('KeyS')) {
                controls._pan(0, -panSpeed);
                didMove = true;
            }
            if (this._keyState.has('KeyA')) {
                controls._pan(panSpeed, 0);
                didMove = true;
            }
            if (this._keyState.has('KeyD')) {
                controls._pan(-panSpeed, 0);
                didMove = true;
            }
        }

        if (controls.enableZoom) {
            let zoomDirection = 0;
            if (this._keyState.has('Equal') || this._keyState.has('NumpadAdd') || this._keyState.has('PageUp')) {
                zoomDirection += 1;
            }
            if (this._keyState.has('Minus') || this._keyState.has('NumpadSubtract') || this._keyState.has('PageDown')) {
                zoomDirection -= 1;
            }
            if (zoomDirection !== 0) {
                this._applyKeyboardZoom(zoomDirection, deltaSeconds);
                didMove = true;
            }
        }

        if (didMove && !this._keyboardActive) {
            this._keyboardActive = true;
            controls.dispatchEvent?.({ type: 'start' });
        } else if (!didMove && this._keyboardActive) {
            this._keyboardActive = false;
            controls.dispatchEvent?.({ type: 'end' });
        }
    };

    _applyKeyboardZoom = (direction, deltaSeconds) => {
        const controls = this.controls;
        if (!controls || !this.camera || !controls.target) return;
        const offset = this._keyboardZoomVector;
        offset.subVectors(this.camera.position, controls.target);
        const currentDistance = offset.length();
        if (!currentDistance || !Number.isFinite(currentDistance)) return;

        const zoomScale = Math.exp(-direction * this._keyboardZoomSpeed * deltaSeconds);
        let desiredDistance = currentDistance * zoomScale;
        const minDistance = (typeof controls.minDistance === 'number') ? controls.minDistance : 0;
        const maxDistance = (typeof controls.maxDistance === 'number') ? controls.maxDistance : Infinity;
        desiredDistance = Math.max(minDistance, Math.min(maxDistance, desiredDistance));
        if (!Number.isFinite(desiredDistance) || desiredDistance === currentDistance) return;

        offset.setLength(desiredDistance);
        this.camera.position.copy(controls.target).add(offset);
    };

    _onPointerDown = (event) => {
        this._updateHoverLabelMode(event.pointerType);
        if (event.pointerType === 'touch') {
            const domElement = this.renderer?.domElement;
            if (domElement && typeof domElement.setPointerCapture === 'function') {
                try {
                    domElement.setPointerCapture(event.pointerId);
                    if (this._capturedPointerIds) {
                        this._capturedPointerIds.add(event.pointerId);
                    }
                } catch (_) { /* best-effort */ }
            }
            // Only track the primary tap candidate to avoid conflicts with multi-touch gestures
            if (this._touchTapData && this._touchTapData.id !== event.pointerId) return;
            const { clientX, clientY } = event;
            if (typeof clientX !== 'number' || typeof clientY !== 'number') return;
            this._touchTapData = {
                id: event.pointerId,
                startX: clientX,
                startY: clientY,
                lastX: clientX,
                lastY: clientY,
                moved: false
            };
            return;
        }

        if (event.button !== 0) return;
        if (this._clickTapData && this._clickTapData.id !== event.pointerId) return;
        const { clientX, clientY } = event;
        if (typeof clientX !== 'number' || typeof clientY !== 'number') return;
        this._clickTapData = {
            id: event.pointerId,
            startX: clientX,
            startY: clientY,
            lastX: clientX,
            lastY: clientY,
            moved: false
        };
    };

    _onPointerCancel = (event) => {
        if (this._touchTapData && this._touchTapData.id === event.pointerId) {
            this._touchTapData = null;
        }
        if (this._clickTapData && this._clickTapData.id === event.pointerId) {
            this._clickTapData = null;
        }
        this._releasePointerCapture(event.pointerId);
    };

    _onTouchStateChange = (event) => {
        if (!event || !event.touches) return;
        this._touchActiveCount = event.touches.length;
        const controls = this.controls;
        if (!controls || !Array.isArray(controls._pointers)) return;
        if (controls._pointers.length > this._touchActiveCount) {
            this._resetControlsState();
        }
    };

    _onPointerUp = (event) => {
        this._releasePointerCapture(event.pointerId);
        if (event.pointerType === 'touch') {
            const tapData = this._touchTapData;
            const isSamePointer = tapData && tapData.id === event.pointerId;
            const clientX = typeof event.clientX === 'number' ? event.clientX : (isSamePointer ? tapData.lastX : null);
            const clientY = typeof event.clientY === 'number' ? event.clientY : (isSamePointer ? tapData.lastY : null);
            const shouldTriggerTap = Boolean(isSamePointer && !tapData.moved && clientX !== null && clientY !== null);
            this._touchTapData = null;

            if (!shouldTriggerTap) return;
            const x = clientX;
            const y = clientY;
            // Defer to the next frame to give OrbitControls a moment to emit its "end" event
            requestAnimationFrame(() => {
                this._performRaycastAt(x, y, { force: true });
                this._performSelectionAt(x, y, { force: true });
            });
            return;
        }

        const clickData = this._clickTapData;
        const isSamePointer = clickData && clickData.id === event.pointerId;
        const clientX = typeof event.clientX === 'number' ? event.clientX : (isSamePointer ? clickData.lastX : null);
        const clientY = typeof event.clientY === 'number' ? event.clientY : (isSamePointer ? clickData.lastY : null);
        const shouldTriggerClick = Boolean(isSamePointer && !clickData.moved && clientX !== null && clientY !== null);
        this._clickTapData = null;
        if (!shouldTriggerClick) return;
        const x = clientX;
        const y = clientY;
        requestAnimationFrame(() => {
            this._performSelectionAt(x, y, { force: true });
        });
    };

    _onContextMenu = (event) => {
        event.preventDefault();
    };

    _onSelectStart = (event) => {
        event.preventDefault();
    };

    _onPointerMove = (event) => {
        this._updateHoverLabelMode(event.pointerType);
        if (event.pointerType === 'touch' && this._touchTapData && this._touchTapData.id === event.pointerId) {
            if (typeof event.clientX === 'number' && typeof event.clientY === 'number') {
                this._touchTapData.lastX = event.clientX;
                this._touchTapData.lastY = event.clientY;
                if (!this._touchTapData.moved) {
                    const dx = event.clientX - this._touchTapData.startX;
                    const dy = event.clientY - this._touchTapData.startY;
                    if (dx * dx + dy * dy > this._touchTapMoveThresholdSq) {
                        this._touchTapData.moved = true;
                    }
                }
            }
        }
        if (event.pointerType !== 'touch' && this._clickTapData && this._clickTapData.id === event.pointerId) {
            if (typeof event.clientX === 'number' && typeof event.clientY === 'number') {
                this._clickTapData.lastX = event.clientX;
                this._clickTapData.lastY = event.clientY;
                if (!this._clickTapData.moved) {
                    const dx = event.clientX - this._clickTapData.startX;
                    const dy = event.clientY - this._clickTapData.startY;
                    if (dx * dx + dy * dy > this._clickTapMoveThresholdSq) {
                        this._clickTapData.moved = true;
                    }
                }
            }
        }
        if (this._hoverLabelsSuppressed) {
            if (this._hoverLabelDiv) this._hoverLabelDiv.style.display = 'none';
            this._setCanvasCursor(false);
            if (this._raycastHoverHandler) this._raycastHoverHandler(null);
            return;
        }
        if (!this._hoverLabelsEnabled) {
            if (this._hoverLabelDiv) this._hoverLabelDiv.style.display = 'none';
            this._setCanvasCursor(false);
            return;
        }
        this._lastPointerX = event.clientX;
        this._lastPointerY = event.clientY;
        this._scheduleRaycast();
    };

    _isRaycastObjectVisible(object) {
        let current = object;
        while (current) {
            if (current.visible === false) return false;
            current = current.parent;
        }
        return true;
    }

    _isRaycastObjectInteractable(object) {
        let current = object;
        while (current) {
            const data = current.userData;
            if (data) {
                if (data.raycastDisabled === true) return false;
                const gate = data.raycastEnabled;
                if (typeof gate === 'boolean' && !gate) return false;
                if (typeof gate === 'function') {
                    let allowed = false;
                    try {
                        allowed = !!gate();
                    } catch (_) {
                        allowed = false;
                    }
                    if (!allowed) return false;
                }
            }
            current = current.parent;
        }
        return true;
    }

    _resolveRaycastLabel(intersects) {
        return resolveRaycastLabelFromIntersections(intersects, {
            isObjectVisible: (obj) => this._isRaycastObjectVisible(obj),
            isObjectInteractable: (obj) => this._isRaycastObjectInteractable(obj),
            layers: this._layers,
            normalizeLabel: normalizeRaycastLabel,
            isCachedKvSelection
        });
    }

    _updateHoverLabelMode(pointerType) {
        const enabled = pointerType !== 'touch';
        if (this._hoverLabelsEnabled === enabled) return;
        this._hoverLabelsEnabled = enabled;
        if (!enabled) {
            if (this._hoverLabelDiv) this._hoverLabelDiv.style.display = 'none';
            this._setCanvasCursor(false);
            if (this._raycastHoverHandler) this._raycastHoverHandler(null);
        }
    }

    _setCanvasCursor(isPointer) {
        if (!this.renderer?.domElement) return;
        const next = isPointer ? 'pointer' : '';
        if (this._hoveringClickable === isPointer) return;
        this._hoveringClickable = isPointer;
        this.renderer.domElement.style.cursor = next;
    }

    _performRaycastAt(clientX, clientY, { force = false } = {}) {
        if (!this._raycastingEnabled) return;
        if (this._hoverLabelsSuppressed) {
            if (this._hoverLabelDiv) this._hoverLabelDiv.style.display = 'none';
            this._setCanvasCursor(false);
            return;
        }
        if (!this._hoverLabelsEnabled) return;
        if (!force && this._isUserNavigating) return;
        if (!Number.isFinite(clientX) || !Number.isFinite(clientY)) return;

        const rect = this._canvasRect;
        if (!rect || !rect.width || !rect.height) return;

        this._pointer.x = ((clientX - rect.left) / rect.width) * 2 - 1;
        this._pointer.y = -((clientY - rect.top) / rect.height) * 2 + 1;

        this._raycaster.setFromCamera(this._pointer, this.camera);
        if (!this._raycastRoots.length) {
            this._hoverLabelDiv.style.display = 'none';
            this._setCanvasCursor(false);
            return;
        }

        const intersects = this._raycaster.intersectObjects(this._raycastRoots, true);
        if (perfStats.enabled) {
            perfStats.inc('raycasts');
            perfStats.inc('raycastIntersects', intersects.length);
        }
        const resolved = this._resolveRaycastLabel(intersects);
        if (resolved && resolved.label) {
            const hoverLabel = simplifyLayerNormParamHoverLabel(
                resolved.label,
                resolved.info,
                resolved.object || resolved.hit?.object || null
            );
            this._hoverLabelDiv.textContent = hoverLabel;
            this._hoverLabelDiv.style.left = `${clientX + 12}px`;
            this._hoverLabelDiv.style.top  = `${clientY + 12}px`;
            this._hoverLabelDiv.style.display = 'block';
            this._setCanvasCursor(true);
            return;
        }
        // No intersection with a labelled object – hide overlay.
        this._hoverLabelDiv.style.display = 'none';
        this._setCanvasCursor(false);
    }

    _performSelectionAt(clientX, clientY, { force = false } = {}) {
        if (!this._raycastingEnabled) {
            if (this._raycastSelectionHandler) this._raycastSelectionHandler(null);
            return;
        }
        if (!force && this._isUserNavigating) return;
        if (!Number.isFinite(clientX) || !Number.isFinite(clientY)) return;

        const rect = this._canvasRect;
        if (!rect || !rect.width || !rect.height) return;

        this._pointer.x = ((clientX - rect.left) / rect.width) * 2 - 1;
        this._pointer.y = -((clientY - rect.top) / rect.height) * 2 + 1;
        this._raycaster.setFromCamera(this._pointer, this.camera);
        if (!this._raycastRoots.length) {
            if (this._raycastSelectionHandler) this._raycastSelectionHandler(null);
            return;
        }

        const intersects = this._raycaster.intersectObjects(this._raycastRoots, true);
        if (perfStats.enabled) {
            perfStats.inc('selectRaycasts');
            perfStats.inc('raycastIntersects', intersects.length);
        }
        const resolved = this._resolveRaycastLabel(intersects);
        if (!resolved || !resolved.label) {
            if (this._raycastSelectionHandler) this._raycastSelectionHandler(null);
            return;
        }
        if (this._raycastSelectionHandler) {
            this._raycastSelectionHandler({
                label: resolved.label,
                kind: resolved.kind || null,
                info: resolved.info || null,
                object: resolved.object || resolved.hit?.object || null,
                hit: resolved.hit || null
            });
        }
    }

    _scheduleRaycast() {
        if (this._raycastRaf !== null) return;
        if (typeof requestAnimationFrame !== 'function') return;
        this._raycastRaf = requestAnimationFrame(() => {
            this._raycastRaf = null;
            this._performRaycastAt(this._lastPointerX, this._lastPointerY);
        });
    }

    _updateCameraFarFromControls() {
        if (!this.camera || !this.controls) return;
        const margin = Math.max(0, this._cameraFarMargin || 0);
        let desiredFar = this.camera.far;
        try {
            const target = this.controls.target || new THREE.Vector3();
            const distToTarget = this.camera.position.distanceTo(target);
            desiredFar = Math.max(desiredFar, distToTarget + margin);
        } catch (_) { /* non-fatal */ }

        try {
            const distFromOrigin = this.camera.position.length();
            desiredFar = Math.max(desiredFar, distFromOrigin + margin);
        } catch (_) { /* non-fatal */ }

        if (desiredFar > this.camera.far) {
            this.camera.far = desiredFar;
            this.camera.updateProjectionMatrix();
        }
    }

    _applyCameraZoomLimit() {
        if (!this.camera || !this.controls) return;
        const margin = Math.max(0, this._cameraFarMargin || 0);
        let baseMaxDistance = this._cameraBaseMaxDistance;

        const targetClone = this.controls.target ? this.controls.target.clone() : null;
        let distToTarget = 0;
        try {
            if (targetClone) {
                distToTarget = this.camera.position.distanceTo(targetClone);
            }
        } catch (_) { /* non-fatal */ }

        if (!distToTarget) {
            try {
                distToTarget = this.camera.position.length();
            } catch (_) {
                distToTarget = 0;
            }
        }

        if (!(typeof baseMaxDistance === 'number' && baseMaxDistance > 0)) {
            const baseDistance = distToTarget > 0 ? distToTarget : 1000;
            if (margin > 0) {
                baseMaxDistance = baseDistance + margin * 0.85;
            } else {
                baseMaxDistance = Math.max(baseDistance * 4, baseDistance + 2000);
            }
            baseMaxDistance = Math.max(baseMaxDistance, baseDistance + 1);
            this._cameraBaseMaxDistance = baseMaxDistance;
        }

        let maxDistance = baseMaxDistance;
        if (this._isTouchPrimaryDevice()) {
            maxDistance = baseMaxDistance * this._mobileZoomOutMultiplier;
        } else if (this._isLargeDesktopViewport()) {
            maxDistance = baseMaxDistance * this._desktopZoomOutMultiplier;
        }

        if (typeof maxDistance === 'number' && Number.isFinite(maxDistance) && maxDistance > 0) {
            this._cameraMaxDistance = maxDistance;
            this.controls.maxDistance = maxDistance;
            this._clampCameraDistanceToLimit(maxDistance);
        }
    }

    _isTouchPrimaryDevice() {
        if (typeof window === 'undefined') return false;
        if (typeof window.matchMedia === 'function') {
            const coarse = window.matchMedia('(pointer: coarse)').matches
                || window.matchMedia('(hover: none) and (pointer: coarse)').matches;
            if (coarse) return true;
        }
        const touchPoints = Number.isFinite(window?.navigator?.maxTouchPoints)
            ? window.navigator.maxTouchPoints
            : 0;
        return touchPoints > 0;
    }

    _isLargeDesktopViewport() {
        if (typeof window === 'undefined') return false;
        if (this._isTouchPrimaryDevice()) return false;
        const width = window.innerWidth || 0;
        const height = window.innerHeight || 0;
        if (!Number.isFinite(width) || !Number.isFinite(height)) return false;
        return width >= this._desktopZoomOutMinWidth && height >= this._desktopZoomOutMinHeight;
    }

    _clampCameraDistanceToLimit(limit) {
        if (!this.camera || !this.controls || !this.controls.target) return;
        if (!(typeof limit === 'number' && Number.isFinite(limit) && limit > 0)) return;

        const offset = new THREE.Vector3().copy(this.camera.position).sub(this.controls.target);
        const distance = offset.length();
        if (!(distance > limit) || distance <= 0) return;

        offset.multiplyScalar(limit / distance);
        this.camera.position.copy(this.controls.target).add(offset);
    }

    _initCameraDebugHelpers() {
        if (this._cameraDebugGroup) return;
        const group = new THREE.Group();
        group.name = 'CameraOrbitGizmo';
        group.visible = this._cameraDebugEnabled;

        const targetMat = new THREE.MeshBasicMaterial({ color: 0x9ad6ff, transparent: true, opacity: 0.9 });
        const targetMesh = new THREE.Mesh(new THREE.SphereGeometry(1, 12, 12), targetMat);
        targetMesh.name = 'CameraOrbitTarget';
        targetMesh.renderOrder = 50;
        targetMesh.material.depthTest = false;
        targetMesh.castShadow = false;
        targetMesh.receiveShadow = false;
        group.add(targetMesh);

        const camMat = new THREE.MeshBasicMaterial({ color: 0xffc857 });
        const camMesh = new THREE.Mesh(new THREE.ConeGeometry(0.6, 1.2, 12), camMat);
        camMesh.name = 'CameraOrbitMarker';
        camMesh.renderOrder = 51;
        camMesh.material.depthTest = false;
        camMesh.castShadow = false;
        camMesh.receiveShadow = false;
        camMesh.rotation.x = -Math.PI / 2;

        const axes = new THREE.AxesHelper(1);
        axes.name = 'CameraOrbitAxes';
        axes.renderOrder = 52;
        axes.material.depthTest = false;
        camMesh.add(axes);

        group.add(camMesh);

        const lineGeom = new THREE.BufferGeometry();
        const linePositions = new Float32Array(6);
        lineGeom.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
        const lineMat = new THREE.LineBasicMaterial({ color: 0x77d1ff, transparent: true, opacity: 0.7 });
        const line = new THREE.Line(lineGeom, lineMat);
        line.name = 'CameraOrbitLine';
        line.renderOrder = 49;
        line.material.depthTest = false;
        group.add(line);

        this._cameraDebugGroup = group;
        this._cameraDebugTargetMarker = targetMesh;
        this._cameraDebugCameraMarker = camMesh;
        this._cameraDebugAxes = axes;
        this._cameraDebugLine = line;

        this.scene.add(group);
    }

    _updateCameraDebug() {
        if (!this._cameraDebugEnabled || !this._cameraDebugGroup) return;
        if (!this.camera || !this.controls || !this.controls.target) return;

        const camera = this.camera;
        const target = this.controls.target;
        const distance = camera.position.distanceTo(target);
        const radius = THREE.MathUtils.clamp(distance * 0.05, this._cameraDebugMinRadius, this._cameraDebugMaxRadius);
        this._cameraDebugRadius = radius;

        const forward = this._cameraDebugForward;
        const right = this._cameraDebugRight;
        const up = this._cameraDebugUp;
        camera.getWorldDirection(forward);
        right.crossVectors(forward, camera.up).normalize();
        up.copy(camera.up).normalize();

        const center = this._cameraDebugCenter;
        center.copy(target);
        center.addScaledVector(right, -radius * 2.6);
        center.addScaledVector(up, -radius * 1.6);
        center.addScaledVector(forward, -radius * 0.8);
        this._cameraDebugGroup.position.copy(center);

        const dir = this._cameraDebugDir;
        dir.copy(camera.position).sub(target);
        if (dir.lengthSq() < 0.0001) {
            dir.set(0, 0, 1);
        }
        dir.normalize();

        const camMarker = this._cameraDebugCameraMarker;
        camMarker.position.copy(dir).multiplyScalar(radius);
        camMarker.quaternion.copy(camera.quaternion);
        camMarker.scale.setScalar(radius * 0.35);

        if (this._cameraDebugAxes) {
            this._cameraDebugAxes.scale.setScalar(radius * 0.65);
        }
        if (this._cameraDebugTargetMarker) {
            this._cameraDebugTargetMarker.scale.setScalar(radius * 0.12);
        }
        if (this._cameraDebugLine) {
            const attr = this._cameraDebugLine.geometry.attributes.position;
            attr.array[0] = 0;
            attr.array[1] = 0;
            attr.array[2] = 0;
            attr.array[3] = camMarker.position.x;
            attr.array[4] = camMarker.position.y;
            attr.array[5] = camMarker.position.z;
            attr.needsUpdate = true;
            this._cameraDebugLine.geometry.computeBoundingSphere();
        }
    }

    _animate = () => {
        requestAnimationFrame(this._animate);
        const now = this._now();

        const liveDpr = (typeof window !== 'undefined' && typeof window.devicePixelRatio === 'number' && window.devicePixelRatio > 0)
            ? window.devicePixelRatio
            : 1;
        if (!Number.isFinite(this._lastDevicePixelRatio) || Math.abs(liveDpr - this._lastDevicePixelRatio) > 0.001) {
            this._lastDevicePixelRatio = liveDpr;
            this._onResize();
        }

        const lastFrameTime = this._lastFrameTime;
        if (lastFrameTime !== null && (now - lastFrameTime) < this._minFrameIntervalMs) {
            return;
        }
        this._lastFrameTime = now;
        const frameDelta = lastFrameTime !== null
            ? Math.min((now - lastFrameTime) / 1000, 0.1)
            : (1 / 60);

        const visibilityPauseOnly = this._paused && this._pauseReasons.size === 1 && this._pauseReasons.has('visibility');
        if (visibilityPauseOnly) return;

        const perfEnabled = perfStats.enabled;
        if (perfEnabled) {
            perfStats.beginFrame(now);
        }

        if (this._devMode && this._stats) this._stats.begin();
        const layerErrorLogThrottleMs = 1000;

        if (!this._paused) {
            const layers = this._layers;
            const updateStart = perfEnabled ? this._now() : 0;
            const rawDt = this._clock.getDelta() * this._speed;
            const dt = Math.min(rawDt, this._maxUpdateDeltaSec);
            for (let i = 0; i < layers.length; i++) {
                const layer = layers[i];
                if (!layer) continue;
                if (layer.isActive || layer._transitionPhase === 'positioning') {
                    try {
                        layer.update(dt);
                    } catch (err) {
                        const logNow = this._now();
                        const lastLogAt = Number.isFinite(layer.__lastUpdateErrorLogAt)
                            ? layer.__lastUpdateErrorLogAt
                            : -Infinity;
                        if ((logNow - lastLogAt) >= layerErrorLogThrottleMs) {
                            console.error(`CoreEngine: layer ${i} update() failed`, err);
                            layer.__lastUpdateErrorLogAt = logNow;
                        }
                    }
                }
            }
            if (perfEnabled) {
                perfStats.addTime('update', this._now() - updateStart);
            }

            if (typeof TWEEN !== 'undefined' && typeof TWEEN.update === 'function') {
                const tweenStart = perfEnabled ? this._now() : 0;
                this._tweenTimelineMs += dt * 1000;
                TWEEN.update(this._tweenTimelineMs);
                if (perfEnabled) {
                    perfStats.addTime('tween', this._now() - tweenStart);
                    if (typeof TWEEN.getAll === 'function') {
                        perfStats.setGauge('tweens', TWEEN.getAll().length);
                    }
                }
            }

            for (let i = 0; i < layers.length; i++) {
                const layer = layers[i];
                if (layer && typeof layer.postUpdate === 'function') {
                    try {
                        layer.postUpdate(dt);
                    } catch (err) {
                        const logNow = this._now();
                        const lastLogAt = Number.isFinite(layer.__lastPostUpdateErrorLogAt)
                            ? layer.__lastPostUpdateErrorLogAt
                            : -Infinity;
                        if ((logNow - lastLogAt) >= layerErrorLogThrottleMs) {
                            console.error(`CoreEngine: layer ${i} postUpdate() failed`, err);
                            layer.__lastPostUpdateErrorLogAt = logNow;
                        }
                    }
                }
            }
        } else {
            // Keep opt-in systems responsive while the main scene is paused
            // (for example: auto-camera transitions after enabling follow mode).
            const layers = this._layers;
            for (let i = 0; i < layers.length; i++) {
                const layer = layers[i];
                if (!layer || layer.updateWhenPaused !== true || typeof layer.update !== 'function') {
                    continue;
                }
                try {
                    layer.update(0);
                } catch (err) {
                    const logNow = this._now();
                    const lastLogAt = Number.isFinite(layer.__lastUpdateErrorLogAt)
                        ? layer.__lastUpdateErrorLogAt
                        : -Infinity;
                    if ((logNow - lastLogAt) >= layerErrorLogThrottleMs) {
                        console.error(`CoreEngine: layer ${i} paused update() failed`, err);
                        layer.__lastUpdateErrorLogAt = logNow;
                    }
                }
            }
        }

        this._applyKeyboardNavigation(frameDelta);
        this.controls.update();
        this._updateCameraDebug();
        const renderStart = perfEnabled ? this._now() : 0;
        if (this.composer) {
            this.composer.render();
        } else {
            this.renderer.render(this.scene, this.camera);
        }
        if (perfEnabled) {
            perfStats.addTime('render', this._now() - renderStart);
        }

        if (this._devMode && this._stats) this._stats.end();
        if (perfEnabled) {
            perfStats.endFrame(this._now());
        }
    };
}

/**
 * Convenience helper: start the engine and return a dispose function.
 * @param {HTMLCanvasElement|HTMLElement} container
 * @param {import('./BaseLayer.js').default[]} layers
 */
export function startEngine(container, layers, opts) {
    const engine = new CoreEngine(container, layers, opts);
    return () => engine.dispose();
} 
