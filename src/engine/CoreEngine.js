import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { QUALITY_PRESET, resolveRenderDprCap } from '../utils/constants.js';
import Gpt2Layer from './layers/Gpt2Layer.js';

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
        // Enable/disable expensive post-processing effects (e.g. bloom).
        // Bloom is disabled by default to reduce initial load; set opts.enableBloom=true to re-enable.
        this._enableBloom = typeof opts.enableBloom === 'boolean' ? opts.enableBloom : false;

        // ────────────────────────────────────────────────────────────────────
        // Scene / camera / renderer
        // ────────────────────────────────────────────────────────────────────
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000000);

        this._cameraFarMargin = typeof opts.cameraFarMargin === 'number' ? opts.cameraFarMargin : 0;
        this._cameraMaxDistance = (typeof opts.cameraMaxDistance === 'number' && opts.cameraMaxDistance > 0)
            ? opts.cameraMaxDistance
            : null;

        this.camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 5, 10000);
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
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        // Improve crispness on HiDPI displays with a higher cap; can be tuned via constants.
        const dprCap = resolveRenderDprCap();
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, dprCap));

        // Cache canvas bounds so pointer events can reuse them without forcing
        // a layout read on every move event. Updated via the resize listener.
        this._canvasRect = this.renderer.domElement.getBoundingClientRect();

        // ────────────────────────────────────────────────────────────────────
        // Raycasting setup for hover labels
        // ────────────────────────────────────────────────────────────────────
        this._raycaster = new THREE.Raycaster();
        this._pointer   = new THREE.Vector2();
        this._raycastingEnabled = true; // can be toggled via public API
        this._raycastRoots = [];

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

        // Bind pointer move handler and add listener
        this._onPointerMove = this._onPointerMove.bind(this);
        this.renderer.domElement.addEventListener('pointermove', this._onPointerMove);

        // Track pointer down / cancel to detect taps on touchscreens
        this._onPointerDown = this._onPointerDown.bind(this);
        this.renderer.domElement.addEventListener('pointerdown', this._onPointerDown);

        this._onPointerCancel = this._onPointerCancel.bind(this);
        this.renderer.domElement.addEventListener('pointercancel', this._onPointerCancel);

        // Bind pointer up handler for touch devices so taps trigger labels
        this._onPointerUp = this._onPointerUp.bind(this);
        this.renderer.domElement.addEventListener('pointerup', this._onPointerUp);

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

        // ────────────────────────────────────────────────────────────────────
        // Controls & basic lighting
        // ────────────────────────────────────────────────────────────────────
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        // Flag to detect when the user is actively orbiting/panning the scene
        this._isUserNavigating = false;
        // OrbitControls dispatches "start"/"end" events when interaction begins/ends
        this.controls.addEventListener('start', () => { this._isUserNavigating = true; });
        this.controls.addEventListener('end',   () => { this._isUserNavigating = false; });
        this._updateCameraFarFromControls = this._updateCameraFarFromControls.bind(this);
        this.controls.addEventListener('change', this._updateCameraFarFromControls);
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

        // ────────────────────────────────────────────────────────────────────
        // Initialise layers
        // ────────────────────────────────────────────────────────────────────
        this._layers.forEach(layer => {
            layer.init(this.scene);
            if (layer instanceof Gpt2Layer && layer?.root) {
                this.registerRaycastRoot(layer.root);
            }
        });

        // ────────────────────────────────────────────────────────────────────
        // Event listeners
        // ────────────────────────────────────────────────────────────────────
        window.addEventListener('resize', this._onResize);
        document.addEventListener('visibilitychange', this._onVisibility);

        // Internal state
        this._clock  = new THREE.Clock();
        this._paused = false;
        this._pauseReasons = new Set();
        this._updatables = new Set();
        const initialTweenNow = (typeof TWEEN !== 'undefined' && typeof TWEEN.now === 'function')
            ? TWEEN.now()
            : (typeof performance !== 'undefined' ? performance.now() : Date.now());
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
            this._stats.dom.style.top  = '0px';
            document.body.appendChild(this._stats.dom);
        }

        // Bind methods so we can add/remove listeners cleanly.
        this._animate   = this._animate.bind(this);
        this._onResize  = this._onResize.bind(this);
        this._onVisibility = this._onVisibility.bind(this);

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

    /** Register an updatable helper that runs each frame alongside layers. */
    registerUpdatable(updatable) {
        if (!updatable || typeof updatable.update !== 'function') return;
        this._updatables.add(updatable);
    }

    /** Remove a previously registered updatable helper. */
    unregisterUpdatable(updatable) {
        if (!updatable) return;
        this._updatables.delete(updatable);
    }

    /** Return current raycasting state. */
    isRaycastingEnabled() {
        return !!this._raycastingEnabled;
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
        window.removeEventListener('resize', this._onResize);
        document.removeEventListener('visibilitychange', this._onVisibility);
        if (this.controls) {
            this.controls.removeEventListener('change', this._updateCameraFarFromControls);
            this.controls.dispose();
        }
        this._layers.forEach(l => l.dispose());
        this._raycastRoots.length = 0;
        this._updatables.clear();

        this.scene.traverse(obj => {
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) {
                if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
                else obj.material.dispose();
            }
        });

        this.composer.passes.forEach(p => p.dispose && p.dispose());
        this.renderer.dispose();
        this.renderer.domElement.removeEventListener('pointermove', this._onPointerMove);
        this.renderer.domElement.removeEventListener('pointerdown', this._onPointerDown);
        this.renderer.domElement.removeEventListener('pointercancel', this._onPointerCancel);
        this.renderer.domElement.removeEventListener('pointerup', this._onPointerUp);
        if (this._hoverLabelDiv && this._hoverLabelDiv.parentElement) {
            this._hoverLabelDiv.parentElement.removeChild(this._hoverLabelDiv);
        }
        if (this._tweenStartRestore && typeof TWEEN !== 'undefined' && TWEEN?.Tween?.prototype) {
            try {
                TWEEN.Tween.prototype.start = this._tweenStartRestore;
            } catch (_) { /* restore best-effort */ }
            this._tweenStartRestore = null;
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // Private helpers
    // ────────────────────────────────────────────────────────────────────────
    _onResize = () => {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        if (this.composer) this.composer.setSize(window.innerWidth, window.innerHeight);
        this._canvasRect = this.renderer.domElement.getBoundingClientRect();
    };

    _onVisibility = () => {
        document.hidden ? this.pause('visibility') : this.resume('visibility');
    };

    _onPointerDown = (event) => {
        if (event.pointerType !== 'touch') return;
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
    };

    _onPointerCancel = (event) => {
        if (this._touchTapData && this._touchTapData.id === event.pointerId) {
            this._touchTapData = null;
        }
    };

    _onPointerUp = (event) => {
        if (event.pointerType !== 'touch') return;
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
        });
    };

    _onPointerMove = (event) => {
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
        this._performRaycastAt(event.clientX, event.clientY);
    };

    _performRaycastAt(clientX, clientY, { force = false } = {}) {
        if (!this._raycastingEnabled) return;
        if (!force && this._isUserNavigating) return;
        if (!Number.isFinite(clientX) || !Number.isFinite(clientY)) return;

        const rect = this._canvasRect;
        if (!rect || !rect.width || !rect.height) return;

        this._pointer.x = ((clientX - rect.left) / rect.width) * 2 - 1;
        this._pointer.y = -((clientY - rect.top) / rect.height) * 2 + 1;

        this._raycaster.setFromCamera(this._pointer, this.camera);
        if (!this._raycastRoots.length) {
            this._hoverLabelDiv.style.display = 'none';
            return;
        }

        const intersects = this._raycaster.intersectObjects(this._raycastRoots, true);
        // Pass 1: Prefer detailed labels from merged K/V instanced meshes anywhere in the hit list
        for (const hit of intersects) {
            try {
                const obj = hit.object;
                if (obj && obj.isInstancedMesh) {
                    // Walk up to find a MHSAAnimation context (via layer reference)
                    for (const layer of this._layers) {
                        if (!layer || !layer.mhsaAnimation) continue;
                        const mhsa = layer.mhsaAnimation;
                        if (typeof mhsa.decodeMergedKVIntersection === 'function') {
                            const info = mhsa.decodeMergedKVIntersection(hit);
                            if (info) {
                                const headText = (typeof info.headIndex === 'number' && info.headIndex >= 0) ? `Head ${info.headIndex + 1}` : 'Head ?';
                                const laneText = (typeof info.laneIndex === 'number' && info.laneIndex >= 0) ? `Lane ${info.laneIndex + 1}` : 'Lane ?';
                                const catText  = info.category === 'V' ? 'Value (Red)' : 'Key (Green)';
                                this._hoverLabelDiv.textContent = `${catText}\n${headText} • ${laneText}`;
                                this._hoverLabelDiv.style.left = `${clientX + 12}px`;
                                this._hoverLabelDiv.style.top  = `${clientY + 12}px`;
                                this._hoverLabelDiv.style.display = 'block';
                                return; // Prefer detailed merged K/V hit over any generic labels
                            }
                        }
                    }
                }
            } catch (_) { /* non-fatal */ }
        }

        // Pass 2: Fallback – show the first generic label found
        for (const hit of intersects) {
            let obj = hit.object;
            while (obj) {
                const lbl = obj.userData?.label || obj.name;
                if (lbl && lbl !== 'Weight Matrix') {
                    this._hoverLabelDiv.textContent = lbl;
                    this._hoverLabelDiv.style.left = `${clientX + 12}px`;
                    this._hoverLabelDiv.style.top  = `${clientY + 12}px`;
                    this._hoverLabelDiv.style.display = 'block';
                    return;
                }
                obj = obj.parent;
            }
        }
        // No intersection with a labelled object – hide overlay.
        this._hoverLabelDiv.style.display = 'none';
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
        let maxDistance = this._cameraMaxDistance;

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

        if (!(typeof maxDistance === 'number' && maxDistance > 0)) {
            const baseDistance = distToTarget > 0 ? distToTarget : 1000;
            if (margin > 0) {
                maxDistance = baseDistance + margin * 0.85;
            } else {
                maxDistance = Math.max(baseDistance * 4, baseDistance + 2000);
            }
            maxDistance = Math.max(maxDistance, baseDistance + 1);
        }

        if (typeof maxDistance === 'number' && Number.isFinite(maxDistance) && maxDistance > 0) {
            this._cameraMaxDistance = maxDistance;
            this.controls.maxDistance = maxDistance;
        }
    }

    _animate = () => {
        requestAnimationFrame(this._animate);

        const visibilityPauseOnly = this._paused && this._pauseReasons.size === 1 && this._pauseReasons.has('visibility');
        if (visibilityPauseOnly) return;

        if (this._stats) this._stats.begin();

        if (!this._paused) {
            const dt = this._clock.getDelta() * this._speed;
            this._layers.forEach(layer => {
                if (!layer) return;
                if (layer.isActive || layer._transitionPhase === 'positioning') {
                    layer.update(dt);
                }
            });

            if (this._updatables.size) {
                for (const helper of this._updatables) {
                    if (!helper || typeof helper.update !== 'function') continue;
                    try {
                        helper.update(dt);
                    } catch (err) {
                        console.error('Auxiliary update failed:', err);
                    }
                }
            }

            if (typeof TWEEN !== 'undefined' && typeof TWEEN.update === 'function') {
                this._tweenTimelineMs += dt * 1000;
                TWEEN.update(this._tweenTimelineMs);
            }
        }

        this.controls.update();
        if (this.composer) {
            this.composer.render();
        } else {
            this.renderer.render(this.scene, this.camera);
        }

        if (this._stats) this._stats.end();
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
