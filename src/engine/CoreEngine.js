import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';

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

        // ────────────────────────────────────────────────────────────────────
        // Scene / camera / renderer
        // ────────────────────────────────────────────────────────────────────
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x111111);

        this.camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 10000);
        this.camera.position.set(0, 150, 800);

        if (container instanceof HTMLCanvasElement) {
            this.renderer = new THREE.WebGLRenderer({ canvas: container, antialias: true });
        } else {
            this.renderer = new THREE.WebGLRenderer({ antialias: true });
            container.appendChild(this.renderer.domElement);
        }
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

        // ────────────────────────────────────────────────────────────────────
        // Post-processing (Bloom for emissive flashes)
        // ────────────────────────────────────────────────────────────────────
        this.composer = new EffectComposer(this.renderer);
        this.composer.addPass(new RenderPass(this.scene, this.camera));
        const bloomPass = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 1.0, 0.4, 0.85);
        this.composer.addPass(bloomPass);

        // ────────────────────────────────────────────────────────────────────
        // Controls & basic lighting
        // ────────────────────────────────────────────────────────────────────
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.target.set(0, 66, 0);

        this.scene.add(new THREE.AmbientLight(0xffffff, 0.7));
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.9);
        dirLight.position.set(25, 40, 40);
        this.scene.add(dirLight);

        // ────────────────────────────────────────────────────────────────────
        // Initialise layers
        // ────────────────────────────────────────────────────────────────────
        this._layers.forEach(layer => layer.init(this.scene));

        // ────────────────────────────────────────────────────────────────────
        // Event listeners
        // ────────────────────────────────────────────────────────────────────
        window.addEventListener('resize', this._onResize);
        document.addEventListener('visibilitychange', this._onVisibility);

        // Internal state
        this._clock  = new THREE.Clock();
        this._paused = false;

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

    pause() { this._paused = true; this._clock.stop(); }
    resume() { this._paused = false; this._clock.start(); }

    dispose() {
        window.removeEventListener('resize', this._onResize);
        document.removeEventListener('visibilitychange', this._onVisibility);
        this.controls.dispose();
        this._layers.forEach(l => l.dispose());

        this.scene.traverse(obj => {
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) {
                if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
                else obj.material.dispose();
            }
        });

        this.composer.passes.forEach(p => p.dispose && p.dispose());
        this.renderer.dispose();
    }

    // ────────────────────────────────────────────────────────────────────────
    // Private helpers
    // ────────────────────────────────────────────────────────────────────────
    _onResize = () => {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.composer.setSize(window.innerWidth, window.innerHeight);
    };

    _onVisibility = () => {
        document.hidden ? this.pause() : this.resume();
    };

    _animate = () => {
        requestAnimationFrame(this._animate);
        if (this._paused) return;

        const dt = this._clock.getDelta() * this._speed;
        this._layers.forEach(layer => layer.update(dt));

        // External libraries like @tweenjs/tween need ticking.
        if (typeof TWEEN !== 'undefined' && TWEEN.update) {
            TWEEN.update();
        }

        this.controls.update();
        this.composer.render();
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