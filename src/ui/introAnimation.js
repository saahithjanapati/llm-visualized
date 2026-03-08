import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { FontLoader } from 'three/examples/jsm/loaders/FontLoader.js';
import { TextGeometry } from 'three/examples/jsm/geometries/TextGeometry.js';
import { appState } from '../state/appState.js';
import { resolveRenderPixelRatio } from '../utils/constants.js';

// Sets up the intro typing animation and HDRI transition.
// Returns a cleanup function for the intro resources.
export function initIntroAnimation(pipeline, gptCanvas) {
    const introCanvas = document.getElementById('introCanvas');
    const renderer = new THREE.WebGLRenderer({ canvas: introCanvas, antialias: true });
    renderer.shadowMap.enabled = false;
    renderer.setPixelRatio(resolveRenderPixelRatio({
        viewportWidth: window.innerWidth,
        viewportHeight: window.innerHeight
    }));
    renderer.setSize(window.innerWidth, window.innerHeight);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);
    const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 1, 20000);
    camera.position.set(0, 0, 1500);

    appState.introSceneRef = scene;
    appState.applyEnvironmentBackground(pipeline, scene);

    const hideLoadingOverlay = () => {
        const overlay = document.getElementById('loadingOverlay');
        if (!overlay) return;
        overlay.classList.add('is-hidden');
    };

    const waitForMainScenePresentation = async () => {
        const engine = pipeline?.engine || null;
        if (engine && typeof engine.whenFirstFramePresented === 'function') {
            await engine.whenFirstFramePresented();
        }
        if (typeof window !== 'undefined' && typeof window.requestAnimationFrame === 'function') {
            await new Promise((resolve) => {
                window.requestAnimationFrame(() => {
                    window.requestAnimationFrame(resolve);
                });
            });
        }
    };

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // Gate and lifecycle for intro RAF/rendering so we can stop it after hand-off
    appState.introActive = true;
    appState.introRaf = 0;
    appState.introCleaned = false;
    let transitionPromise = null;

    function cleanupIntro() {
        if (appState.introCleaned) return;
        appState.introCleaned = true;
        appState.introActive = false;
        if (appState.introRaf) cancelAnimationFrame(appState.introRaf);
        controls?.dispose?.();
        renderer?.dispose?.();
        scene.traverse((child) => {
            if (child.geometry) child.geometry.dispose();
            if (child.material) {
                if (Array.isArray(child.material)) child.material.forEach((m) => m && m.dispose && m.dispose());
                else if (child.material.dispose) child.material.dispose();
            }
        });
    }

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(5, 10, 7.5);
    dirLight.castShadow = false;
    scene.add(dirLight);

    const TEXT = 'Can machines think?';
    const TYPE_DELAY = 120;
    let cursorMesh;
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
                setTimeout(transitionToGPT, 1500);
                return;
            }
            const mesh = charMeshes[index];
            const width = charWidths[index];
            const gap = (index === charMeshes.length - 1) ? extraCursorGap : charSpacing;
            new TWEEN.Tween(cursorMesh.position).to({ x: cursorMesh.position.x + width + gap }, TYPE_DELAY * 0.9).start();
            if (mesh) {
                new TWEEN.Tween(mesh.material).to({ opacity: 1 }, TYPE_DELAY).start();
            }
            setTimeout(() => revealChar(index + 1), TYPE_DELAY);
        };
        const blinkCursor = () => setInterval(() => { cursorMesh.visible = !cursorMesh.visible; }, 500);
        setTimeout(() => revealChar(0), 500);
    });

    const finalizeMainSceneTransition = async () => {
        if (transitionPromise) return transitionPromise;
        transitionPromise = (async () => {
            try {
                await waitForMainScenePresentation();
            } catch (_) {
                // Best-effort only; if the engine fails to signal, still reveal.
            }
            introCanvas.style.display = 'none';
            gptCanvas.style.display = 'block';
            cleanupIntro();
            hideLoadingOverlay();
        })();
        return transitionPromise;
    };

    appState.setEnvironmentKey(appState.selectedEnvironmentKey, pipeline, scene, { persist: false }).then(() => {
        scene.traverse((obj) => { if (obj.isAmbientLight) scene.remove(obj); });
        if (pipeline?.engine?.scene && typeof pipeline.engine.scene.traverse === 'function') {
            pipeline.engine.scene.traverse((obj) => { if (obj.isAmbientLight) pipeline.engine.scene.remove(obj); });
        }
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.0;
        if (pipeline?.engine?.renderer) {
            pipeline.engine.renderer.toneMapping = THREE.ACESFilmicToneMapping;
            pipeline.engine.renderer.toneMappingExposure = 1.0;
        }
        void finalizeMainSceneTransition();
    }).catch((err) => {
        console.warn('HDRI failed to load:', err);
        void finalizeMainSceneTransition();
    });

    window.addEventListener('resize', () => {
        const { innerWidth, innerHeight } = window;
        if (appState.introActive) {
            camera.aspect = innerWidth / innerHeight;
            camera.updateProjectionMatrix();
            renderer.setPixelRatio(resolveRenderPixelRatio({
                viewportWidth: innerWidth,
                viewportHeight: innerHeight
            }));
            renderer.setSize(innerWidth, innerHeight);
        }
    });

    const loopIntro = () => {
        if (!appState.introActive) return;
        appState.introRaf = requestAnimationFrame(loopIntro);
        if (appState.modalPaused || appState.userPaused) return;
        controls.update();
        if (typeof TWEEN !== 'undefined') TWEEN.update();
        renderer.render(scene, camera);
    };

    if (!appState.skipIntro) {
        loopIntro();
    } else {
        finalizeMainSceneTransition().catch((err) => {
            console.error('Failed to transition to GPT:', err);
            const ic = document.getElementById('introCanvas');
            if (ic) ic.style.display = 'none';
            hideLoadingOverlay();
        });
    }

    function transitionToGPT() {
        void finalizeMainSceneTransition();
    }

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

    return cleanupIntro;
}
