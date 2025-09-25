import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { FontLoader } from 'three/examples/jsm/loaders/FontLoader.js';
import { TextGeometry } from 'three/examples/jsm/geometries/TextGeometry.js';
import { EXRLoader } from 'three/examples/jsm/loaders/EXRLoader.js';
import { appState } from '../state/appState.js';
import { resolveRenderDprCap } from '../utils/constants.js';
import hdrBackgroundUrl from '../../rogland_clear_night_64.exr?url';

// Sets up the intro typing animation and HDRI transition.
// Returns a cleanup function for the intro resources.
export function initIntroAnimation(pipeline, gptCanvas) {
    const introCanvas = document.getElementById('introCanvas');
    const renderer = new THREE.WebGLRenderer({ canvas: introCanvas, antialias: true });
    renderer.shadowMap.enabled = false;
    const cap = resolveRenderDprCap();
    const dpr = typeof window.devicePixelRatio === 'number' ? window.devicePixelRatio : 1;
    renderer.setPixelRatio(Math.min(dpr, cap));
    renderer.setSize(window.innerWidth, window.innerHeight);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);
    const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 1, 20000);
    camera.position.set(0, 0, 1500);

    appState.introSceneRef = scene;
    appState.applyEnvironmentBackground(pipeline, scene);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // Gate and lifecycle for intro RAF/rendering so we can stop it after hand-off
    appState.introActive = true;
    appState.introRaf = 0;
    appState.introCleaned = false;

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

    const dirLight = new THREE.DirectionalLight(0x45d7ff, 0.9);
    dirLight.position.set(6, 12, 10);
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
            const mat = new THREE.MeshStandardMaterial({
                color: 0x18f6ff,
                emissive: new THREE.Color(0x0d9be0),
                emissiveIntensity: 1.4,
                metalness: 0.6,
                roughness: 0.25,
                transparent: true,
                opacity: 0
            });
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

        const cursorGeo = new THREE.BoxGeometry(18, textHeight, 20);
        const cursorMat = new THREE.MeshBasicMaterial({ color: 0x72faff });
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

    const exrLoader = new EXRLoader().setDataType(THREE.HalfFloatType);
    exrLoader.load(hdrBackgroundUrl, (texture) => {
        texture.mapping = THREE.EquirectangularReflectionMapping;
        texture.center.set(0.5, 0.5);
        texture.rotation = Math.PI;
        texture.needsUpdate = true;
        appState.environmentTexture = texture;
        scene.environment = texture;
        pipeline.engine.scene.environment = texture;
        appState.applyEnvironmentBackground(pipeline, scene);
        scene.traverse((obj) => { if (obj.isAmbientLight) scene.remove(obj); });
        pipeline.engine.scene.traverse((obj) => { if (obj.isAmbientLight) pipeline.engine.scene.remove(obj); });
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.0;
        pipeline.engine.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        pipeline.engine.renderer.toneMappingExposure = 1.0;
        introCanvas.style.display = 'none';
        cleanupIntro();
        const gif = document.getElementById('loadingGif');
        if (gif) gif.style.display = 'none';
    }, undefined, (err) => {
        console.warn('HDRI failed to load:', err);
        introCanvas.style.display = 'none';
        cleanupIntro();
        const gif = document.getElementById('loadingGif');
        if (gif) gif.style.display = 'none';
    });

    window.addEventListener('resize', () => {
        const { innerWidth, innerHeight } = window;
        if (appState.introActive) {
            camera.aspect = innerWidth / innerHeight;
            camera.updateProjectionMatrix();
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
        try {
            transitionToGPT();
        } catch (err) {
            console.error('Failed to transition to GPT:', err);
            const ic = document.getElementById('introCanvas');
            if (ic) ic.style.display = 'none';
        }
        const gif = document.getElementById('loadingGif');
        if (gif) gif.style.display = 'none';
    }

    function transitionToGPT() {
        introCanvas.style.display = 'none';
        gptCanvas.style.display = 'block';
        cleanupIntro();
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
