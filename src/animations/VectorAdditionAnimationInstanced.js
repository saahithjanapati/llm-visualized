import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { VectorVisualizationInstanced } from '../components/VectorVisualizationInstanced.js';
import { VECTOR_LENGTH, SPHERE_DIAMETER } from '../utils/constants.js';
import { mapValueToColor } from '../utils/colors.js';

// Note: Assumes TWEEN is loaded globally via <script> tag

export function initVectorAdditionAnimationInstanced(containerElement) {
    // --- Basic Three.js setup ---
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);

    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 5, 15);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    containerElement.appendChild(renderer.domElement);

    // --- Post Processing Setup ---
    const composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));
    const bloomPass = new UnrealBloomPass(
        new THREE.Vector2(window.innerWidth, window.innerHeight),
        1.2, 0.5, 0.8 // strength, radius, threshold
    );
    composer.addPass(bloomPass);

    // --- Lighting ---
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(5, 10, 7);
    scene.add(directionalLight);

    // --- Controls ---
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // --- Vectors Setup ---
    const data1 = Array.from({ length: VECTOR_LENGTH }, () => Math.random() * 2 - 1);
    const data2 = Array.from({ length: VECTOR_LENGTH }, () => Math.random() * 2 - 1);
    const yOffset = 1.5;
    const vector1 = new VectorVisualizationInstanced(data1, new THREE.Vector3(0, yOffset, 0));
    const vector2 = new VectorVisualizationInstanced(data2, new THREE.Vector3(0, -yOffset, 0));
    scene.add(vector1.group);
    scene.add(vector2.group);

    // --- Core Animation Logic ---
    function startAdditionAnimation(v1, v2) {
        if (typeof TWEEN === 'undefined' || typeof TWEEN.Tween !== 'function' || typeof TWEEN.Easing === 'undefined') {
            console.error('Global TWEEN object not loaded!');
            return;
        }

        const duration = 750;
        const flashDuration = 150;
        const delayBetweenCubes = 75;

        // The instances inside vector1 are initially at local y = 0 (their group is already offset by +yOffset).
        // To reach vector2's group (offset -yOffset in world space) they must travel -2 * yOffset in *local* space.
        const localTargetDelta = -yOffset * 2; // e.g. from 0 to -3 if yOffset = 1.5

        for (let i = 0; i < VECTOR_LENGTH; i++) {
            const startYOffset = 0; // initial local y
            const targetYOffset = localTargetDelta; // final local y (negative)

            const tweenObject = { y: startYOffset };
            const moveTween = new TWEEN.Tween(tweenObject)
                .to({ y: targetYOffset }, duration)
                .easing(TWEEN.Easing.Quadratic.InOut)
                .delay(i * delayBetweenCubes)
                .onUpdate(() => {
                    v1.setInstanceYOffset(i, tweenObject.y);
                })
                .onComplete(() => {
                    // Create a temporary emissive flash mesh at the merge location
                    // Compute local position of instance in vector2
                    const xLocal = (i - VECTOR_LENGTH / 2) * SPHERE_DIAMETER;
                    const flashPos = new THREE.Vector3(xLocal, 0, 0);
                    v2.group.localToWorld(flashPos);
                    // Prepare a bright emissive material for flash
                    const flashMat = new THREE.MeshStandardMaterial({
                        color: 0xffffff,
                        emissive: 0xffffff,
                        emissiveIntensity: 3.0,
                        metalness: 0.0,
                        roughness: 0.0
                    });
                    const flashMesh = new THREE.Mesh(v2.mesh.geometry, flashMat);
                    flashMesh.position.copy(flashPos);
                    scene.add(flashMesh);
                    // Schedule revert + update to sum color
                    setTimeout(() => {
                        const sum = v1.rawData[i] + v2.rawData[i];
                        v2.rawData[i] = sum;
                        v2.normalizedData[i] = sum; // For now just store raw
                        const newColor = mapValueToColor(sum);
                        v2.setInstanceColor(i, newColor);

                        // Hide the moving instance by putting it far away
                        v1.setInstanceYOffset(i, -999);
                        // Remove flash mesh
                        scene.remove(flashMesh);
                        flashMat.dispose();
                    }, flashDuration);
                });
            moveTween.start();
        }
    }

    startAdditionAnimation(vector1, vector2);

    // --- Animation Loop ---
    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        if (typeof TWEEN !== 'undefined' && typeof TWEEN.update === 'function') {
            TWEEN.update();
        }
        composer.render();
    }

    function onWindowResize() {
        const width = window.innerWidth;
        const height = window.innerHeight;
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height);
        composer.setSize(width, height);
    }
    window.addEventListener('resize', onWindowResize);

    animate();

    return () => {
        window.removeEventListener('resize', onWindowResize);
        if (renderer.domElement.parentElement) {
            renderer.domElement.parentElement.removeChild(renderer.domElement);
        }
        controls.dispose();
        vector1.dispose();
        vector2.dispose();
        renderer.dispose();
        composer.dispose();
    };
} 