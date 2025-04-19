import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { VectorVisualization } from '../components/VectorVisualization.js';
import { VECTOR_LENGTH } from '../utils/constants.js';
import { mapValueToColor } from '../utils/colors.js';

// Assumes TWEEN is loaded globally

export function initVectorMultiplicationAnimation(containerElement) {
    // --- Basic Three.js setup (Similar to Addition) ---
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 5, 15);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    containerElement.appendChild(renderer.domElement);

    // --- Post Processing (Bloom for flash) ---
    const composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));
    const bloomPass = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 1.2, 0.5, 0.8);
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
    const vector1 = new VectorVisualization(data1, new THREE.Vector3(0, yOffset, 0));
    const vector2 = new VectorVisualization(data2, new THREE.Vector3(0, -yOffset, 0));
    vector1.data = [...data1]; // Store original data copies
    vector2.data = [...data2];
    scene.add(vector1.group);
    scene.add(vector2.group);

    // --- Core Multiplication Animation Logic ---
    function startMultiplicationAnimation(vec1, vec2) {
        if (typeof TWEEN === 'undefined') {
            console.error("Global TWEEN object not loaded!"); return;
        }

        const duration = 750; // Movement duration
        const flashDuration = 150;
        const vectorLength = vec1.ellipses.length;

        if (vectorLength !== vec2.ellipses.length || !vec1.data || !vec2.data) {
            console.error("Vector length mismatch or data missing."); return;
        }

        console.log("Starting vector multiplication animation sequence...");

        let moveTweensCompleted = 0;
        const originalMaterialsVec2 = []; // Store original materials before flashing

        // Start all movement tweens simultaneously (no delay)
        for (let i = 0; i < vectorLength; i++) {
            const ellipse1 = vec1.ellipses[i];
            const ellipse2 = vec2.ellipses[i];
            if (!ellipse1 || !ellipse2) continue;

            const targetPosition = new THREE.Vector3();
            ellipse2.getWorldPosition(targetPosition);
            const localTargetPosition = ellipse1.parent.worldToLocal(targetPosition.clone());

            const moveTween = new TWEEN.Tween(ellipse1.position)
                .to({ y: localTargetPosition.y }, duration)
                .easing(TWEEN.Easing.Quadratic.InOut)
                // No delay - all start together
                .onComplete(() => {
                    moveTweensCompleted++;
                    // When the LAST move tween finishes, trigger the simultaneous flash
                    if (moveTweensCompleted === vectorLength) {
                        triggerSimultaneousFlash(vec1, vec2);
                    }
                });
            moveTween.start();
        }
    }

    // Function to handle the simultaneous flash and color update
    function triggerSimultaneousFlash(vec1, vec2) {
        const flashDuration = 150;
        const vectorLength = vec2.ellipses.length;
        const originalMaterialsVec2 = [];

         console.log("All move tweens completed. Triggering flash...");

        // 1. Store originals and set all to white/bright
        for (let i = 0; i < vectorLength; i++) {
            const ellipse2 = vec2.ellipses[i];
            if (!ellipse2) continue;
            originalMaterialsVec2[i] = {
                color: ellipse2.material.color.clone(),
                emissive: ellipse2.material.emissive.clone(),
                emissiveIntensity: ellipse2.material.emissiveIntensity
            };
            ellipse2.material.color.set(0xffffff);
            ellipse2.material.emissive.set(0xffffff);
            ellipse2.material.emissiveIntensity = 1.0; // Make it bright for bloom
        }

        // 2. Create a single tween for the flash duration
        const flashEffectTween = new TWEEN.Tween({})
            .to({}, flashDuration)
            .onComplete(() => {
                 console.log("Flash complete. Setting final colors...");
                // 3. Set final colors based on product
                for (let i = 0; i < vectorLength; i++) {
                    const ellipse1 = vec1.ellipses[i];
                    const ellipse2 = vec2.ellipses[i];
                    if (!ellipse1 || !ellipse2 || !originalMaterialsVec2[i]) continue;

                    const product = vec1.data[i] * vec2.data[i]; // Use original data for calculation
                    vec2.data[i] = product; // Update target vector's data (optional, depends if needed later)
                    const newColor = mapValueToColor(product);

                    ellipse2.material.color.copy(newColor);
                    ellipse2.material.emissive.copy(newColor);
                    ellipse2.material.emissiveIntensity = originalMaterialsVec2[i].emissiveIntensity; // Restore original intensity
                    ellipse1.visible = false; // Hide source ellipse
                }
                // Consider triggering a reset/loop here if needed in the future
            });
        flashEffectTween.start();
    }

    // Start the animation sequence automatically
    startMultiplicationAnimation(vector1, vector2);

    // --- Animation Loop (Same as Addition) ---
    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        TWEEN.update();
        composer.render();
    }

    // --- Handle Resize (Same as Addition) ---
    function onWindowResize() {
        const width = window.innerWidth;
        const height = window.innerHeight;
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height);
        composer.setSize(width, height);
    }
    window.addEventListener('resize', onWindowResize);

    // --- Start Loop ---
    animate();

    // --- Return Cleanup Function (Same as Addition) ---
    return () => {
        console.log("Cleaning up Vector Multiplication Animation scene...");
        window.removeEventListener('resize', onWindowResize);
        if (renderer.domElement.parentElement) {
             renderer.domElement.parentElement.removeChild(renderer.domElement);
        }
        controls.dispose();
        vector1.dispose();
        vector2.dispose();
        scene.traverse(object => {
            if (object.geometry) object.geometry.dispose();
            if (object.material) {
                if (Array.isArray(object.material)) {
                    object.material.forEach(material => material.dispose());
                } else {
                    object.material.dispose();
                }
            }
        });
        renderer.dispose();
        // Safely dispose composer passes
        composer.passes.forEach(pass => { if(pass.dispose) pass.dispose(); });
        // composer.dispose(); // EffectComposer itself doesn't have a standard dispose
    };
} 