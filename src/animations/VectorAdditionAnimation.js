import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { VectorVisualization } from '../components/VectorVisualization.js'; // Adjusted path
import { VECTOR_LENGTH } from '../utils/constants.js'; // Adjusted path
import { mapValueToColor } from '../utils/colors.js'; // Adjusted path

// Note: Assumes TWEEN is loaded globally via <script> tag

export function initVectorAdditionAnimation(containerElement) {
    // --- Basic Three.js setup ---
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);

    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 5, 15);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    containerElement.appendChild(renderer.domElement); // Attach to container

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
    const vector1 = new VectorVisualization(data1, new THREE.Vector3(0, yOffset, 0));
    const vector2 = new VectorVisualization(data2, new THREE.Vector3(0, -yOffset, 0));
    vector1.data = data1; // Store original data
    vector2.data = data2;
    scene.add(vector1.group);
    scene.add(vector2.group);

    // --- Core Animation Logic (Merged from vector_addition_animation.js) ---
    function startAdditionAnimation(vec1, vec2) {
        if (typeof TWEEN === 'undefined' || typeof TWEEN.Tween !== 'function' || typeof TWEEN.Easing === 'undefined') {
             console.error("Global TWEEN object (or Tween/Easing) not loaded!");
             return;
        }

        const duration = 750; // Decreased
        const flashDuration = 150;
        const delayBetweenCubes = 75; // Decreased
        const vectorLength = vec1.ellipses.length;

        if (vectorLength !== vec2.ellipses.length || !vec1.data || !vec2.data) {
            console.error("Vector length mismatch or data missing.");
            return;
        }

        console.log("Starting vector addition animation sequence...");

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
                .delay(i * delayBetweenCubes)
                .onComplete(() => {
                    const originalColor = ellipse2.material.color.clone();
                    const originalEmissive = ellipse2.material.emissive.clone();
                    const originalIntensity = ellipse2.material.emissiveIntensity;

                    ellipse2.material.color.set(0xffffff);
                    ellipse2.material.emissive.set(0xffffff);
                    ellipse2.material.emissiveIntensity = 1.0; // Make it bright for bloom

                    const flashTween = new TWEEN.Tween(ellipse2.material)
                        .to({}, flashDuration) // Dummy target, only need onComplete
                        .onComplete(() => {
                            const sum = vec1.data[i] + vec2.data[i];
                            vec2.data[i] = sum; // Update data for target vector
                            const newColor = mapValueToColor(sum); // Use imported function
                            ellipse2.material.color.copy(newColor);
                            ellipse2.material.emissive.copy(newColor); // Keep emissive for color
                            ellipse2.material.emissiveIntensity = originalIntensity; // Restore original intensity
                            ellipse1.visible = false;
                        });
                    flashTween.start();
                });
            moveTween.start();
        }
        console.log("All animation tweens created and started.");
    }

    // Start the animation sequence automatically
    startAdditionAnimation(vector1, vector2);

    // --- Animation Loop ---
    function animate() {
        requestAnimationFrame(animate);

        if (controls && typeof controls.update === 'function') {
            controls.update();
        }

        if (typeof TWEEN !== 'undefined' && typeof TWEEN.update === 'function') {
            TWEEN.update();
        } else {
            console.error("Global TWEEN.update not available!");
            // Maybe stop the loop? return;
        }

        if (composer && typeof composer.render === 'function') {
            composer.render();
        } else {
            console.error("Composer not available!");
            // Maybe stop the loop? return;
        }
    }

    // --- Handle Resize ---
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

    // --- Return Cleanup Function ---
    return () => {
        console.log("Cleaning up Vector Addition Animation scene...");
        window.removeEventListener('resize', onWindowResize);
        if (renderer.domElement.parentElement) {
             renderer.domElement.parentElement.removeChild(renderer.domElement);
        }
        controls.dispose();
        // Dispose geometries, materials
        vector1.dispose();
        vector2.dispose();
        // Clean up scene children explicitly if needed
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
        composer.dispose(); // Assuming composer has a dispose method or needs manual pass disposal
    };
} 