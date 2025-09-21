import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { VectorVisualization } from '../components/VectorVisualization.js'; // Adjusted path
import { VECTOR_LENGTH } from '../utils/constants.js'; // Adjusted path
import { mapValueToColor } from '../utils/colors.js'; // Adjusted path

const ADDITION_ACCENT_COLORS = [
    0xff6f61,
    0xffd166,
    0x06d6a0,
    0x7c3aed,
];

// Note: Assumes TWEEN is loaded globally via <script> tag

export function initVectorAdditionAnimation(containerElement) {
    // --- Basic Three.js setup ---
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b1026);

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
        1.4, 0.55, 0.8 // strength, radius, threshold
    );
    composer.addPass(bloomPass);

    // --- Lighting ---
    const ambientLight = new THREE.AmbientLight(0xfff4e6, 0.65);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.7);
    directionalLight.position.set(6, 11, 7);
    scene.add(directionalLight);
    const rimLight = new THREE.PointLight(0x5eead4, 0.65, 60);
    rimLight.position.set(-8, 4, 6);
    scene.add(rimLight);
    const warmLight = new THREE.PointLight(0xff85a1, 0.6, 70);
    warmLight.position.set(8, 6, -5);
    scene.add(warmLight);

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

    const accentTop = new THREE.Color(ADDITION_ACCENT_COLORS[0]);
    const accentBottom = new THREE.Color(ADDITION_ACCENT_COLORS[1]);
    vector1.ellipses.forEach((ellipse, idx) => {
        const tint = ellipse.material.color.clone().lerp(accentTop, 0.3 + (idx / VECTOR_LENGTH) * 0.2);
        ellipse.material.color.copy(tint);
        ellipse.material.emissive.copy(accentTop);
        ellipse.material.emissiveIntensity = 0.35;
        ellipse.material.needsUpdate = true;
    });
    vector2.ellipses.forEach((ellipse, idx) => {
        const tint = ellipse.material.color.clone().lerp(accentBottom, 0.35 + ((VECTOR_LENGTH - idx) / VECTOR_LENGTH) * 0.2);
        ellipse.material.color.copy(tint);
        ellipse.material.emissive.copy(accentBottom);
        ellipse.material.emissiveIntensity = 0.35;
        ellipse.material.needsUpdate = true;
    });

    const wiggleState = [
        { group: vector1.group, speed: 1.2, amplitude: 0.18, offset: Math.random() * Math.PI * 2 },
        { group: vector2.group, speed: 1.1, amplitude: 0.16, offset: Math.random() * Math.PI * 2 },
    ];

    // --- Core Animation Logic (Merged from vector_addition_animation.js) ---
    function startAdditionAnimation(vec1, vec2) {
        if (typeof TWEEN === 'undefined' || typeof TWEEN.Tween !== 'function' || typeof TWEEN.Easing === 'undefined') {
             console.error("Global TWEEN object (or Tween/Easing) not loaded!");
             return;
        }

        const duration = 620;
        const flashDuration = 220;
        const delayBetweenCubes = 65;
        const anticipationDuration = 180;
        const settleDuration = 320;
        const anticipationDepth = 0.6;
        const overshootDistance = 0.55;
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

            const initialY = ellipse1.position.y;
            const targetY = localTargetPosition.y;
            const startDelay = i * delayBetweenCubes;

            const anticipationState = { t: 0 };
            const moveState = { t: 0 };
            const settleState = { t: 0 };

            const settleTween = new TWEEN.Tween(settleState)
                .to({ t: 1 }, settleDuration)
                .easing(TWEEN.Easing.Bounce.Out)
                .onUpdate(({ t }) => {
                    const y = THREE.MathUtils.lerp(targetY + overshootDistance, targetY, t);
                    const relax = 1 - t;
                    ellipse1.position.y = y;
                    const scale = 1 + 0.25 * relax;
                    ellipse1.scale.set(scale, 1 - 0.25 * relax, scale);
                })
                .onComplete(() => {
                    ellipse1.position.y = targetY;
                    ellipse1.scale.set(1, 1, 1);
                    ellipse1.visible = false;
                });

            const triggerMerge = () => {
                const originalIntensity = ellipse2.material.emissiveIntensity;

                const flashState = { t: 0 };
                const flashTween = new TWEEN.Tween(flashState)
                    .to({ t: 1 }, flashDuration)
                    .easing(TWEEN.Easing.Sinusoidal.InOut)
                    .yoyo(true)
                    .repeat(1)
                    .onStart(() => {
                        ellipse2.material.color.set(0xffffff);
                        ellipse2.material.emissive.set(0xffffff);
                        ellipse2.material.emissiveIntensity = 1.1;
                    })
                    .onUpdate(({ t }) => {
                        const pulse = THREE.MathUtils.lerp(1, 1.45, t);
                        ellipse2.scale.setScalar(pulse);
                        bloomPass.strength = 1.3 + 0.5 * t;
                    })
                    .onComplete(() => {
                        const sum = vec1.data[i] + vec2.data[i];
                        vec2.data[i] = sum; // Update data for target vector
                        const newColor = mapValueToColor(sum);
                        const accent = new THREE.Color(ADDITION_ACCENT_COLORS[(i + 2) % ADDITION_ACCENT_COLORS.length]);
                        newColor.lerp(accent, 0.35);
                        ellipse2.material.color.copy(newColor);
                        ellipse2.material.emissive.copy(newColor);
                        ellipse2.material.emissiveIntensity = originalIntensity;
                        ellipse2.scale.setScalar(1);
                        ellipse2.material.needsUpdate = true;
                        bloomPass.strength = 1.4;
                    });
                flashTween.start();
            };

            const moveTween = new TWEEN.Tween(moveState)
                .to({ t: 1 }, duration)
                .easing(TWEEN.Easing.Cubic.Out)
                .delay(startDelay + anticipationDuration)
                .onUpdate(({ t }) => {
                    const y = THREE.MathUtils.lerp(initialY - anticipationDepth, targetY + overshootDistance, t);
                    ellipse1.position.y = y;
                    const stretch = THREE.MathUtils.lerp(1, 1.35, t);
                    ellipse1.scale.set(1 - 0.25 * t, stretch, 1 - 0.25 * t);
                })
                .onComplete(() => {
                    triggerMerge();
                    settleTween.start();
                });

            const anticipationTween = new TWEEN.Tween(anticipationState)
                .to({ t: 1 }, anticipationDuration)
                .delay(startDelay)
                .easing(TWEEN.Easing.Quadratic.Out)
                .onUpdate(({ t }) => {
                    const y = THREE.MathUtils.lerp(initialY, initialY - anticipationDepth, t);
                    ellipse1.position.y = y;
                    const squash = THREE.MathUtils.lerp(1, 0.7, t);
                    ellipse1.scale.set(1 + 0.25 * t, squash, 1 + 0.25 * t);
                })
                .onComplete(() => {
                    ellipse1.position.y = initialY - anticipationDepth;
                });

            anticipationTween.start();
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

        const time = performance.now() * 0.001;
        wiggleState.forEach((state, idx) => {
            const { group, speed, amplitude, offset } = state;
            const phase = time * speed + offset;
            group.position.x = Math.sin(phase) * amplitude;
            group.rotation.z = Math.sin(phase * 1.2) * 0.18;
        });
        bloomPass.strength = 1.35 + 0.25 * (Math.sin(time * 0.8) + 1) * 0.5;

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