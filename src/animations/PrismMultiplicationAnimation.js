import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { VectorVisualizationInstancedPrism } from '../components/VectorVisualizationInstancedPrism.js';
import { VECTOR_LENGTH_PRISM, HIDE_INSTANCE_Y_OFFSET } from '../utils/constants.js'; // Use PRISM constants
import { mapValueToColor } from '../utils/colors.js';

// Assumes TWEEN is loaded globally

export function initPrismMultiplicationAnimation(containerElement) {
    // --- Basic Three.js setup ---
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 2000); // Adjusted far plane for potentially larger scenes
    camera.position.set(0, 5, 25); // Adjusted camera position for prism visualization
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
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7); // Slightly increased ambient light
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.9); // Slightly increased directional light
    directionalLight.position.set(5, 10, 7.5);
    scene.add(directionalLight);

    // --- Controls ---
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxDistance = 1000; // Allow zooming out further

    // --- Vectors Setup ---
    // For prisms, yOffset might need to be larger due to their height
    const yOffset = 5; // Adjusted yOffset
    const vector1 = new VectorVisualizationInstancedPrism(null, new THREE.Vector3(0, yOffset, 0));
    const vector2 = new VectorVisualizationInstancedPrism(null, new THREE.Vector3(0, -yOffset, 0));
    // rawData is already stored in the prism class, and normalizedData is calculated
    scene.add(vector1.group);
    scene.add(vector2.group);

    // --- Core Multiplication Animation Logic ---
    function startMultiplicationAnimation(vec1, vec2) {
        if (typeof TWEEN === 'undefined') {
            console.error("Global TWEEN object not loaded!"); return;
        }

        const duration = 750; // Movement duration
        const vectorLength = VECTOR_LENGTH_PRISM; // Use prism vector length

        if (!vec1.rawData || !vec2.rawData || vec1.rawData.length !== vectorLength || vec2.rawData.length !== vectorLength) {
            console.error("Vector data missing or length mismatch."); return;
        }

        console.log("Starting prism multiplication animation sequence...");

        // Store initial Y positions for each instance of vec1
        const initialYPositionsVec1 = [];
        for (let i = 0; i < vectorLength; i++) {
            const matrix = new THREE.Matrix4();
            vec1.mesh.getMatrixAt(i, matrix);
            const position = new THREE.Vector3();
            matrix.decompose(position, new THREE.Quaternion(), new THREE.Vector3());
            initialYPositionsVec1[i] = position.y;
        }
        
        const targetYPositionVec2 = vec2.group.position.y + vec2.getUniformHeight() / 2; // Target the top of vec2 instances

        let moveTweensCompleted = 0;

        for (let i = 0; i < vectorLength; i++) {
            // We animate the yOffset property for setInstanceAppearance
            // The actual y position will be initialYPositionsVec1[i] + animatedOffset.y
            // Target y is targetYPositionVec2, so the offset needs to be targetYPositionVec2 - initialYPositionsVec1[i]
            
            const animatedOffset = { y: 0 }; // Start with zero offset
            const targetOffset = targetYPositionVec2 - initialYPositionsVec1[i];
            
            const moveTween = new TWEEN.Tween(animatedOffset)
                .to({ y: targetOffset }, duration)
                .easing(TWEEN.Easing.Quadratic.InOut)
                .onUpdate(() => {
                    // Apply yOffset for movement. Color remains default during movement.
                    vec1.setInstanceAppearance(i, animatedOffset.y, null);
                })
                .onComplete(() => {
                    moveTweensCompleted++;
                    if (moveTweensCompleted === vectorLength) {
                        triggerSimultaneousFlash(vec1, vec2, initialYPositionsVec1);
                    }
                });
            moveTween.start();
        }
    }

    // Function to handle the simultaneous flash and color update
    function triggerSimultaneousFlash(vec1, vec2, initialYPositionsVec1) {
        const flashDuration = 150;
        const vectorLength = VECTOR_LENGTH_PRISM;
        const originalColorsVec2 = [];

        console.log("All move tweens completed. Triggering flash...");

        // 1. Store original colors of vec2 and set all to white/bright for flash
        for (let i = 0; i < vectorLength; i++) {
            const currentColor = new THREE.Color();
            vec2.mesh.getColorAt(i, currentColor);
            originalColorsVec2[i] = currentColor.clone();
            
            // Use setInstanceAppearance to change color temporarily
            vec2.setInstanceAppearance(i, 0, new THREE.Color(0xffffff)); // yOffset = 0, bright color
        }

        // 2. Create a single tween for the flash duration
        const flashEffectTween = new TWEEN.Tween({})
            .to({}, flashDuration)
            .onComplete(() => {
                console.log("Flash complete. Setting final colors and hiding vec1 instances...");
                // 3. Set final colors on vec2 based on product, and "hide" vec1 instances
                for (let i = 0; i < vectorLength; i++) {
                    if (!vec1.rawData[i] || !vec2.rawData[i]) continue;

                    const product = vec1.rawData[i] * vec2.rawData[i];
                    vec2.rawData[i] = product; // Update underlying data
                    // Note: We might need to re-normalize and update geometry if heights change.
                    // For this animation, we'll just update color.
                    // A more complex version might snap to new heights based on the product.
                    
                    // const newColor = mapValueToColor(product); // mapValueToColor needs to handle new data range
                    // vec2.setInstanceAppearance(i, 0, newColor); // yOffset = 0, final color -- This line is removed

                    // "Hide" instance from vec1 by moving it far away
                    vec1.setInstanceAppearance(i, HIDE_INSTANCE_Y_OFFSET, null); 
                }
                // Optional: If vec2's heights should change based on new data, call updateDataAndSnapVisuals
                // vec2.updateDataAndSnapVisuals(vec2.rawData); 
                // For now, only color changes.

                // Update vec2 colors based on the new rawData (product results) using a gradient from 30 samples
                vec2.updateKeyColorsFromData(vec2.rawData, 30);

                console.log("Prism multiplication animation finished.");
            });
        flashEffectTween.start();
    }

    // Start the animation sequence automatically
    // Ensure rawData is populated before starting
    if (vector1.rawData.length > 0 && vector2.rawData.length > 0) {
        startMultiplicationAnimation(vector1, vector2);
    } else {
        // Fallback if data generation was delayed (shouldn't happen with current constructor)
        setTimeout(() => startMultiplicationAnimation(vector1, vector2), 100);
    }


    // --- Animation Loop ---
    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        TWEEN.update();
        composer.render();
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
        console.log("Cleaning up Prism Multiplication Animation scene...");
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
        composer.passes.forEach(pass => { if(pass.dispose) pass.dispose(); });
    };
} 