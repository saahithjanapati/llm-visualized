import * as THREE from 'three';
// OrbitControls is loaded via CDN and attached to THREE.OrbitControls
// TWEEN is loaded via CDN and is available globally as TWEEN

import { MHSAAnimation } from '../src/animations/MHSAAnimation.js';
import { VectorVisualizationInstancedPrism } from '../src/components/VectorVisualizationInstancedPrism.js';
import { NUM_HEAD_SETS_LAYER, VECTOR_LENGTH_PRISM, MHA_MATRIX_PARAMS, GLOBAL_ANIM_SPEED_MULT, HEAD_VECTOR_STOP_BELOW } from '../src/utils/constants.js';

let scene, camera, renderer, controls, clock, mhsaAnimation;
let lanes = [];
const NUM_TEST_LANES = 1; // For simplicity, we'll visualize one 'lane' of vectors

function init() {
    // Scene setup
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);

    // Camera setup
    camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 10000);
    camera.position.set(0, MHA_MATRIX_PARAMS.height * 2, 800); // Adjusted for better view

    // Renderer setup
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    document.body.appendChild(renderer.domElement);

    // Controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, MHA_MATRIX_PARAMS.height / 2, 0); // Target the center of where matrices will be

    // Lights
    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.9);
    dirLight.position.set(20, 50, 30);
    scene.add(dirLight);

    // Clock
    clock = new THREE.Clock();

    // Instantiate MHSAAnimation
    // Centering the MHSA block around X=0. Adjust mhsaBaseY so matrices are roughly centered vertically.
    const mhsaBaseY = 0; // MHA matrices will be centered around MHA_MATRIX_PARAMS.height / 2
    const branchX = 0;   // Center the whole MHSA block at X=0
    mhsaAnimation = new MHSAAnimation(scene, branchX, mhsaBaseY, clock, 'temp');

    // Prepare mock 'lanes' data with vectors in position
    prepareMockLanes();

    // Set phase and initiate pass-through animation
    mhsaAnimation.mhaPassThroughPhase = 'ready_for_parallel_pass_through';
    console.log("MHSA Test: Initiating pass-through animations with mock lanes:", lanes);
    mhsaAnimation.initiateParallelHeadPassThroughAnimations(lanes);

    // Handle window resize
    window.addEventListener('resize', onWindowResize);

    // Start animation loop
    animate();
}

function prepareMockLanes() {
    const testData = Array.from({ length: VECTOR_LENGTH_PRISM }, () => Math.random() * 0.2 - 0.1); // Softer initial data
    const yPos = mhsaAnimation.headStopY;

    for (let i = 0; i < NUM_TEST_LANES; i++) {
        // For this test, all lanes will share the same Z position for simplicity.
        // In a real scenario, each lane would have a different Z.
        const zPos = 0; 
        const laneData = {
            upwardCopies: [], // For K vectors
            sideCopies: [],   // For Q and V vectors
            zPos: zPos,
            // Add other properties if MHSAAnimation.update might expect them, though for pass-through these are key
            horizPhase: 'finishedHeads', // To prevent MHSAAnimation.update from trying to move them
            travellingVec: null 
        };

        for (let headIdx = 0; headIdx < NUM_HEAD_SETS_LAYER; headIdx++) {
            const headCoords = mhsaAnimation.headCoords[headIdx];
            if (!headCoords) {
                console.error(`MHSA Test: No headCoords found for headIndex ${headIdx}`);
                continue;
            }

            // K Vector (goes into upwardCopies)
            const kVec = new VectorVisualizationInstancedPrism(testData.slice(), new THREE.Vector3(headCoords.k, yPos, zPos));
            scene.add(kVec.group);
            laneData.upwardCopies[headIdx] = kVec;

            // Q Vector (goes into sideCopies)
            const qVec = new VectorVisualizationInstancedPrism(testData.slice(), new THREE.Vector3(headCoords.q, yPos, zPos));
            scene.add(qVec.group);
            laneData.sideCopies.push({
                vec: qVec,
                targetX: headCoords.q,
                type: 'Q',
                matrixRef: mhsaAnimation.mhaVisualizations[headIdx * 3 + 0], // Q matrix is at index headIdx*3 + 0
                headIndex: headIdx
            });

            // V Vector (goes into sideCopies)
            const vVec = new VectorVisualizationInstancedPrism(testData.slice(), new THREE.Vector3(headCoords.v, yPos, zPos));
            scene.add(vVec.group);
            laneData.sideCopies.push({
                vec: vVec,
                targetX: headCoords.v,
                type: 'V',
                matrixRef: mhsaAnimation.mhaVisualizations[headIdx * 3 + 2], // V matrix is at index headIdx*3 + 2
                headIndex: headIdx
            });
        }
        lanes.push(laneData);
    }
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
    requestAnimationFrame(animate);
    const deltaTime = clock.getDelta();
    const timeNow = performance.now();

    TWEEN.update(); // Update all tweens
    controls.update();
    
    // Though tweens drive the pass-through, calling update might be important for other potential logic in MHSAAnimation.
    // For this specific test focusing on pass-through tweens, it might not be strictly necessary if all vectors are static before tweens.
    if (mhsaAnimation && mhsaAnimation.mhaPassThroughPhase === 'parallel_pass_through_active') {
        // mhsaAnimation.update(deltaTime, timeNow, lanes); // Call if needed for any non-Tween logic within MHSAAnimation
    }

    renderer.render(scene, camera);
}

// Run the initialization function once the DOM is loaded
if (document.readyState === 'loading') { // Loading hasn't finished yet
    document.addEventListener('DOMContentLoaded', init);
} else { // `DOMContentLoaded` has already fired
    init();
} 