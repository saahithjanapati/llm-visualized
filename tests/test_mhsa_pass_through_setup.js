import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { MHSAAnimation } from '../src/animations/MHSAAnimation.js';
import { VectorVisualizationInstancedPrism } from '../src/components/VectorVisualizationInstancedPrism.js';
import { NUM_HEAD_SETS_LAYER, MHA_MATRIX_PARAMS, HEAD_VECTOR_STOP_BELOW } from '../src/utils/constants.js';

// Basic Three.js setup
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);
const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 10000);
camera.position.set(0, 150, 500); // Adjusted for a typical view of MHSA

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 50, 0); // Adjust target to center on MHSA generally

scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
dirLight.position.set(10, 30, 20);
scene.add(dirLight);

const clock = new THREE.Clock();

// --- MHSAAnimation Setup ---
const branchX = 0; // Centering the MHSA block for this test
const mhsaBaseY = 0; // Base Y for the MHSA block
const mhsaAnimation = new MHSAAnimation(scene, branchX, mhsaBaseY, clock);

// --- Mocking 'lanes' data as if vectors are in position --- 
const numTestLanes = 5; // Create 5 lanes of vectors
const mockLanes = [];

// Calculate headStopY based on MHSAAnimation's internal calculation if possible,
// or use the imported constants directly.
// mhsa_matrix_center_y = mhsaBaseY + MHA_MATRIX_PARAMS.height / 2;
// headStopY = mhsa_matrix_center_y - HEAD_VECTOR_STOP_BELOW;
const headStopY = (mhsaBaseY + MHA_MATRIX_PARAMS.height / 2) - HEAD_VECTOR_STOP_BELOW;


for (let i = 0; i < numTestLanes; i++) {
    const lane = {
        // Space out the 5 lanes along the Z axis.
        // Calculate spacing similar to LayerNorm holes if needed, or use a fixed value.
        // For this test, a fixed stagger should be fine.
        // LayerAnimation uses: const slitSpacing = LN_PARAMS.depth / (numVectors + 1);
        // const zPos = -LN_PARAMS.depth / 2 + slitSpacing * (i + 1);
        // Let's use a simpler stagger for clarity in this isolated test.
        zPos: (i - (numTestLanes -1) / 2) * 80, // Spread around z=0
        originalVec: null, // Not relevant for this test phase
        movingVec: null,   // Not relevant
        multTarget: null,  // Not relevant
        normStarted: true,
        multStarted: true,
        multDone: true,
        horizPhase: 'finishedHeads', // Indicate prior phases are done
        resultVec: null, // Not directly used but part of lane structure
        travellingVec: null, // Main travelling vec that spawns upwardCopies
        upwardCopies: [],
        headIndex: NUM_HEAD_SETS_LAYER, // Signifies all heads visited by travellingVec
        finalAscend: false,
        sideCopies: [],
        // Trails not strictly needed for pass-through logic but good to have
        origTrail: { points: [], geometry: { setDrawRange: () => {}, attributes: { position: { needsUpdate: false, setXYZ: () => {} } } } },
        branchTrail: { points: [], geometry: { setDrawRange: () => {}, attributes: { position: { needsUpdate: false, setXYZ: () => {} } } } },
        upwardTrails: [],
        sideTrails: []
    };

    // Create mock vectors for each head, as if they rose and spawned side copies
    for (let headIdx = 0; headIdx < NUM_HEAD_SETS_LAYER; headIdx++) {
        // K vector (center vector for the head)
        const kVecData = Array.from({ length: 100 }, () => Math.random() * 2 - 1);
        const kVec_X = mhsaAnimation.headsCentersX[headIdx];
        const kVec = new VectorVisualizationInstancedPrism(kVecData, new THREE.Vector3(kVec_X, headStopY, lane.zPos));
        kVec.userData = {}; // Initialize userData object directly on the instance
        kVec.userData.sideSpawned = true; // Mark as side copies having been spawned
        kVec.userData.headIndex = headIdx;
        scene.add(kVec.group);
        lane.upwardCopies.push(kVec);

        // Q vector (side copy)
        const qVecData = Array.from({ length: 100 }, () => Math.random() * 2 - 1);
        const qVec_X = mhsaAnimation.headCoords[headIdx].q;
        const qVec = new VectorVisualizationInstancedPrism(qVecData, new THREE.Vector3(qVec_X, headStopY, lane.zPos));
        qVec.userData = {}; // Initialize userData object directly on the instance (if needed in future)
        scene.add(qVec.group);
        lane.sideCopies.push({
            vec: qVec,
            targetX: qVec_X,
            type: 'Q',
            matrixRef: mhsaAnimation.mhaVisualizations[headIdx * 3 + 0], // Q matrix
            headIndex: headIdx
        });

        // V vector (side copy)
        const vVecData = Array.from({ length: 100 }, () => Math.random() * 2 - 1);
        const vVec_X = mhsaAnimation.headCoords[headIdx].v;
        const vVec = new VectorVisualizationInstancedPrism(vVecData, new THREE.Vector3(vVec_X, headStopY, lane.zPos));
        vVec.userData = {}; // Initialize userData object directly on the instance (if needed in future)
        scene.add(vVec.group);
        lane.sideCopies.push({
            vec: vVec,
            targetX: vVec_X,
            type: 'V',
            matrixRef: mhsaAnimation.mhaVisualizations[headIdx * 3 + 2], // V matrix
            headIndex: headIdx
        });
    }
    mockLanes.push(lane);
}

// Manually set the phase and trigger the animation
if (mhsaAnimation.areAllMHAVectorsInPosition(mockLanes)) {
    console.log("Mocked state: All MHSA vectors are in position.");
    mhsaAnimation.mhaPassThroughPhase = 'ready_for_parallel_pass_through';
    mhsaAnimation.initiateParallelHeadPassThroughAnimations(mockLanes);
} else {
    console.error("Error in mocking: areAllMHAVectorsInPosition is false. Check mockLanes setup.");
    // Log details of why it might be false
    mockLanes.forEach((lane, laneIdx) => {
        if (!lane.upwardCopies || lane.upwardCopies.length !== NUM_HEAD_SETS_LAYER) {
            console.error(`Lane ${laneIdx}: Incorrect upwardCopies count. Expected ${NUM_HEAD_SETS_LAYER}, Got ${lane.upwardCopies?.length}`);
        }
        lane.upwardCopies.forEach((kVec, headIdx) => {
            if (!kVec || Math.abs(kVec.group.position.y - headStopY) > 0.1) {
                console.error(`Lane ${laneIdx}, Head ${headIdx} K-Vec: Incorrect Y position or missing. Expected ${headStopY}, Got ${kVec?.group.position.y}`);
            }
            if (!kVec?.userData.sideSpawned) {
                 console.error(`Lane ${laneIdx}, Head ${headIdx} K-Vec: sideSpawned is false.`);
            }
        });
        if (!lane.sideCopies || lane.sideCopies.length !== NUM_HEAD_SETS_LAYER * 2) {
             console.error(`Lane ${laneIdx}: Incorrect sideCopies count. Expected ${NUM_HEAD_SETS_LAYER * 2}, Got ${lane.sideCopies?.length}`);
        }
        lane.sideCopies.forEach(sc => {
            if (!sc || !sc.vec) console.error(`Lane ${laneIdx}: Missing sideCopy vector object.`);
            if (Math.abs(sc.vec.group.position.y - headStopY) > 0.1) {
                 console.error(`Lane ${laneIdx}, SideCopy Type ${sc.type}, Head ${sc.headIndex}: Incorrect Y. Expected ${headStopY}, Got ${sc.vec.group.position.y}`);
            }
            if (Math.abs(sc.vec.group.position.x - sc.targetX) > 0.1) {
                 console.error(`Lane ${laneIdx}, SideCopy Type ${sc.type}, Head ${sc.headIndex}: Incorrect X. Expected ${sc.targetX}, Got ${sc.vec.group.position.x}`);
            }
        });
    });
}

// Animation loop
function animate() {
    requestAnimationFrame(animate);
    const deltaTime = clock.getDelta();
    // We don't call mhsaAnimation.update() here as we want to test only the pass-through
    // which is initiated once and then driven by TWEEN.
    if (typeof TWEEN !== 'undefined') {
        TWEEN.update();
    }
    controls.update();
    renderer.render(scene, camera);
}

animate();

// Handle window resize
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}); 