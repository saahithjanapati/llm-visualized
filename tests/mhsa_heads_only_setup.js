import * as THREE from 'three';
// OrbitControls is loaded via CDN and attached to THREE.OrbitControls
// TWEEN is loaded via CDN and is available globally as TWEEN

import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { MHSAAnimation } from '../src/animations/MHSAAnimation.js';
import { VectorVisualizationInstancedPrism } from '../src/components/VectorVisualizationInstancedPrism.js';
import { NUM_HEAD_SETS_LAYER, NUM_VECTOR_LANES, VECTOR_LENGTH_PRISM, MHA_MATRIX_PARAMS } from '../src/utils/constants.js';

let scene, camera, renderer, controls, clock, mhsaAnimation;
let lanes = [];
const NUM_TEST_LANES = NUM_VECTOR_LANES; // create lanes for every available lane

function init() {
    // Scene setup
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);

    // Camera
    camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 10000);
    camera.position.set(0, MHA_MATRIX_PARAMS.height * 2, 800);

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    document.body.appendChild(renderer.domElement);

    // Controls
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, MHA_MATRIX_PARAMS.height / 2, 0);

    // Lighting
    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.9);
    dirLight.position.set(20, 50, 30);
    scene.add(dirLight);

    // Clock
    clock = new THREE.Clock();

    // Instantiate MHSAAnimation in heads_only mode
    const branchX = 0;
    const mhsaBaseY = 0;
    mhsaAnimation = new MHSAAnimation(scene, branchX, mhsaBaseY, clock, 'heads_only', {trainMotion: true});

    // Prepare mock data representing already-aligned vectors beneath each head
    prepareMockLanes();

    // Kick off the pass-through stage immediately
    mhsaAnimation.mhaPassThroughPhase = 'ready_for_parallel_pass_through';
    mhsaAnimation.initiateParallelHeadPassThroughAnimations(lanes);

    window.addEventListener('resize', onWindowResize);
    animate();
}

function prepareMockLanes() {
    const dataTemplate = Array.from({ length: VECTOR_LENGTH_PRISM }, () => Math.random() * 0.2 - 0.1);
    const yPos = mhsaAnimation.headStopY;

    // Z coordinate for each lane aligns with slit positions inside the matrices.
    const slitSpacing = MHA_MATRIX_PARAMS.depth / (NUM_VECTOR_LANES + 1);
    for (let i = 0; i < NUM_TEST_LANES; i++) {
        const laneZ = -MHA_MATRIX_PARAMS.depth / 2 + slitSpacing * (i + 1);
        const lane = {
            upwardCopies: [],
            sideCopies: [],
            zPos: laneZ,
            horizPhase: 'finishedHeads',
            travellingVec: null
        };

        for (let headIdx = 0; headIdx < NUM_HEAD_SETS_LAYER; headIdx++) {
            const coords = mhsaAnimation.headCoords[headIdx];
            if (!coords) continue;

            // K vector
            const kVec = new VectorVisualizationInstancedPrism(dataTemplate.slice(), new THREE.Vector3(coords.k, yPos, laneZ));
            scene.add(kVec.group);
            lane.upwardCopies[headIdx] = kVec;

            // Q vector
            const qVec = new VectorVisualizationInstancedPrism(dataTemplate.slice(), new THREE.Vector3(coords.q, yPos, laneZ));
            scene.add(qVec.group);
            lane.sideCopies.push({
                vec: qVec,
                targetX: coords.q,
                type: 'Q',
                matrixRef: mhsaAnimation.mhaVisualizations[headIdx * 3 + 0],
                headIndex: headIdx
            });

            // V vector
            const vVec = new VectorVisualizationInstancedPrism(dataTemplate.slice(), new THREE.Vector3(coords.v, yPos, laneZ));
            scene.add(vVec.group);
            lane.sideCopies.push({
                vec: vVec,
                targetX: coords.v,
                type: 'V',
                matrixRef: mhsaAnimation.mhaVisualizations[headIdx * 3 + 2],
                headIndex: headIdx
            });
        }
        lanes.push(lane);
    }
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
    requestAnimationFrame(animate);
    const delta = clock.getDelta();

    TWEEN.update();
    controls.update();

    renderer.render(scene, camera);
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}