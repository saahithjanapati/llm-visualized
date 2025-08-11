import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { LayerNormalizationVisualization } from '../components/LayerNormalizationVisualization.js';
import { VectorVisualizationInstancedPrism } from '../components/VectorVisualizationInstancedPrism.js';
import { WeightMatrixVisualization } from '../components/WeightMatrixVisualization.js';
import { MHSAAnimation } from './MHSAAnimation.js';
// Trail functionality removed – no-ops keep API intact
function createTrailLine() {
  return {
    line: { material: { opacity: 0, needsUpdate: false } },
    geometry: {
      attributes: { position: { setXYZ: () => {}, needsUpdate: false } },
      setDrawRange: () => {},
      computeBoundingSphere: () => {},
    },
    positions: [],
    points: [],
    isFrozen: false,
  };
}
function updateTrail() {}


import { PrismLayerNormAnimation } from '../animations/PrismLayerNormAnimation.js';
import { 
    VECTOR_LENGTH,
    VECTOR_LENGTH_PRISM,
    LN_TO_MHA_GAP,
    BRANCH_X,
    LAYER_NORM_1_Y_POS,
    LN_PARAMS,
    NUM_HEAD_SETS_LAYER,
    HEAD_SET_GAP_LAYER,
    MHA_INTERNAL_MATRIX_SPACING,
    MHA_MATRIX_PARAMS,
    MLP_VECTOR_MULTIPLIER,
    MLP_MATRIX_STYLE_PARAMS,
    MLP_D_MODEL_VISUAL_DEPTH,
    ANIM_OFFSET_Y_ORIGINAL_SPAWN,
    ANIM_MEET_Y_OFFSET_ABOVE_LN1,
    ANIM_RISE_SPEED_ORIGINAL,
    ANIM_HORIZ_SPEED,
    ANIM_RISE_SPEED_INSIDE_LN,
    ANIM_RISE_SPEED_POST_SPLIT_LN1,
    ANIM_RISE_SPEED_POST_SPLIT_LN2,
    
    ANIM_RISE_SPEED_HEAD,
    HEAD_VECTOR_STOP_BELOW,
    GLOBAL_ANIM_SPEED_MULT,
    SIDE_COPY_DELAY_MS,
    SIDE_COPY_HORIZ_SPEED,
    HIDE_INSTANCE_Y_OFFSET,
    LAYER_NORM_2_Y_POS,
    LN2_TO_MLP_GAP,
    MLP_INTER_MATRIX_GAP,
    MLP_MATRIX_PARAMS_UP,
    MLP_MATRIX_PARAMS_DOWN,
    ORIGINAL_TO_PROCESSED_GAP,
    DECORATIVE_FADE_MS,
    DECORATIVE_FADE_DELAY_MS
} from '../utils/constants.js';
import { mapValueToColor } from '../utils/colors.js';

// NOTE: Requires global TWEEN.js (loaded separately via <script>)

// Use live binding of GLOBAL_ANIM_SPEED_MULT at each use; do not cache

export function initLayerAnimation(container) {
    // -------------------------------------------------------------------------
    //  Basic Three.js setup
    // -------------------------------------------------------------------------
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);

    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 10000);
    camera.position.set(2140, 150, 3500);

    let renderer;
    if (container instanceof HTMLCanvasElement) {
        renderer = new THREE.WebGLRenderer({ canvas: container, antialias: true });
    } else {
        renderer = new THREE.WebGLRenderer({ antialias: true });
        container.appendChild(renderer.domElement);
    }
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);

    // Initialize the clock earlier
    const clock = new THREE.Clock(); 

    // Add pause/resume support for tab visibility
    let isPaused = false;
    function pauseAnimation() {
        if (isPaused) return;
        isPaused = true;
        clock.stop();
    }
    function resumeAnimation() {
        if (!isPaused) return;
        isPaused = false;
        clock.start();
    }
    const visibilityHandler = () => {
        if (document.hidden) pauseAnimation();
        else resumeAnimation();
    };
    document.addEventListener('visibilitychange', visibilityHandler);

    // -------------------------------------------------------------------------
    //  Post-processing (subtle bloom for emissive flashes)
    // -------------------------------------------------------------------------
    const composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));
    const bloomPass = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 1.0, 0.4, 0.85);
    composer.addPass(bloomPass);

    // -------------------------------------------------------------------------
    //  Controls & lights
    // -------------------------------------------------------------------------
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(2140, 66, 0);

    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(10, 30, 10);
    scene.add(dirLight);

    // -------------------------------------------------------------------------
    //  Component Parameters (Values from LayerAnimationConstants.js)
    // -------------------------------------------------------------------------
    // const VERTICAL_GAP_COMPONENTS = 20; // Now from constants

    // LayerNorm1 (existing)
    // const lnParams = { ... }; // Now LN_PARAMS from constants
    // const branchX = 80; // Now BRANCH_X from constants
    // const layerNorm1_Y_pos = -10; // Now LAYER_NORM_1_Y_POS from constants

    const layerNorm1 = new LayerNormalizationVisualization(
        new THREE.Vector3(BRANCH_X, LAYER_NORM_1_Y_POS, 0),
        LN_PARAMS.width,
        LN_PARAMS.height,
        LN_PARAMS.depth,
        LN_PARAMS.wallThickness,
        LN_PARAMS.numberOfHoles,
        LN_PARAMS.holeWidth,
        LN_PARAMS.holeWidthFactor
    );
    // Match LayerNorm appearance to the light gray used for MHSA head matrices
    layerNorm1.setColor(new THREE.Color(0x404040));
    layerNorm1.group.children.forEach(child => {
        if (child.material) {
            child.material.transparent = true;
            child.material.opacity = 0.7; // Same opacity as MHSA head matrices
        }
    });
    scene.add(layerNorm1.group);

    // -------------------------------------------------------------------------
    //  Multi-Head Self-Attention (MHSA)
    // -------------------------------------------------------------------------
    // const NUM_HEAD_SETS_LAYER = 12; // Now from constants
    // const HEAD_SET_GAP_LAYER = 10;   // Now from constants
    // const MHA_INTERNAL_MATRIX_SPACING = 37.5; // Now from constants
    // const mhaMatrixParams = { ... }; // Now MHA_MATRIX_PARAMS from constants

    const ln1_top_y = layerNorm1.group.position.y + LN_PARAMS.height / 2;
    const MHSA_BASE_Y = ln1_top_y + LN_TO_MHA_GAP;
    // const mha_matrix_center_y = MHSA_BASE_Y + MHA_MATRIX_PARAMS.height / 2; // Moved to MHSAAnimation

    // const mhaVisualizations = []; // Managed by MHSAAnimation
    // const headsCentersX = []; // Managed by MHSAAnimation
    // const headCoords = []; // Managed by MHSAAnimation

    const darkGrayColor = new THREE.Color(0x404040);
    const matrixOpacity = 0.7;

    // -------------------------------------------------------------------------
    //  SECOND LayerNorm (LN2) and MLP block (Up-/Down-projection)
    // -------------------------------------------------------------------------
    // Colour/opacity settings reused from other matrices
    const mlpDarkGray = new THREE.Color(0x404040);
    const mlpMatrixOpacity = 1.0;

    // ── LayerNorm 2 ───────────────────────────────────────────────────────────
    const layerNorm2 = new LayerNormalizationVisualization(
        new THREE.Vector3(BRANCH_X, LAYER_NORM_2_Y_POS, 0),
        LN_PARAMS.width,
        LN_PARAMS.height,
        LN_PARAMS.depth,
        LN_PARAMS.wallThickness,
        LN_PARAMS.numberOfHoles,
        LN_PARAMS.holeWidth,
        LN_PARAMS.holeWidthFactor
    );
    // Apply the same light gray style to the second LayerNorm
    layerNorm2.setColor(new THREE.Color(0x404040));
    layerNorm2.group.children.forEach(child => {
        if (child.material) {
            child.material.transparent = false;
            child.material.opacity = mlpMatrixOpacity;
        }
    });
    scene.add(layerNorm2.group);

    // Compute helper Y positions for the stacked MLP matrices
    const ln2_top_y = layerNorm2.group.position.y + LN_PARAMS.height / 2;

    // Basic dimensions – tweak as desired for visual clarity
    const MLP_MATRIX_WIDTH = 150;
    const MLP_MATRIX_HEIGHT = 40; // Taller than MHSA matrices for distinction
    const MLP_MATRIX_DEPTH = LN_PARAMS.depth; // Keep depth equal so slits line up in Z

    const mlpMatrixUp_centerY = ln2_top_y + LN2_TO_MLP_GAP + MLP_MATRIX_HEIGHT / 2;
    const mlpMatrixDown_centerY = mlpMatrixUp_centerY + MLP_MATRIX_HEIGHT / 2 + MLP_INTER_MATRIX_GAP + MLP_MATRIX_HEIGHT / 2;

    // ── Up-projection matrix (d_model → 4·d_model) ──
    const mlpMatrixUp = new WeightMatrixVisualization(
        null,
        new THREE.Vector3(BRANCH_X, mlpMatrixUp_centerY, 0),
        MLP_MATRIX_PARAMS_UP.width,
        MLP_MATRIX_PARAMS_UP.height,
        MLP_MATRIX_PARAMS_UP.depth,
        MLP_MATRIX_PARAMS_UP.topWidthFactor,
        MLP_MATRIX_PARAMS_UP.cornerRadius,
        MLP_MATRIX_PARAMS_UP.numberOfSlits,
        MLP_MATRIX_PARAMS_UP.slitWidth,
        MLP_MATRIX_PARAMS_UP.slitDepthFactor,
        MLP_MATRIX_PARAMS_UP.slitBottomWidthFactor,
        MLP_MATRIX_PARAMS_UP.slitTopWidthFactor
    );
    mlpMatrixUp.setColor(mlpDarkGray);
    mlpMatrixUp.group.children.forEach(child => {
        if (child.material) {
            child.material.transparent = false;
            child.material.opacity = mlpMatrixOpacity;
        }
    });
    scene.add(mlpMatrixUp.group);

    // ── Down-projection matrix (4·d_model → d_model) ──
    const mlpMatrixDown = new WeightMatrixVisualization(
        null,
        new THREE.Vector3(BRANCH_X, mlpMatrixDown_centerY, 0),
        MLP_MATRIX_PARAMS_DOWN.width,
        MLP_MATRIX_PARAMS_DOWN.height,
        MLP_MATRIX_PARAMS_DOWN.depth,
        MLP_MATRIX_PARAMS_DOWN.topWidthFactor,
        MLP_MATRIX_PARAMS_DOWN.cornerRadius,
        MLP_MATRIX_PARAMS_DOWN.numberOfSlits,
        MLP_MATRIX_PARAMS_DOWN.slitWidth,
        MLP_MATRIX_PARAMS_DOWN.slitDepthFactor,
        MLP_MATRIX_PARAMS_DOWN.slitBottomWidthFactor,
        MLP_MATRIX_PARAMS_DOWN.slitTopWidthFactor
    );
    mlpMatrixDown.setColor(mlpDarkGray);
    mlpMatrixDown.group.children.forEach(child => {
        if (child.material) {
            child.material.transparent = false;
            child.material.opacity = mlpMatrixOpacity;
        }
    });
    scene.add(mlpMatrixDown.group);

    // MHSA Pass-Through Animation State - Reworked
    // let mhaPassThroughPhase = 'positioning_mha_vectors'; // Managed by MHSAAnimation

    // Instantiate MHSAAnimation
    const mhsaAnimation = new MHSAAnimation(scene, BRANCH_X, MHSA_BASE_Y, clock, 'temp'); // temp mode for in-progress behaviour

    // The MHSA visualization setup loop is now in MHSAAnimation constructor
    /*
    for (let i = 0; i < NUM_HEAD_SETS_LAYER; i++) {
        // ... Entire MHSA setup loop removed ...
    }
    */

    // -------------------------------------------------------------------------
    //  Main path parameters (Values from LayerAnimationConstants.js)
    // -------------------------------------------------------------------------
    // const offsetY = 10; // Now ANIM_OFFSET_Y_ORIGINAL_SPAWN
    const startY = LAYER_NORM_1_Y_POS - LN_PARAMS.height / 2 - ANIM_OFFSET_Y_ORIGINAL_SPAWN;
    // const meetYOffset = 5; // Now ANIM_MEET_Y_OFFSET_ABOVE_LN1
    const meetY = LAYER_NORM_1_Y_POS + LN_PARAMS.height / 2 + ANIM_MEET_Y_OFFSET_ABOVE_LN1;

    const branchStartY = startY + 5;

    const numVectors = LN_PARAMS.numberOfHoles;
    const slitSpacing = LN_PARAMS.depth / (numVectors + 1);

    // Motion speeds (Now from constants)
    // const riseSpeedOriginal = 3;
    // const horizSpeed = 15;
    // const riseSpeedInsideLN = 6;
    // const mergeGap = 7; // Not currently used due to animation change

    const originals = [];
    const lanes = [];

    // -------------------------------------------------------------------------
    //  State tracking for LayerNorm2 appearance so it doesn't revert to dark/black
    //  once vectors have completely exited the block.
    // -------------------------------------------------------------------------
    let ln2ColorLocked = false;      // becomes true after the final bright transition
    let ln2LockedColor = null;       // stores the color to keep (e.g., brightYellow)
    let ln2LastColor = new THREE.Color(0x404040);
    let ln2LastOpacity = 1.0;



    for (let i = 0; i < numVectors; i++) {
        const zPos = -LN_PARAMS.depth / 2 + slitSpacing * (i + 1);

        // ---------- Original vector on main (centre) path ----------
        const data = Array.from({ length: VECTOR_LENGTH }, () => Math.random() * 2 - 1); // Reverted to random data
        const origVec = new VectorVisualizationInstancedPrism(data, new THREE.Vector3(0, startY, zPos));
        scene.add(origVec.group);
        originals.push(origVec);

        // ---------- Duplicate moving vector (will branch) ----------
        const movingVec = new VectorVisualizationInstancedPrism(data, new THREE.Vector3(0, startY, zPos));
        movingVec.updateDataInternal(data);
        scene.add(movingVec.group);

        // Start hidden – will appear once branch begins
        movingVec.group.visible = false;

        // ---------- Static vectors inside LayerNorm ----------
        const multTarget = new VectorVisualizationInstancedPrism(data.slice(), new THREE.Vector3(BRANCH_X, 3.3, zPos));
        scene.add(multTarget.group);

        // ---------- Static vectors inside SECOND LayerNorm (LN2) ----------
        // Place the static vector higher inside LN2 so it matches the relative
        // vertical position of the vector inside LN1 (≈13.3 units above the
        // centre of the block).  This avoids the distracting "pop-in" that was
        // visible when the camera first entered the second LayerNorm.
        const multTargetLN2 = new VectorVisualizationInstancedPrism(data.slice(), new THREE.Vector3(
            BRANCH_X,
            LAYER_NORM_2_Y_POS + 13.3, // align with LN1 relative placement
            zPos
        ));
        // Keep the multiplication target inside LN2 visible from the beginning
        multTargetLN2.group.visible = true;
        scene.add(multTargetLN2.group);

        // Create layer norm animation controller for this lane's vector
        const normAnimation = new PrismLayerNormAnimation(movingVec);

        // Create trails (now uses utility function, passing the scene)
        const origTrail = createTrailLine(scene);
        const branchTrail = createTrailLine(scene);

        lanes.push({
            zPos,
            originalVec: origVec,
            movingVec,
            multTarget,
            multTargetLN2,
            normAnimation,
            // Pipeline flags
            normStarted: false,
            multStarted: false,
            multDone: false,
            // Horizontal / merge states
            horizPhase: 'waiting', // waiting | right | insideLN | moveLeft | merged
            resultVec: null,
            mergeStarted: false,
            origTrail,
            branchTrail,
            branchTrailLN2: null,
            // New MHSA traversal properties
            travellingVec: null,
            upwardCopies: [],
            headIndex: 0,
            finalAscend: false,
            sideCopies: [],
            sideTrails: [],
            upwardTrails: [],
            // ---------------- LN2 pipeline flags / objects ----------------
            ln2Phase: 'notStarted', // notStarted | preRise | right | insideLN | riseToMLP | done
            postAdditionVec: null,  // residual vector after MHSA addition
            movingVecLN2: null,
            normAnimationLN2: null,
            normStartedLN2: false,
            multDoneLN2: false,
            resultVecLN2: null,
        });
    }

    // -------------------------------------------------------------------------
    //  Helper: multiplication animation (copied from pipeline)
    // -------------------------------------------------------------------------
    function startMultiplicationAnimation(vec1, vec2, onCompleteCallback) {
        // vec1 is movingVec (source), vec2 is multTarget (destination/target of multiplication)
        if (typeof TWEEN === 'undefined') {
            console.error("Global TWEEN object not loaded!");
            if (onCompleteCallback) onCompleteCallback();
            return;
        }

        const flashDuration = 150 / GLOBAL_ANIM_SPEED_MULT; // Adjusted by speed
        const vectorLength = VECTOR_LENGTH_PRISM; // Prisms use VECTOR_LENGTH_PRISM

        if (!vec1.rawData || !vec2.rawData || vec1.rawData.length !== vectorLength || vec2.rawData.length !== vectorLength) {
            console.error("Vector data missing or length mismatch for prism multiplication.",
                          `Vec1 length: ${vec1.rawData?.length}, Vec2 length: ${vec2.rawData?.length}, Expected: ${vectorLength}`);
            if (onCompleteCallback) onCompleteCallback();
            return;
        }

        // 1. Store original colors of vec2 and set all to white/bright for flash
        // This assumes vec1 (movingVec) is already in position.
        // The flash happens on vec2 (multTarget).
        for (let i = 0; i < vectorLength; i++) {
            // Use setInstanceAppearance to change color temporarily. yOffset = 0 means no positional change.
            vec2.setInstanceAppearance(i, 0, new THREE.Color(0xffffff)); 
        }

        // 2. Create a single tween for the flash duration
                    new TWEEN.Tween({})
                        .to({}, flashDuration)
            .onComplete(() => {
                // 3. Calculate product, update vec2.rawData, set final colors on vec2, and "hide" vec1 instances
                for (let i = 0; i < vectorLength; i++) {
                    // Ensure rawData elements exist before multiplication
                    if (typeof vec1.rawData[i] !== 'number' || typeof vec2.rawData[i] !== 'number') {
                        console.warn(`Skipping multiplication for index ${i} due to missing data: vec1[${i}]=${vec1.rawData[i]}, vec2[${i}]=${vec2.rawData[i]}`);
                        // Set product to a default or skip update for this element if data is bad
                        // For now, if data is bad on one, we may result in NaN or error if not careful.
                        // Let's assume valid numbers or that multiplication handles it gracefully (e.g. undefined * number = NaN)
                        // vec2.rawData[i] might remain unchanged or become NaN.
                    }
                    
                    const product = vec1.rawData[i] * vec2.rawData[i];
                    vec2.rawData[i] = product; // Update underlying data of vec2 (multTarget)

                    // "Hide" instance from vec1 (movingVec) by moving it far away
                    vec1.setInstanceAppearance(i, HIDE_INSTANCE_Y_OFFSET, null); 
                }

                // Update vec2 (multTarget) colors based on the new rawData (product results) using a gradient
                vec2.updateKeyColorsFromData(vec2.rawData, 30); // Use 30 samples for the gradient

                if (onCompleteCallback) {
                    onCompleteCallback();
                }
            })
            .start();
    }

    // -------------------------------------------------------------------------
    //  Resize handler
    // -------------------------------------------------------------------------
    function onWindowResize() {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
        composer.setSize(window.innerWidth, window.innerHeight);
    }
    window.addEventListener('resize', onWindowResize);

    // -------------------------------------------------------------------------
    //  Animation loop
    // -------------------------------------------------------------------------
    // const clock = new THREE.Clock(); // Moved earlier
    // const headStopY = mhsa_matrix_center_y - HEAD_VECTOR_STOP_BELOW; // Defined in MHSAAnimation
    // const mhaPassThroughTargetY = mhsa_matrix_center_y + MHA_MATRIX_PARAMS.height / 2 + 20; // Defined in MHSAAnimation
    // const mhaPassThroughDuration = 2000 / SPEED_MULT; // Defined in MHSAAnimation
    // const outputVectorLength = 64; // Defined in MHSAAnimation
    // const mhaResultRiseOffsetY = 50; // Defined in MHSAAnimation
    // const mhaResultRiseDuration = 500 / SPEED_MULT; // Defined in MHSAAnimation

    // Colors for matrix flash & final state (MHSA specific colors moved to MHSAAnimation)
    // const brightGreen = new THREE.Color(0x33FF33);
    // const darkTintedGreen = new THREE.Color(0x002200);
    // const brightBlue = new THREE.Color(0x6666FF);
    // const darkTintedBlue = new THREE.Color(0x000022);
    // const brightRed = new THREE.Color(0xFF3333);
    // const darkTintedRed = new THREE.Color(0x220000);

    // Helper function to check if all MHSA vectors are in their start positions for pass-through
    // function areAllMHAVectorsInPosition() { ... } // Moved to MHSAAnimation.js

    // Helper function to animate a single vector and its matrix
    // function animateVectorMatrixPassThrough(...) { ... } // Moved to MHSAAnimation.js

    // Main function to initiate parallel pass-through for all heads
    // function initiateParallelHeadPassThroughAnimations(allLanes) { ... } // Moved to MHSAAnimation.js

    function animate() {
        requestAnimationFrame(animate);
        if (isPaused) {
            return;
        }
        const deltaTime = clock.getDelta();
        const timeNow = performance.now();

        // --- LayerNorm Appearance Control ---
        const darkGray = new THREE.Color(0x404040);
        const lightYellow = new THREE.Color(0xFFFF99); // For semi-transparent state
        const brightYellow = new THREE.Color(0xFFFF00); // For final opaque state
        const opaqueOpacity = 1.0;
        const semiTransparentOpacity = 0.6;
        const exitTransitionRange = 10; // Y-distance over which the exit transition occurs

        const firstMovingVecY = lanes.length > 0 ? lanes[0].movingVec.group.position.y : startY;
        const bottomY_ln1_abs = LAYER_NORM_1_Y_POS - LN_PARAMS.height / 2;
        const midY_ln1_abs = LAYER_NORM_1_Y_POS;
        const topY_ln1_abs = LAYER_NORM_1_Y_POS + LN_PARAMS.height / 2;

        // LayerNorm2 positions - used for detecting when vectors pass through LN2
        const bottomY_ln2_abs = LAYER_NORM_2_Y_POS - LN_PARAMS.height / 2;
        const midY_ln2_abs = LAYER_NORM_2_Y_POS;
        const topY_ln2_abs = LAYER_NORM_2_Y_POS + LN_PARAMS.height / 2;

        let targetColor = darkGray;
        let targetOpacity = opaqueOpacity;
        let lerpFactor = 0;

        if (firstMovingVecY >= bottomY_ln1_abs && firstMovingVecY < midY_ln1_abs) {
            // Entering (Bottom Half): Lerp from Dark Gray to Light Yellow / Semi-Transparent
            lerpFactor = (firstMovingVecY - bottomY_ln1_abs) / (midY_ln1_abs - bottomY_ln1_abs);
            targetColor = darkGray.clone().lerp(lightYellow, lerpFactor);
            targetOpacity = opaqueOpacity + (semiTransparentOpacity - opaqueOpacity) * lerpFactor;
        } else if (firstMovingVecY >= midY_ln1_abs && firstMovingVecY < topY_ln1_abs) {
            // Inside (Top Half): Stay at Light Yellow / Semi-Transparent
            targetColor = lightYellow;
            targetOpacity = semiTransparentOpacity;
        } else if (firstMovingVecY >= topY_ln1_abs) {
            // Exiting: Lerp from Light Yellow / Semi-Transparent to Bright Yellow / Opaque
            lerpFactor = Math.min(1, (firstMovingVecY - topY_ln1_abs) / exitTransitionRange);
            targetColor = lightYellow.clone().lerp(brightYellow, lerpFactor);
            targetOpacity = semiTransparentOpacity + (opaqueOpacity - semiTransparentOpacity) * lerpFactor;
            // Ensure opacity reaches exactly 1.0 when lerpFactor is 1
            if (lerpFactor >= 1.0) {
                targetOpacity = opaqueOpacity; // Force full opaqueness
            }
        } // Else (below) remains darkGray and opaque

        // Update appearance of the FIRST LayerNorm only.  The second will
        // remain grey until its own vectors reach it later in the pipeline.
        layerNorm1.group.children.forEach(child => {
            if (child instanceof THREE.Mesh && child.material) {
                child.material.transparent = targetOpacity < 1.0;
                child.material.color.copy(targetColor);
                child.material.opacity = targetOpacity;
                child.material.needsUpdate = true;
            }
        });

        // Handle LayerNorm2 color separately, based on MHSA output vectors
        // First find if any vectors have reached LayerNorm2
        let ln2TargetColor = darkGray.clone(); // Ensure it's a clone
        let ln2TargetOpacity = opaqueOpacity;
        let ln2LerpFactor = 0;
        
        // Find the highest Y position of any vector currently moving within or towards LN2
        let highestMovingVecLN2_Y = -Infinity;
        let anyVectorInOrNearLN2 = false;

        lanes.forEach(lane => {
            let vecY_forLN2Color = -Infinity;

            if (lane.movingVecLN2 && lane.movingVecLN2.group.visible) {
                vecY_forLN2Color = lane.movingVecLN2.group.position.y;
            } else if (lane.resultVecLN2 && lane.resultVecLN2.group.visible) {
                // If movingVecLN2 is done/invisible, and resultVecLN2 is active,
                // use resultVecLN2 for color calculation as it's the one exiting LN2.
                vecY_forLN2Color = lane.resultVecLN2.group.position.y;
            }

            if (vecY_forLN2Color > -Infinity) {
                highestMovingVecLN2_Y = Math.max(highestMovingVecLN2_Y, vecY_forLN2Color);
                // Consider a vector "near" LN2 if it's above the bottom boundary or slightly below it
                if (vecY_forLN2Color >= bottomY_ln2_abs - exitTransitionRange) { // Use exitTransitionRange as a buffer
                    anyVectorInOrNearLN2 = true;
                }
            }
        });
        
        // Apply color transitions to LN2 based on the highest vector's position
        if (anyVectorInOrNearLN2 && highestMovingVecLN2_Y > -Infinity) {
            if (highestMovingVecLN2_Y >= bottomY_ln2_abs && highestMovingVecLN2_Y < midY_ln2_abs) {
                // Vector entering bottom half of LN2
                ln2LerpFactor = (highestMovingVecLN2_Y - bottomY_ln2_abs) / (midY_ln2_abs - bottomY_ln2_abs);
                // Clamp lerpFactor to [0, 1] to avoid issues if vecY is slightly outside bounds due to timing
                ln2LerpFactor = Math.max(0, Math.min(1, ln2LerpFactor));
                ln2TargetColor = darkGray.clone().lerp(lightYellow, ln2LerpFactor);
                ln2TargetOpacity = opaqueOpacity + (semiTransparentOpacity - opaqueOpacity) * ln2LerpFactor;
            } else if (highestMovingVecLN2_Y >= midY_ln2_abs && highestMovingVecLN2_Y < topY_ln2_abs) {
                // Vector inside top half of LN2
                ln2TargetColor = lightYellow.clone();
                ln2TargetOpacity = semiTransparentOpacity;
            } else if (highestMovingVecLN2_Y >= topY_ln2_abs) {
                // Vector exiting LN2
                ln2LerpFactor = (highestMovingVecLN2_Y - topY_ln2_abs) / exitTransitionRange;
                // Clamp lerpFactor to [0, 1]
                ln2LerpFactor = Math.max(0, Math.min(1, ln2LerpFactor));
                ln2TargetColor = lightYellow.clone().lerp(brightYellow, ln2LerpFactor);
                ln2TargetOpacity = semiTransparentOpacity + (opaqueOpacity - semiTransparentOpacity) * ln2LerpFactor;
                if (ln2LerpFactor >= 1.0) {
                    ln2TargetOpacity = opaqueOpacity;
                }
            }
            // If highestMovingVecLN2_Y is below bottomY_ln2_abs but still considered "near"
            // it will default to darkGray unless caught by the conditions above, which is fine.
        }
        // If no vectors are in or near LN2, keep the last known colour/opacity so it doesn't snap to dark.
        if (!anyVectorInOrNearLN2) {
            ln2TargetColor.copy(ln2LastColor);
            ln2TargetOpacity = ln2LastOpacity;
        } else {
            // Store for future frames
            ln2LastColor.copy(ln2TargetColor);
            ln2LastOpacity = ln2TargetOpacity;
        }
        
        // If the color has been locked, override the computed target so LN2 keeps
        // its bright appearance.
        if (ln2ColorLocked && ln2LockedColor) {
            ln2TargetColor = ln2LockedColor;
            ln2TargetOpacity = opaqueOpacity;
        }

        // Apply LN2 appearance updates
        layerNorm2.group.children.forEach(child => {
            if (child instanceof THREE.Mesh && child.material) {
                child.material.transparent = ln2TargetOpacity < 1.0;
                child.material.color.copy(ln2TargetColor);
                child.material.opacity = ln2TargetOpacity;
                child.material.needsUpdate = true;
            }
        });

        lanes.forEach((lane, idx) => {
            const { originalVec, movingVec, multTarget } = lane;

            // --- MultTarget Appearance Control ---
            const movingVecY = movingVec.group.position.y;
            // Lerp factor based on movingVec's progress in the bottom half of LN
            // This needs to be relative to layerNorm1's position
            const movingVecY_relativeTo_LN1_bottom = movingVecY - (LAYER_NORM_1_Y_POS - LN_PARAMS.height / 2);
            const ln1_height = LN_PARAMS.height;
            const multLerpFactor = THREE.MathUtils.clamp(movingVecY_relativeTo_LN1_bottom / (ln1_height / 2), 0, 1);

            // TODO: Refactor for InstancedPrism - multTarget appearance
            // The old code iterated ellipses. VectorVisualizationInstancedPrism handles its own appearance.
            // If dynamic appearance is needed beyond default, use setInstanceAppearance or update data.
            /*
            const initialMultEmissiveIntensity = 0.01;
            const finalMultEmissiveIntensity = 0.4; // Slightly brighter than default

            for (let i = 0; i < VECTOR_LENGTH; i++) {
                const ellipse = multTarget.ellipses[i];
                if (ellipse && ellipse.material) {
                    const dataColor = mapValueToColor(multTarget.rawData[i]); // Use rawData
                    const darkDataColor = dataColor.clone().multiplyScalar(0.2); // Darker version

                    // Lerp color from dark to full, lerp emissive intensity
                    ellipse.material.color.copy(darkDataColor).lerp(dataColor, multLerpFactor);
                    ellipse.material.emissive.copy(darkDataColor).lerp(dataColor, multLerpFactor);

                    // Lerp emissive intensity
                    const currentIntensity = initialMultEmissiveIntensity + (finalMultEmissiveIntensity - initialMultEmissiveIntensity) * multLerpFactor;
                    ellipse.material.emissiveIntensity = currentIntensity;

                    // Keep fully opaque
                    ellipse.material.opacity = 1.0;
                    ellipse.material.transparent = false;

                    ellipse.material.needsUpdate = true;
                }
            }
            */

            // -------------------- ORIGINAL VEC RISE --------------------
            const branchFinalY = meetY; // branched vectors end at the merge height
            const originalTargetY = meetY;

            if (originalVec.group.position.y < originalTargetY) {
                originalVec.group.position.y += ANIM_RISE_SPEED_ORIGINAL * GLOBAL_ANIM_SPEED_MULT * deltaTime;
                if (originalVec.group.position.y > originalTargetY) originalVec.group.position.y = originalTargetY;
            }

            // Update original trail: track center prism during merge, else use group position –
            // BUT skip updates while the vector is "frozen" for the addition animation (the
            // MHSAAnimation._startAdditionAnimation method sets `group.userData.stopRise`).

            const ud = originalVec.group.userData || {};
            const isTrailSuspended = ud.stopRise || (typeof ud.skipTrailResumeY === 'number' && originalVec.group.position.y <= ud.skipTrailResumeY);

            if (!isTrailSuspended) {
                if (lane.mergeStarted) { // mergeStarted only used for future merges now
                    // During merge we would normally track the centre prism, but for InstancedPrism the
                    // group position is close enough for the trail visual.
                    updateTrail(lane.origTrail, originalVec.group.position);
                } else {
                    updateTrail(lane.origTrail, originalVec.group.position);
                }

                // Once we've resumed updates beyond the suspension threshold, clean up the flag.
                if (typeof ud.skipTrailResumeY === 'number' && originalVec.group.position.y > ud.skipTrailResumeY) {
                    delete ud.skipTrailResumeY;
                }
            }

            // -------------------- DUPLICATE / MOVING VEC LOGIC --------------------
            switch (lane.horizPhase) {
                case 'waiting': {
                    if (originalVec.group.position.y >= branchStartY) {
                        lane.horizPhase = 'right';
                        movingVec.group.visible = true;
                        movingVec.group.position.y = originalVec.group.position.y; // sync Y
                    }
                    break;
                }
                case 'right': {
                    // Horizontal move to LayerNorm X
                    const dx = ANIM_HORIZ_SPEED * GLOBAL_ANIM_SPEED_MULT * deltaTime;
                    movingVec.group.position.x = Math.min(BRANCH_X, movingVec.group.position.x + dx);
                    if (movingVec.group.position.x >= BRANCH_X) {
                        movingVec.group.position.x = BRANCH_X;
                        lane.horizPhase = 'insideLN';
                    }
                    if (lane.branchTrailLN2) updateTrail(lane.branchTrailLN2, movingVec.group.position);
                    break;
                }
                case 'insideLN': {
                    // ---------------- LayerNorm pipeline behaviour ----------------
                    // Y positions relative to the current LayerNorm (layerNorm1 for now)
                    const currentLN_bottomY_abs = LAYER_NORM_1_Y_POS - LN_PARAMS.height / 2;
                    const currentLN_midY_abs = LAYER_NORM_1_Y_POS;
                    // const currentLN_topY_abs = LAYER_NORM_1_Y_POS + LN_PARAMS.height / 2;

                    // Start normalization when reaching 35% height above bottom of current LN
                    const normStartY_abs = currentLN_bottomY_abs + (LN_PARAMS.height * 0.35);
                    if (!lane.normStarted && movingVec.group.position.y >= normStartY_abs) {
                        // Start normalization animation 
                        lane.normAnimation.start(movingVec.rawData.slice());
                        lane.normStarted = true;
                    }

                    // Update normalization visuals
                    if (lane.normStarted && lane.normAnimation) {
                        lane.normAnimation.update(deltaTime);
                    }

                    // Move up (only when not actively normalizing)
                    const normAnimating = lane.normStarted && lane.normAnimation.isAnimating;
                    if (!lane.multStarted && !normAnimating) {
                        movingVec.group.position.y += ANIM_RISE_SPEED_INSIDE_LN * GLOBAL_ANIM_SPEED_MULT * deltaTime;
                    }

                    // Trigger multiplication at centre of current LN
                    if (!lane.multStarted && movingVec.group.position.y >= currentLN_midY_abs) {
                        lane.multStarted = true;
                        startMultiplicationAnimation(movingVec, multTarget, () => {
                            lane.multDone = true;
                            movingVec.group.visible = false;
                            // Create result vector immediately after multiplication
                            if (!lane.resultVec) {
                                // Hide the multiplication target vector
                                multTarget.group.visible = false;
                                const resultData = [...multTarget.rawData]; // Use rawData from multTarget after multiplication
                                // const resultVec = new VectorVisualization(resultData, multTarget.group.position.clone());
                                const resultVec = new VectorVisualizationInstancedPrism(resultData, multTarget.group.position.clone());
                                // resultVec.data = [...resultData];
                                scene.add(resultVec.group);
                                // Copy material appearance from multTarget
                                // TODO: Refactor for InstancedPrism - Appearance transfer for instanced prisms.
                                // This might involve setting similar subsection counts/colors or copying instance user data if applicable.
                                // For now, resultVec will have its default appearance.
                                /*
                                for (let i = 0; i < VECTOR_LENGTH; i++) {
                                    if (multTarget.ellipses[i] && resultVec.ellipses[i]) {
                                        resultVec.ellipses[i].material.color.copy(multTarget.ellipses[i].material.color);
                                        resultVec.ellipses[i].material.emissive.copy(multTarget.ellipses[i].material.emissive);
                                        resultVec.ellipses[i].material.emissiveIntensity = multTarget.ellipses[i].material.emissiveIntensity;
                                    }
                                }
                                */
                                lane.resultVec = resultVec;

                                // Rise just above LN top
                                const finalY = branchFinalY; // branchFinalY is meetY
                                const distance = finalY - resultVec.group.position.y;
                                const riseDuration = (distance / (ANIM_RISE_SPEED_INSIDE_LN * GLOBAL_ANIM_SPEED_MULT)) * 1000;

                                new TWEEN.Tween(resultVec.group.position)
                                    .to({ y: finalY }, riseDuration)
                                    .easing(TWEEN.Easing.Linear.None)
                                    .onComplete(() => {
                                        // Vector has passed LN1 and is at branchX, finalY
                                        // Now, this vector needs to interact with MHSA
                                        // The 'travelMHSA' phase and related logic are handled by MHSAAnimation.update
                                        lane.horizPhase = 'travelMHSA'; 
                                        lane.travellingVec = resultVec;
                                        // lane.headIndex = 0; // MHSAAnimation will initialize this if needed when phase starts
                                        // lane.upwardCopies = []; // MHSAAnimation will manage this
                                    })
                                    .start();
                            }
                        });
                    }
                    if (lane.branchTrailLN2) updateTrail(lane.branchTrailLN2, movingVec.group.position);
                    break;
                }
                case 'moveLeft': {
                    // if (!lane.mergeStarted && lane.resultVec && lane.resultVec.group.position.x <= 0.01) {
                    //     lane.mergeStarted = true;
                    //     startAdditionAnimation(originalVec, lane.resultVec, () => {
                    //         lane.resultVec.group.visible = false;
                    //         lane.horizPhase = 'merged';
                    //     });
                    // }
                    break;
                }
                // MHSA specific phases are now handled by mhsaAnimation.update()
                /*
                case 'travelMHSA': { ... } // Moved to MHSAAnimation
                case 'finishedHeads': { ... } // Moved to MHSAAnimation
                */
                case 'merged': 
                default:
                    break;
            }

            // --- Upward movement for copies under heads --- // Moved to MHSAAnimation
            /*
            const headStopY = mhsa_matrix_center_y - HEAD_VECTOR_STOP_BELOW;
            if (lane.upwardCopies && lane.upwardCopies.length) { ... }
            if (lane.finalAscend && lane.travellingVec) { ... }
            */

            // Spawn side copies once centre vector settled (with delay) // Moved to MHSAAnimation
            /*
            if (lane.upwardCopies) { ... }
            */

            // Move side copies horizontally and vertically align // Moved to MHSAAnimation
            /*
            if (mhaPassThroughPhase === 'positioning_mha_vectors' && lane.sideCopies && lane.sideCopies.length) { ... }
            */

            // Determine which branched object position to follow for trail
            let branchPos = null;
            const centerIndex = Math.floor(VECTOR_LENGTH / 2);
            // During addition inside LayerNorm, follow the center ellipse movement
            if (lane.multStarted && lane.multDone && !lane.resultVec) {
                // const centerEllipse = lane.multTarget.ellipses[centerIndex]; // No ellipses
                // const worldPos = new THREE.Vector3();
                // centerEllipse.getWorldPosition(worldPos);
                // branchPos = worldPos;
                // TODO: Refactor for InstancedPrism - If a specific point on the prism is needed, calculate it.
                // For now, using group position of multTarget (which is now the result's source).
                branchPos = lane.multTarget.group.position;
            } else if (lane.resultVecLN2 && lane.resultVecLN2.group.visible) {
                branchPos = lane.resultVecLN2.group.position;
            } else if (lane.resultVec && lane.resultVec.group.visible) {
                branchPos = lane.resultVec.group.position;
            } else if (lane.movingVecLN2 && lane.movingVecLN2.group.visible) {
                branchPos = lane.movingVecLN2.group.position;
            } else if (lane.expandedVecGroup && lane.expandedVecGroup.visible) {
                branchPos = lane.expandedVecGroup.position;
            } else if (lane.movingVec.group.visible) {
                branchPos = lane.movingVec.group.position;
            }
            if (branchPos) {
                const activeTrail = lane.branchTrailLN2 || lane.branchTrail;
                updateTrail(activeTrail, branchPos);
            }

            // ---------------------------------------------------------------------
            //  SECOND LAYERNORM / MLP ROUTING  (per-lane ln2Phase state machine)
            // ---------------------------------------------------------------------
            switch (lane.ln2Phase) {
                case 'preRise': {
                    const v = lane.postAdditionVec;
                    if (!v) break;
                    const targetY = bottomY_ln2_abs - 10; // stop a little below LN2
                    if (v.group.position.y < targetY) {
                        v.group.position.y = Math.min(targetY, v.group.position.y + ANIM_RISE_SPEED_POST_SPLIT_LN2 * GLOBAL_ANIM_SPEED_MULT * deltaTime);
                    } else {
                        // Keep the original branch trail active so it continues
                        // to reflect the upward motion of the residual-stream
                        // vector while the duplicate travels through LN2/MLP.
                        // (Previously we froze the trail at this point.)

                        if (lane.branchTrail) {
                            lane.branchTrail.isFrozen = false;
                        }

                        // ------------------------------------------------------------------
                        //  Let residual-stream vectors keep rising in parallel with the
                        //  LN2/MLP branch by extending the global target height.  Bringing
                        //  them just below the Up-projection matrix creates a pleasing
                        //  stagger before the final merge.
                        // ------------------------------------------------------------------
                        if (mhsaAnimation && typeof mhsaAnimation.finalOriginalY === 'number') {
                            // Extend the residual-stream target to just below the *top* of the
                            // MLP Up-projection matrix (instead of its bottom).  This lets the
                            // original vectors keep rising in parallel with the processed vector
                            // while it travels through the matrix, eliminating the mid-animation
                            // pause the user observed.
                            const newTarget = mlpMatrixUp_centerY + MLP_MATRIX_PARAMS_UP.height / 2 - ORIGINAL_TO_PROCESSED_GAP;
                            if (newTarget > mhsaAnimation.finalOriginalY) {
                                mhsaAnimation.finalOriginalY = newTarget;
                            }
                            mhsaAnimation.postSplitRiseSpeed = ANIM_RISE_SPEED_POST_SPLIT_LN2;
                        }

                        // Now create the duplicate that will travel into LN2.
                        const mv = new VectorVisualizationInstancedPrism(v.rawData.slice(), v.group.position.clone());
                        scene.add(mv.group);
                        lane.movingVecLN2 = mv;
                        lane.normAnimationLN2 = new PrismLayerNormAnimation(mv);
                        // Create a new trail for the LN2 branch so its motion is visualised
                        lane.branchTrailLN2 = createTrailLine(scene);
                        // Slightly lower the opacity so overlapping with the frozen trail doesn't brighten excessively.
                        if (lane.branchTrailLN2 && lane.branchTrailLN2.line && lane.branchTrailLN2.line.material) {
                            
                            lane.branchTrailLN2.line.material.needsUpdate = true;
                        }
                        // Seed the LN2 branch trail with the starting position so it is visible immediately
                        updateTrail(lane.branchTrailLN2, mv.group.position);
                        lane.ln2Phase = 'right';
                    }
                    break;
                }
                case 'right': {
                    const mv = lane.movingVecLN2;
                    if (!mv) break;
                    mv.group.visible = true;
                    const dx2 = ANIM_HORIZ_SPEED * GLOBAL_ANIM_SPEED_MULT * deltaTime;
                    mv.group.position.x = Math.min(BRANCH_X, mv.group.position.x + dx2);
                    if (mv.group.position.x >= BRANCH_X - 0.01) {
                        mv.group.position.x = BRANCH_X;
                        lane.multTargetLN2.group.visible = true;
                        lane.ln2Phase = 'insideLN';
                    }
                    if (lane.branchTrailLN2) updateTrail(lane.branchTrailLN2, mv.group.position);
                    break;
                }
                case 'insideLN': {
                    const mv = lane.movingVecLN2;
                    if (!mv) break;
                    const normStartY2 = bottomY_ln2_abs + LN_PARAMS.height * 0.15; // start a bit sooner for clearer visuals
                    // START NORMALISATION ONCE WE'VE REACHED THE THRESHOLD ---------------------------------
                    if (!lane.normStartedLN2 && mv.group.position.y >= normStartY2) {
                        lane.normAnimationLN2.start(mv.rawData.slice());
                        lane.normStartedLN2 = true;
                    }
                    // Update the normalisation animation each frame
                    if (lane.normStartedLN2 && lane.normAnimationLN2) {
                        lane.normAnimationLN2.update(deltaTime);
                    }
                    // Re-evaluate whether the animation is currently playing *after* possible start/update
                    const isNormAnimating2 = lane.normStartedLN2 && lane.normAnimationLN2 && lane.normAnimationLN2.isAnimating;

                    // Only let the vector rise when it is NOT actively normalising
                    if (!lane.multDoneLN2 && !isNormAnimating2) {
                        mv.group.position.y += ANIM_RISE_SPEED_INSIDE_LN * GLOBAL_ANIM_SPEED_MULT * deltaTime;
                    }
                    if (!lane.multDoneLN2 && mv.group.position.y >= midY_ln2_abs) {
                        lane.multDoneLN2 = true;
                        startMultiplicationAnimation(mv, lane.multTargetLN2, () => {
                            mv.group.visible = false;
                            lane.multTargetLN2.group.visible = false;
                            const resVec = new VectorVisualizationInstancedPrism(lane.multTargetLN2.rawData.slice(), lane.multTargetLN2.group.position.clone());
                            scene.add(resVec.group);
                            lane.resultVecLN2 = resVec;
                            const destY = mlpMatrixUp_centerY - MLP_MATRIX_PARAMS_UP.height / 2 - 10; // just beneath first MLP matrix
                            const dist2 = destY - resVec.group.position.y;
                            const dur2 = (dist2 / (ANIM_RISE_SPEED_INSIDE_LN * GLOBAL_ANIM_SPEED_MULT)) * 1000;
                            new TWEEN.Tween(resVec.group.position)
                                .to({ y: destY }, dur2)
                                .easing(TWEEN.Easing.Linear.None)
                                .onComplete(() => { lane.ln2Phase = 'done'; })
                                .start();
                        });
                    }
                    // Only update trail with the "moving" LN2 vector while it's actually moving.
                    if (!lane.multDoneLN2 && lane.branchTrailLN2) {
                        updateTrail(lane.branchTrailLN2, mv.group.position);
                    }
                    break;
                }
                case 'done':
                default:
                    break;
            }
        });

        // Add MLP Up pass-through animation for vectors that have completed LN2
        lanes.forEach(lane => {
            if (lane.ln2Phase === 'done' && !lane.mlpUpStarted) {
                lane.mlpUpStarted = true;
                lane.mlpUpTrail = createTrailLine(scene);
                const vec = lane.resultVecLN2;
                if (vec) {
                    const bottomY = mlpMatrixUp_centerY - MLP_MATRIX_PARAMS_UP.height / 2;
                    const topY = mlpMatrixUp_centerY + MLP_MATRIX_PARAMS_UP.height / 2;
                    const distance = topY - vec.group.position.y;
                    const duration = (distance / (ANIM_RISE_SPEED_INSIDE_LN * GLOBAL_ANIM_SPEED_MULT)) * 1000;
                    const matrixStartColor = mlpDarkGray.clone();
                    const matrixEndColor = new THREE.Color(0xb07c13); // bright orange
                    new TWEEN.Tween({ t: 0 })
                        .to({ t: 1 }, duration)
                        .easing(TWEEN.Easing.Quadratic.InOut)
                        .onUpdate(o => {
                            const col = matrixStartColor.clone().lerp(matrixEndColor, o.t);
                            mlpMatrixUp.setColor(col);
                            mlpMatrixUp.setEmissive(col, 0.5);
                        })
                        .start();
                    new TWEEN.Tween(vec.group.position)
                        .to({ y: topY }, duration)
                        .easing(TWEEN.Easing.Linear.None)
                        .onUpdate(() => {
                            updateTrail(lane.mlpUpTrail, vec.group.position);
                        })
                        .onStart(() => {
                            // Instantly shrink vector to match matrix width as soon as it enters the matrix
                            vec.group.scale.setScalar(0.6);
                        })
                        .onComplete(() => {
                            // Restore scale before creating expanded vector
                            vec.group.scale.setScalar(0.6);
                            mlpMatrixUp.setColor(matrixEndColor);
                            mlpMatrixUp.setEmissive(matrixEndColor, 0.5);

                            // ------------------------------------------------------------
                            //  Expand to 4× 768-dim segments (simulate 3072-dim output)
                            // ------------------------------------------------------------
                            const segments = 4;
                            const segWidth = vec.getBaseWidthConstant() * vec.getWidthScale() * VECTOR_LENGTH_PRISM;
                            const expandedGroup = new THREE.Group();
                            const segmentVecs = [];

                            for (let s = 0; s < segments; s++) {
                                // Duplicate the raw data for each segment – for visual purposes this is acceptable
                                const segVec = new VectorVisualizationInstancedPrism(vec.rawData.slice(), new THREE.Vector3());

                                // Copy the key color gradient from the source vector so colour scheme matches
                                if (Array.isArray(vec.currentKeyColors) && vec.currentKeyColors.length) {
                                    segVec.currentKeyColors = vec.currentKeyColors.map(c => c.clone());
                                    segVec.updateInstanceGeometryAndColors();
                                }

                                // Position the segment side-by-side along X so the overall width is 4×
                                const localX = (s - (segments - 1) / 2) * segWidth;
                                segVec.group.position.set(localX, 0, 0);
                                expandedGroup.add(segVec.group);
                                segmentVecs.push(segVec);
                            }

                            // Position the whole expanded group where the original vector ended
                            expandedGroup.position.copy(vec.group.position);
                            scene.add(expandedGroup);

                            // Hide the original 768-dim vector
                            vec.group.visible = false;

                            // Store reference on lane for later stages / trail following
                            lane.expandedVecGroup = expandedGroup;
                            lane.expandedVecSegments = segmentVecs;

                            // Continue trail updates with the new group centre
                            if (lane.mlpUpTrail) {
                                updateTrail(lane.mlpUpTrail, expandedGroup.position);
                            }

                            // ------------------------------------------------------------
                            //  EXTRA RISE + PAUSE BEFORE DOWN-PROJECTION
                            // ------------------------------------------------------------
                            const extraRise = 60; // world units
                            const pauseMs = DECORATIVE_FADE_DELAY_MS;  // reuse decorative delay as pause

                            new TWEEN.Tween(expandedGroup.position)
                                .to({ y: expandedGroup.position.y + extraRise }, 800)
                                .easing(TWEEN.Easing.Quadratic.InOut)
                                .onUpdate(() => {
                                    updateTrail(lane.mlpUpTrail, expandedGroup.position);
                                })
                                .onComplete(() => {
                                    // After the pause, start the down-projection pass-through
                                    setTimeout(() => {
                                        const orangeColor = new THREE.Color(0xb07c13);

                                        const downBottomY = mlpMatrixDown_centerY - MLP_MATRIX_PARAMS_DOWN.height / 2;
                                        const downTopY = mlpMatrixDown_centerY + MLP_MATRIX_PARAMS_DOWN.height / 2;

                                        // Move vector to bottom of matrix first if below it
                                        const startY = expandedGroup.position.y;
                                        const totalDist = downTopY - startY;
                                        const durationDown = (Math.abs(totalDist) / (ANIM_RISE_SPEED_INSIDE_LN * GLOBAL_ANIM_SPEED_MULT)) * 1000;

                                        // Matrix color tween
                                        new TWEEN.Tween({ t: 0 })
                                            .to({ t: 1 }, durationDown)
                                            .easing(TWEEN.Easing.Quadratic.InOut)
                                            .onUpdate(o => {
                                                const col = mlpDarkGray.clone().lerp(orangeColor, o.t);
                                                mlpMatrixDown.setColor(col);
                                                mlpMatrixDown.setEmissive(col, 0.5);
                                            })
                                            .start();

                                        // Move the expanded vector through the matrix
                                        new TWEEN.Tween(expandedGroup.position)
                                            .to({ y: downTopY }, durationDown)
                                            .easing(TWEEN.Easing.Linear.None)
                                            .onUpdate(() => {
                                                updateTrail(lane.mlpUpTrail, expandedGroup.position);
                                            })
                                            .onStart(() => {
                                                // Instantly shrink the widened 3072-dim vector so it fits within the narrowing matrix
                                                expandedGroup.scale.setScalar(0.25);
                                            })
                                            .onComplete(() => {
                                                mlpMatrixDown.setColor(orangeColor);
                                                mlpMatrixDown.setEmissive(orangeColor, 0.5);

                                                // --------------------------------------------------
                                                //  Collapse back to a single 768-dim vector
                                                // --------------------------------------------------
                                                const collapseVec = new VectorVisualizationInstancedPrism(segmentVecs[0].rawData.slice(), expandedGroup.position.clone());

                                                // Copy gradient colours
                                                if (Array.isArray(segmentVecs[0].currentKeyColors) && segmentVecs[0].currentKeyColors.length) {
                                                    collapseVec.currentKeyColors = segmentVecs[0].currentKeyColors.map(c => c.clone());
                                                    collapseVec.updateInstanceGeometryAndColors();
                                                }

                                                scene.add(collapseVec.group);

                                                // Hide expanded 3072-dim group
                                                expandedGroup.visible = false;

                                                // Update lane reference for future phases (if any)
                                                lane.finalVecAfterMlp = collapseVec;

                                                // Ensure trail keeps following
                                                updateTrail(lane.mlpUpTrail, collapseVec.group.position);

                                                // ------------------------------
                                                //  Rise a bit above the matrix
                                                // ------------------------------
                                                const riseAbove = 40; // units
                                                const riseDur = (riseAbove / (ANIM_RISE_SPEED_INSIDE_LN * GLOBAL_ANIM_SPEED_MULT)) * 1000;

                                                new TWEEN.Tween(collapseVec.group.position)
                                                    .to({ y: collapseVec.group.position.y + riseAbove }, riseDur)
                                                    .easing(TWEEN.Easing.Quadratic.InOut)
                                                    .onUpdate(() => {
                                                        updateTrail(lane.mlpUpTrail, collapseVec.group.position);
                                                    })
                                                    .onComplete(() => {
                                                        // Update the target height for the residual-stream vectors so they
                                                        // continue to rise in parallel with the post-MLP vectors.
                                                        if (mhsaAnimation) {
                                                            mhsaAnimation.finalOriginalY = collapseVec.group.position.y - ORIGINAL_TO_PROCESSED_GAP;
                                                        }

                                                        // ------------------------------
                                                        //  Move left to residual stream (x=0)
                                                        // ------------------------------
                                                        const horizDist = Math.abs(collapseVec.group.position.x);
                                                        const horizDur = (horizDist / (ANIM_HORIZ_SPEED * GLOBAL_ANIM_SPEED_MULT)) * 1000;

                                                        new TWEEN.Tween(collapseVec.group.position)
                                                            .to({ x: 0 }, horizDur)
                                                            .easing(TWEEN.Easing.Quadratic.InOut)
                                                            .onUpdate(() => {
                                                                updateTrail(lane.mlpUpTrail, collapseVec.group.position);
                                                            })
                                                            .onComplete(() => {
                                                                // Perform the residual addition once the processed vector
                                                                // has aligned with the main stream.
                                                                if (mhsaAnimation && lane.originalVec) {
                                                                    mhsaAnimation._startAdditionAnimation(lane.originalVec, collapseVec, lane);
                                                                }
                                                            })
                                                            .start();
                                                    })
                                                    .start();
                                            })
                                            .start();
                                    }, pauseMs);
                                })
                                .start();
                        })
                        .start();
                }
            }
        });

        // Call MHSAAnimation update method
        if (mhsaAnimation) {
            mhsaAnimation.update(deltaTime, timeNow, lanes);
        }

        // Check for MHSA pass-through readiness // Moved to MHSAAnimation
        /*
        if (mhaPassThroughPhase === 'positioning_mha_vectors') {
            if (areAllMHAVectorsInPosition()) {
                mhaPassThroughPhase = 'ready_for_parallel_pass_through';
                console.log("All MHSA vectors are in position. Ready for PARALLEL pass-through.");
                initiateParallelHeadPassThroughAnimations(lanes);
            }
        }
        */

        // Update tweens
        if (typeof TWEEN !== 'undefined' && TWEEN.update) TWEEN.update();

        controls.update();
        composer.render();
    }
    animate();

    // -------------------------------------------------------------------------
    //  Cleanup (dispose resources)
    // -------------------------------------------------------------------------
    return () => {
        document.removeEventListener('visibilitychange', visibilityHandler);
        controls.dispose();
        lanes.forEach(l => {
            l.originalVec.dispose();
            l.movingVec.dispose();
            l.multTarget.dispose();
            if (l.resultVec) l.resultVec.dispose();
            if (l.movingVecLN2) l.movingVecLN2.dispose && l.movingVecLN2.dispose();
            if (l.multTargetLN2) l.multTargetLN2.dispose && l.multTargetLN2.dispose();
            if (l.resultVecLN2) l.resultVecLN2.dispose && l.resultVecLN2.dispose();
        });
        layerNorm1.dispose && layerNorm1.dispose();
        layerNorm2.dispose && layerNorm2.dispose();

        // mhaVisualizations and mlpMatrix1/mlpMatrix2 contain WeightMatrixVisualization instances.
        // Their groups and meshes will be handled by the scene.traverse below.
        // If WeightMatrixVisualization had its own dispose, we'd call it here.
        // e.g., mhaVisualizations.forEach(vis => vis.dispose && vis.dispose());
        // e.g., if (mlpMatrix1.dispose) mlpMatrix1.dispose();
        // e.g., if (mlpMatrix2.dispose) mlpMatrix2.dispose();

        scene.traverse(obj => {
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) {
                if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
                else obj.material.dispose();
            }
        });
        composer.passes.forEach(p => { if (p.dispose) p.dispose(); });
        renderer.dispose();
    };
} 