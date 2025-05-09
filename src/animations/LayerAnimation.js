import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { LayerNormalizationVisualization } from '../components/LayerNormalizationVisualization.js';
import { VectorVisualizationInstancedPrism } from '../components/VectorVisualizationInstancedPrism.js';
import { WeightMatrixVisualization } from '../components/WeightMatrixVisualization.js';
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
    MAX_TRAIL_POINTS,
    ANIM_RISE_SPEED_HEAD,
    HEAD_VECTOR_STOP_BELOW,
    GLOBAL_ANIM_SPEED_MULT,
    SIDE_COPY_DELAY_MS,
    SIDE_COPY_HORIZ_SPEED,
    HIDE_INSTANCE_Y_OFFSET
} from '../utils/constants.js';
import { mapValueToColor } from '../utils/colors.js';

// NOTE: Requires global TWEEN.js (loaded separately via <script>)

// Define speed multiplier
const SPEED_MULT = GLOBAL_ANIM_SPEED_MULT;

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
    const mhsa_matrix_center_y = MHSA_BASE_Y + MHA_MATRIX_PARAMS.height / 2;

    const mhaVisualizations = [];
    const headsCentersX = [];
    const headCoords = [];

    const darkGrayColor = new THREE.Color(0x404040); // Define dark gray color once
    const matrixOpacity = 0.7; // Define desired opacity

    // MHSA Pass-Through Animation State - Reworked
    let mhaPassThroughPhase = 'positioning_mha_vectors';

    for (let i = 0; i < NUM_HEAD_SETS_LAYER; i++) {
        const headSetWidth = MHA_INTERNAL_MATRIX_SPACING * 2 + MHA_MATRIX_PARAMS.width;
        const currentHeadSetBaseX = BRANCH_X - MHA_INTERNAL_MATRIX_SPACING + i * (headSetWidth + HEAD_SET_GAP_LAYER);

        const x_q = currentHeadSetBaseX;
        const x_k = currentHeadSetBaseX + MHA_INTERNAL_MATRIX_SPACING;
        const x_v = currentHeadSetBaseX + MHA_INTERNAL_MATRIX_SPACING * 2;

        const queryMatrix = new WeightMatrixVisualization(
            null, new THREE.Vector3(x_q, mhsa_matrix_center_y, 0),
            MHA_MATRIX_PARAMS.width, MHA_MATRIX_PARAMS.height, MHA_MATRIX_PARAMS.depth,
            MHA_MATRIX_PARAMS.topWidthFactor, MHA_MATRIX_PARAMS.cornerRadius, MHA_MATRIX_PARAMS.numberOfSlits,
            MHA_MATRIX_PARAMS.slitWidth, MHA_MATRIX_PARAMS.slitDepthFactor,
            MHA_MATRIX_PARAMS.slitBottomWidthFactor, MHA_MATRIX_PARAMS.slitTopWidthFactor
        );
        queryMatrix.setColor(darkGrayColor);
        queryMatrix.group.children.forEach(child => {
            if (child.material) {
                child.material.transparent = true;
                child.material.opacity = matrixOpacity;
            }
        });
        scene.add(queryMatrix.group);
        mhaVisualizations.push(queryMatrix);

        const keyMatrix = new WeightMatrixVisualization(
            null, new THREE.Vector3(x_k, mhsa_matrix_center_y, 0),
            MHA_MATRIX_PARAMS.width, MHA_MATRIX_PARAMS.height, MHA_MATRIX_PARAMS.depth,
            MHA_MATRIX_PARAMS.topWidthFactor, MHA_MATRIX_PARAMS.cornerRadius, MHA_MATRIX_PARAMS.numberOfSlits,
            MHA_MATRIX_PARAMS.slitWidth, MHA_MATRIX_PARAMS.slitDepthFactor,
            MHA_MATRIX_PARAMS.slitBottomWidthFactor, MHA_MATRIX_PARAMS.slitTopWidthFactor
        );
        keyMatrix.setColor(darkGrayColor);
        keyMatrix.group.children.forEach(child => {
            if (child.material) {
                child.material.transparent = true;
                child.material.opacity = matrixOpacity;
            }
        });
        scene.add(keyMatrix.group);
        mhaVisualizations.push(keyMatrix);

        const valueMatrix = new WeightMatrixVisualization(
            null, new THREE.Vector3(x_v, mhsa_matrix_center_y, 0),
            MHA_MATRIX_PARAMS.width, MHA_MATRIX_PARAMS.height, MHA_MATRIX_PARAMS.depth,
            MHA_MATRIX_PARAMS.topWidthFactor, MHA_MATRIX_PARAMS.cornerRadius, MHA_MATRIX_PARAMS.numberOfSlits,
            MHA_MATRIX_PARAMS.slitWidth, MHA_MATRIX_PARAMS.slitDepthFactor,
            MHA_MATRIX_PARAMS.slitBottomWidthFactor, MHA_MATRIX_PARAMS.slitTopWidthFactor
        );
        valueMatrix.setColor(darkGrayColor);
        valueMatrix.group.children.forEach(child => {
            if (child.material) {
                child.material.transparent = true;
                child.material.opacity = matrixOpacity;
            }
        });
        scene.add(valueMatrix.group);
        mhaVisualizations.push(valueMatrix);

        headsCentersX.push(x_k); // Use K matrix centre as canonical head centre
        headCoords.push({ q: x_q, k: x_k, v: x_v });
    }

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

    // --- Trail line support --------------------------------------------------------
    // const MAX_TRAIL_POINTS = 1500; // Now from constants
    function createTrailLine(color) {
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(MAX_TRAIL_POINTS * 3);
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setDrawRange(0, 0);
        const material = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.12 });
        const line = new THREE.Line(geometry, material);
        // Ensure the trail line is always rendered (disable frustum culling)
        line.frustumCulled = false;
        // Initialize bounding sphere so frustum culling calculations include entire trail
        geometry.computeBoundingSphere();
        scene.add(line);
        return { line, geometry, positions, points: [] };
    }

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

        // Create trails
        const origTrail = createTrailLine(0xffffff);
        const branchTrail = createTrailLine(0xffffff);

        lanes.push({
            zPos,
            originalVec: origVec,
            movingVec,
            multTarget,
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
            // New MHSA traversal properties
            travellingVec: null,
            upwardCopies: [],
            headIndex: 0,
            finalAscend: false,
            sideCopies: [],
            sideTrails: [],
            upwardTrails: []
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

        const flashDuration = 150 / SPEED_MULT; // Adjusted by SPEED_MULT
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
    //  Helper: addition animation (same as pipeline)
    // -------------------------------------------------------------------------
    function startAdditionAnimation(vec1, vec2, onComplete) {
        console.warn("startAdditionAnimation needs refactoring for InstancedPrism.");
        if (onComplete) onComplete();
        /*
        const ADD_DURATION = 500;
        const ADD_FLASH = 120;
        const ADD_DELAY_BETWEEN = 50;
        const duration = ADD_DURATION;
        const flashDuration = ADD_FLASH;
        const delayBetweenCubes = ADD_DELAY_BETWEEN;
        const vectorLength = vec1.ellipses.length;
        let completed = 0;

        for (let i = 0; i < vectorLength; i++) {
            const ellipse1 = vec1.ellipses[i];
            const ellipse2 = vec2.ellipses[i];
            if (!ellipse1 || !ellipse2) continue;

            const targetPosition = new THREE.Vector3();
            ellipse2.getWorldPosition(targetPosition);
            const localTargetPosition = ellipse1.parent.worldToLocal(targetPosition.clone());

            new TWEEN.Tween(ellipse1.position)
                .to({ y: localTargetPosition.y }, duration)
                .easing(TWEEN.Easing.Quadratic.InOut)
                .delay(i * delayBetweenCubes)
                .onComplete(() => {
                    const originalColor = ellipse2.material.color.clone();
                    const originalEmissive = ellipse2.material.emissive.clone();
                    const originalIntensity = ellipse2.material.emissiveIntensity;

                    ellipse2.material.color.set(0xffffff);
                    ellipse2.material.emissive.set(0xffffff);
                    ellipse2.material.emissiveIntensity = 1.5;

                    new TWEEN.Tween(ellipse2.material)
                        .to({}, flashDuration)
                        .onComplete(() => {
                            const sum = vec1.data[i] + vec2.data[i];
                            vec2.data[i] = sum;
                            const newColor = mapValueToColor(sum);
                            ellipse2.material.color.copy(newColor);
                            ellipse2.material.emissive.copy(newColor);
                            ellipse2.material.emissiveIntensity = originalIntensity;
                            ellipse1.visible = false;
                            completed++;
                            if (completed === vectorLength && onComplete) onComplete();
                        })
                        .start();
                })
                .start();
        }
        */
    }

    // helper to push position into trail
    function updateTrail(trailObj, pos) {
        // Ensure pos is valid and is a Vector3-like object
        if (!pos || typeof pos.x !== 'number' || typeof pos.y !== 'number' || typeof pos.z !== 'number') {
            // console.warn('updateTrail received invalid pos:', pos);
            return; // Invalid position, do nothing
        }

        const pts = trailObj.points;
        let needsToPushNewPoint = false;

        if (pts.length === 0) {
            needsToPushNewPoint = true;
        } else {
            const lastPt = pts[pts.length - 1];
            // Check if last point is a valid array [x,y,z]
            if (Array.isArray(lastPt) && lastPt.length === 3 &&
                typeof lastPt[0] === 'number' && typeof lastPt[1] === 'number' && typeof lastPt[2] === 'number') {
                if (pos.x !== lastPt[0] || pos.y !== lastPt[1] || pos.z !== lastPt[2]) {
                    needsToPushNewPoint = true;
                }
            } else {
                // Last point is corrupted, treat as if point changed to re-sync / clear and push.
                // For now, just mark that we need to push the current valid 'pos'.
                // console.warn("Corrupted last point in trail. Forcing update with current pos.", lastPt);
                needsToPushNewPoint = true;
                // Optional: clear corrupted trail: pts.length = 0;
            }
        }

        if (needsToPushNewPoint) {
            if (pts.length < MAX_TRAIL_POINTS) {
                const lastPos = pts.length > 0 ? pts[pts.length - 1] : null;
                
                // For high speeds (large jumps), add intermediate points
                if (lastPos && SPEED_MULT > 10) {
                    // Calculate the distance moved
                    const dx = pos.x - lastPos[0];
                    const dy = pos.y - lastPos[1];
                    const dz = pos.z - lastPos[2];
                    const distance = Math.sqrt(dx*dx + dy*dy + dz*dz);
                    
                    // If distance is large, add intermediate points
                    if (distance > 5) {
                        // Calculate number of points to add based on distance
                        const steps = Math.min(10, Math.ceil(distance / 5));
                        
                        for (let i = 1; i < steps; i++) {
                            const t = i / steps;
                            const ix = lastPos[0] + dx * t;
                            const iy = lastPos[1] + dy * t;
                            const iz = lastPos[2] + dz * t;
                            
                            pts.push([ix, iy, iz]);
                            const interpolatedIdx = pts.length - 1;
                            trailObj.geometry.attributes.position.setXYZ(interpolatedIdx, ix, iy, iz);
                        }
                    }
                }
                
                // Add the actual position
                pts.push([pos.x, pos.y, pos.z]);
                const idx = pts.length - 1;
                trailObj.geometry.attributes.position.setXYZ(idx, pos.x, pos.y, pos.z);
                trailObj.geometry.setDrawRange(0, pts.length);
                trailObj.geometry.attributes.position.needsUpdate = true;
                if (idx === 0 || idx % 100 === 0) trailObj.geometry.computeBoundingSphere();
            }
        }
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
    const clock = new THREE.Clock();
    const headStopY = mhsa_matrix_center_y - HEAD_VECTOR_STOP_BELOW; // Define once
    const mhaPassThroughTargetY = mhsa_matrix_center_y + MHA_MATRIX_PARAMS.height / 2 + 20; // Target Y for vectors passing through (+20 overshoot)
    const mhaPassThroughDuration = 2000 / SPEED_MULT; // Duration for pass-through - INCREASED
    const outputVectorLength = 64; // Target length for output vectors
    const mhaResultRiseOffsetY = 50; // New: Offset for final rise
    const mhaResultRiseDuration = 500 / SPEED_MULT; // New: Duration for final rise

    // Colors for matrix flash & final state
    const brightGreen = new THREE.Color(0x33FF33);
    const darkTintedGreen = new THREE.Color(0x002200);
    const brightBlue = new THREE.Color(0x6666FF);
    const darkTintedBlue = new THREE.Color(0x000022);
    const brightRed = new THREE.Color(0xFF3333);
    const darkTintedRed = new THREE.Color(0x220000);

    // Helper function to check if all MHSA vectors are in their start positions for pass-through
    function areAllMHAVectorsInPosition() {
        if (!lanes.length) return false; // No lanes, nothing to check

        for (const lane of lanes) {
            if (!lane.upwardCopies || lane.upwardCopies.length !== NUM_HEAD_SETS_LAYER) {
                // console.log(`Lane ${lanes.indexOf(lane)}: upwardCopies not fully populated or missing.`);
                return false; // Not all K-vectors (upwardCopies) are created yet
            }

            for (let headIdx = 0; headIdx < NUM_HEAD_SETS_LAYER; headIdx++) {
                const kVec = lane.upwardCopies[headIdx];
                if (!kVec || Math.abs(kVec.group.position.y - headStopY) > 0.1) {
                    // console.log(`Lane ${lanes.indexOf(lane)}, Head ${headIdx} (K): K-vector not at headStopY.`);
                    return false; // K-vector not in position
                }
                if (!kVec.userData.sideSpawned) {
                    // console.log(`Lane ${lanes.indexOf(lane)}, Head ${headIdx} (K): Side copies not spawned yet.`);
                    return false; // Q/V side copies not spawned yet for this K-vector
                }
            }

            // Check side copies for this lane
            // Each K-vector (upwardCopy) should have spawned 2 side copies (Q & V)
            // Total side copies per lane = NUM_HEAD_SETS_LAYER * 2
            if (!lane.sideCopies || lane.sideCopies.length !== NUM_HEAD_SETS_LAYER * 2) {
                 // console.log(`Lane ${lanes.indexOf(lane)}: Incorrect number of sideCopies. Expected ${NUM_HEAD_SETS_LAYER * 2}, got ${lane.sideCopies.length}`);
                return false;
            }

            for (const sideCopyObj of lane.sideCopies) {
                if (!sideCopyObj || !sideCopyObj.vec) {
                    // console.log(`Lane ${lanes.indexOf(lane)}: Corrupt sideCopyObj.`);
                    return false;
                }
                if (Math.abs(sideCopyObj.vec.group.position.y - headStopY) > 0.1) {
                    // console.log(`Lane ${lanes.indexOf(lane)}, SideCopy (Q/V) TargetX ${sideCopyObj.targetX}: Not at headStopY.`);
                    return false; // Q or V vector not at headStopY
                }
                if (Math.abs(sideCopyObj.vec.group.position.x - sideCopyObj.targetX) > 0.1) {
                    // console.log(`Lane ${lanes.indexOf(lane)}, SideCopy (Q/V) TargetX ${sideCopyObj.targetX}: Not at targetX.`);
                    return false; // Q or V vector not at targetX
                }
            }
        }
        return true; // All vectors in all lanes are correctly positioned
    }

    // Helper function to animate a single vector and its matrix
    function animateVectorMatrixPassThrough(vector, matrix, brightMatrixColor, darkTintedMatrixColor, finalVectorHue, passThroughY, duration, riseOffset, riseDurationVal, outLength, animationCompletionCallback) {
        if (!vector || !matrix) {
            console.warn("Missing vector or matrix for pass-through animation.");
            animationCompletionCallback(); // Decrement count as if it completed
            return;
        }

        const originalMatrixEmissive = matrix.mesh.material.emissive.clone();
        const originalMatrixIntensity = matrix.mesh.material.emissiveIntensity;
        const tweenState = { y: vector.group.position.y, progress: 0, colorR: 1, colorG: 1, colorB: 1, matrixEmissiveIntensity: originalMatrixIntensity };
        const initialVecColor = new THREE.Color();
        if(vector.mesh.instanceColor) { vector.mesh.getColorAt(0, initialVecColor); } else { initialVecColor.setRGB(0.5,0.5,0.5); }
        tweenState.colorR = initialVecColor.r; tweenState.colorG = initialVecColor.g; tweenState.colorB = initialVecColor.b;

        new TWEEN.Tween(tweenState)
            .to({ y: passThroughY, progress: 1.0, colorR: 1.0, colorG: 1.0, colorB: 1.0, matrixEmissiveIntensity: 1.5 }, duration)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onUpdate(() => {
                vector.group.position.y = tweenState.y;
                const numCentralUnits = outLength;
                const startVisibleIndex = Math.floor((VECTOR_LENGTH_PRISM - numCentralUnits) / 2);
                const endVisibleIndex = startVisibleIndex + numCentralUnits - 1;
                const currentWhite = new THREE.Color(tweenState.colorR, tweenState.colorG, tweenState.colorB);
                for (let i = 0; i < VECTOR_LENGTH_PRISM; i++) {
                    let targetScaleY = vector.getUniformHeight();
                    let instanceYOffset = 0;
                    if (i < startVisibleIndex || i > endVisibleIndex) {
                        targetScaleY = THREE.MathUtils.lerp(vector.getUniformHeight(), 0.001, tweenState.progress);
                        if (targetScaleY < 0.01 && tweenState.progress > 0.5) { instanceYOffset = HIDE_INSTANCE_Y_OFFSET - vector.group.position.y; }
                    }
                    vector.setInstanceAppearance(i, instanceYOffset, currentWhite, new THREE.Vector3(vector.getWidthScale(), targetScaleY, vector.getDepthScale()));
                }
                matrix.setColor(brightMatrixColor); matrix.setEmissive(brightMatrixColor, tweenState.matrixEmissiveIntensity); matrix.setOpacity(1.0);
            })
            .onComplete(() => {
                const processedData = vector.rawData.slice(0, outLength);
                vector.applyProcessedVisuals(processedData, outLength, { numKeyColors: 3, generationOptions: { type: 'monochromatic', baseHue: finalVectorHue, saturation: 0.9, minLightness: 0.4, maxLightness: 0.8 }});
                vector.group.position.set(vector.group.position.x, passThroughY, vector.group.position.z);
                matrix.setColor(darkTintedMatrixColor); matrix.setEmissive(darkTintedMatrixColor, 0.1); matrix.setOpacity(1.0);

                new TWEEN.Tween(vector.group.position)
                    .to({ y: passThroughY + riseOffset }, riseDurationVal)
                    .easing(TWEEN.Easing.Cubic.Out)
                    .onComplete(animationCompletionCallback) // Call the main completion callback
                    .start();
            })
            .start();
    }

    // Main function to initiate parallel pass-through for all heads
    function initiateParallelHeadPassThroughAnimations(allLanes) {
        if (mhaPassThroughPhase !== 'ready_for_parallel_pass_through') return;
        console.log("Initiating Parallel MHSA Head Pass-Through Animations...");
        mhaPassThroughPhase = 'parallel_pass_through_active';

        let totalAnimationsToComplete = allLanes.length * NUM_HEAD_SETS_LAYER * 3; // K, Q, V for each head in each lane
        let animationsCompleted = 0;

        function singleAnimationDone() {
            animationsCompleted++;
            if (animationsCompleted === totalAnimationsToComplete) {
                console.log("All MHSA parallel pass-through animations complete.");
                mhaPassThroughPhase = 'mha_pass_through_complete';
            }
        }

        allLanes.forEach((lane) => {
            for (let headIdx = 0; headIdx < NUM_HEAD_SETS_LAYER; headIdx++) {
                const kVec = lane.upwardCopies[headIdx];
                const kMatrix = mhaVisualizations[headIdx * 3 + 1];
                animateVectorMatrixPassThrough(kVec, kMatrix, brightGreen, darkTintedGreen, 0.333, mhaPassThroughTargetY, mhaPassThroughDuration, mhaResultRiseOffsetY, mhaResultRiseDuration, outputVectorLength, singleAnimationDone);

                const qSideCopy = lane.sideCopies.find(sc => sc.headIndex === headIdx && sc.type === 'Q');
                if (qSideCopy && qSideCopy.vec) {
                    animateVectorMatrixPassThrough(qSideCopy.vec, qSideCopy.matrixRef, brightBlue, darkTintedBlue, 0.666, mhaPassThroughTargetY, mhaPassThroughDuration, mhaResultRiseOffsetY, mhaResultRiseDuration, outputVectorLength, singleAnimationDone);
                } else { totalAnimationsToComplete--; } // Adjust if Q-vec missing

                const vSideCopy = lane.sideCopies.find(sc => sc.headIndex === headIdx && sc.type === 'V');
                if (vSideCopy && vSideCopy.vec) {
                    animateVectorMatrixPassThrough(vSideCopy.vec, vSideCopy.matrixRef, brightRed, darkTintedRed, 0.0, mhaPassThroughTargetY, mhaPassThroughDuration, mhaResultRiseOffsetY, mhaResultRiseDuration, outputVectorLength, singleAnimationDone);
                } else { totalAnimationsToComplete--; } // Adjust if V-vec missing
            }
        });

        if (totalAnimationsToComplete === 0 && allLanes.length > 0) { // Handle case where no animations were started
             console.log("No valid K,Q,V vectors found to animate for parallel pass-through.");
             mhaPassThroughPhase = 'mha_pass_through_complete'; // Still mark as complete
        }
    }

    function animate() {
        requestAnimationFrame(animate);
        const deltaTime = clock.getDelta();
        const timeNow = performance.now();

        // --- LayerNorm Appearance Control ---
        const darkGray = new THREE.Color(0x333333);
        const lightYellow = new THREE.Color(0xFFFF99); // For semi-transparent state
        const brightYellow = new THREE.Color(0xFFFF00); // For final opaque state
        const opaqueOpacity = 1.0;
        const semiTransparentOpacity = 0.6;
        const exitTransitionRange = 10; // Y-distance over which the exit transition occurs

        const firstMovingVecY = lanes.length > 0 ? lanes[0].movingVec.group.position.y : startY;
        const bottomY_ln1_abs = LAYER_NORM_1_Y_POS - LN_PARAMS.height / 2;
        const midY_ln1_abs = LAYER_NORM_1_Y_POS;
        const topY_ln1_abs = LAYER_NORM_1_Y_POS + LN_PARAMS.height / 2;

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

        layerNorm1.group.children.forEach(child => {
            if (child instanceof THREE.Mesh && child.material) {
                // Set transparent flag based on final opacity value
                child.material.transparent = targetOpacity < 1.0;
                child.material.color.copy(targetColor);
                child.material.opacity = targetOpacity;
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
                originalVec.group.position.y += ANIM_RISE_SPEED_ORIGINAL * SPEED_MULT * deltaTime;
                if (originalVec.group.position.y > originalTargetY) originalVec.group.position.y = originalTargetY;
            }

            // Update original trail: track center ellipse during merge, else use group position
            if (lane.mergeStarted) { // This mergeStarted will now only be for future merge operations, not LN1
                const centerIndex = Math.floor(VECTOR_LENGTH / 2);
                // const centerEllipse = originalVec.ellipses[centerIndex]; // No ellipses
                // const worldPos = new THREE.Vector3();
                // centerEllipse.getWorldPosition(worldPos);
                // updateTrail(lane.origTrail, worldPos);
                // TODO: Refactor for InstancedPrism - If a specific point on the prism is needed, calculate it.
                // For now, using group position.
                updateTrail(lane.origTrail, originalVec.group.position);
            } else {
                updateTrail(lane.origTrail, originalVec.group.position);
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
                    const dx = ANIM_HORIZ_SPEED * SPEED_MULT * deltaTime;
                    movingVec.group.position.x = Math.min(BRANCH_X, movingVec.group.position.x + dx);
                    if (movingVec.group.position.x >= BRANCH_X) {
                        movingVec.group.position.x = BRANCH_X;
                        lane.horizPhase = 'insideLN';
                    }
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
                        // movingVec.startAnimation(); // VectorVisualizationInstancedPrism does not have startAnimation
                        // TODO: Refactor for InstancedPrism - Trigger normalization effect if needed.
                        // This might involve a custom animation loop that calls setInstanceAppearance on movingVec
                        // or a method within VectorVisualizationInstancedPrism to animate its normalization.
                        // For now, we assume normalization is reflected by updating its data.
                        movingVec.updateDataInternal(movingVec.rawData.slice()); // This re-calculates normalizedData
                        lane.normStarted = true;
                    }

                    // Update normalization visuals
                    // movingVec.update(timeNow); // VectorVisualizationInstancedPrism does not have update()
                    // TODO: Refactor for InstancedPrism - If there's an ongoing animation for normalization, update it here.

                    // Move up (only when not actively normalizing)
                    // const normAnimating = lane.normStarted && movingVec.animationState.isAnimating; // animationState doesn't exist
                    const normAnimating = false; // Placeholder, as direct animation state isn't available
                    if (!lane.multStarted && !normAnimating) {
                        movingVec.group.position.y += ANIM_RISE_SPEED_INSIDE_LN * SPEED_MULT * deltaTime;
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
                                const finalY = branchFinalY;
                                const distance = finalY - resultVec.group.position.y;
                                const riseDuration = (distance / (ANIM_RISE_SPEED_INSIDE_LN * SPEED_MULT)) * 1000;

                                new TWEEN.Tween(resultVec.group.position)
                                    .to({ y: finalY }, riseDuration)
                                    .easing(TWEEN.Easing.Linear.None)
                                    .onComplete(() => {
                                        // Vector has passed LN1 and is at branchX, finalY
                                        lane.horizPhase = 'travelMHSA'; // Start moving through MHSA heads
                                        lane.travellingVec = resultVec;
                                        lane.headIndex = 0;
                                        lane.upwardCopies = [];
                                    })
                                    .start();
                            }
                        });
                    }
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
                case 'travelMHSA': {
                    const tVec = lane.travellingVec;
                    if (!tVec) break;
                    const targetHeadIdx = lane.headIndex || 0;
                    const targetX = headsCentersX[Math.min(targetHeadIdx, headsCentersX.length - 1)];
                    const dx = ANIM_HORIZ_SPEED * SPEED_MULT * deltaTime;
                    if (tVec.group.position.x < targetX - 0.01) {
                        tVec.group.position.x = Math.min(targetX, tVec.group.position.x + dx);
                    } else {
                        // Arrived at (or passed) the head centre — duplicate upward copy for every head
                        const dupeData = [...tVec.rawData];
                        const upVec = new VectorVisualizationInstancedPrism(dupeData, tVec.group.position.clone());
                        scene.add(upVec.group);
                        upVec.userData = { headIndex: targetHeadIdx, sideSpawned: false };
                        lane.upwardCopies.push(upVec);
                        // Create trail for upward movement
                        const upTrail = createTrailLine(0xffffff); // White color for all trails
                        updateTrail(upTrail, upVec.group.position);
                        lane.upwardTrails = lane.upwardTrails || [];
                        lane.upwardTrails.push(upTrail);

                        // After duplicating at last head, hide travelling vector and mark finished
                        if (targetHeadIdx === NUM_HEAD_SETS_LAYER - 1) {
                            tVec.group.visible = false;
                            lane.horizPhase = 'finishedHeads';
                        }
                        lane.headIndex = targetHeadIdx + 1;
                        // If finished traversing all heads, stop horizontal motion
                        if (lane.headIndex >= NUM_HEAD_SETS_LAYER) {
                            lane.horizPhase = 'finishedHeads';
                        }
                    }
                    break;
                }
                case 'finishedHeads': {
                    // No-op for now; vectors already positioned under heads
                    break;
                }
                case 'merged': // This case will no longer be triggered by the LN1 output
                default:
                    break;
            }

            // --- Upward movement for copies under heads ---
            const headStopY = mhsa_matrix_center_y - HEAD_VECTOR_STOP_BELOW;
            if (lane.upwardCopies && lane.upwardCopies.length) {
                lane.upwardCopies.forEach((upVec, idx) => {
                    if (upVec.group.position.y < headStopY) {
                        upVec.group.position.y = Math.min(headStopY, upVec.group.position.y + ANIM_RISE_SPEED_HEAD * SPEED_MULT * deltaTime);
                        // Update vertical trail
                        if (lane.upwardTrails && lane.upwardTrails[idx]) {
                            updateTrail(lane.upwardTrails[idx], upVec.group.position);
                        }
                    }
                });
            }
            if (lane.finalAscend && lane.travellingVec) {
                const tVec = lane.travellingVec;
                if (tVec.group.position.y < headStopY) {
                    tVec.group.position.y = Math.min(headStopY, tVec.group.position.y + ANIM_RISE_SPEED_HEAD * SPEED_MULT * deltaTime);
                }
            }

            // Spawn side copies once centre vector settled (with delay)
            if (lane.upwardCopies) {
                lane.upwardCopies.forEach(centerVec => {
                    // If first time under centre, schedule spawn
                    if (!centerVec.userData.sideSpawnRequested && Math.abs(centerVec.group.position.y - headStopY) < 0.1) {
                        // schedule side copy spawn
                        centerVec.userData.sideSpawnRequested = true;
                        centerVec.userData.sideSpawnTime = timeNow + SIDE_COPY_DELAY_MS / SPEED_MULT; // Respect SPEED_MULT
                        return; // wait for delay
                    }
                    // After delay has elapsed, spawn copies
                    if (centerVec.userData.sideSpawnRequested && !centerVec.userData.sideSpawned && timeNow >= centerVec.userData.sideSpawnTime) {
                        const hIdx = centerVec.userData.headIndex;
                        const coord = headCoords[hIdx];
                        if (coord) {
                            const qMatrixForHead = mhaVisualizations[hIdx * 3];
                            const vMatrixForHead = mhaVisualizations[hIdx * 3 + 2];

                            const qVec = new VectorVisualizationInstancedPrism(centerVec.rawData.slice(), centerVec.group.position.clone());
                            const vVec = new VectorVisualizationInstancedPrism(centerVec.rawData.slice(), centerVec.group.position.clone());
                            scene.add(qVec.group);
                            scene.add(vVec.group);
                            
                            // Add type and matrixRef to sideCopy objects
                            lane.sideCopies.push({ vec: qVec, targetX: coord.q, type: 'Q', matrixRef: qMatrixForHead, headIndex: hIdx });
                            lane.sideCopies.push({ vec: vVec, targetX: coord.v, type: 'V', matrixRef: vMatrixForHead, headIndex: hIdx });
                            
                            // white trails
                            lane.sideTrails.push(createTrailLine(0xffffff));
                            lane.sideTrails.push(createTrailLine(0xffffff));
                            // initialize trail
                            updateTrail(lane.sideTrails[lane.sideTrails.length-2], qVec.group.position);
                            updateTrail(lane.sideTrails[lane.sideTrails.length-1], vVec.group.position);
                            centerVec.userData.sideSpawned = true;
                        }
                    }
                });
            }

            // Move side copies horizontally and vertically align
            if (mhaPassThroughPhase === 'positioning_mha_vectors' && lane.sideCopies && lane.sideCopies.length) {
                lane.sideCopies.forEach((obj, idx) => {
                    const v = obj.vec;
                    const dx = SIDE_COPY_HORIZ_SPEED * SPEED_MULT * deltaTime;
                    if (Math.abs(v.group.position.x - obj.targetX) > 0.01) {
                        const dir = v.group.position.x < obj.targetX ? 1 : -1;
                        v.group.position.x += dir * dx;
                        if ((dir === 1 && v.group.position.x > obj.targetX) || (dir === -1 && v.group.position.x < obj.targetX))
                            v.group.position.x = obj.targetX;
                    }
                    // Ensure vertical position stays at headStopY
                    v.group.position.y = headStopY;
                    // Update trail
                    if (lane.sideTrails[idx]) updateTrail(lane.sideTrails[idx], v.group.position);
                });
            }

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
            } else if (lane.resultVec && lane.resultVec.group.visible) {
                branchPos = lane.resultVec.group.position;
            } else if (lane.movingVec.group.visible) {
                branchPos = lane.movingVec.group.position;
            }
            if (branchPos) {
                updateTrail(lane.branchTrail, branchPos);
            }
        });

        // Check for MHSA pass-through readiness
        if (mhaPassThroughPhase === 'positioning_mha_vectors') {
            if (areAllMHAVectorsInPosition()) {
                mhaPassThroughPhase = 'ready_for_parallel_pass_through';
                console.log("All MHSA vectors are in position. Ready for PARALLEL pass-through.");
                initiateParallelHeadPassThroughAnimations(lanes);
            }
        }

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
        controls.dispose();
        lanes.forEach(l => {
            l.originalVec.dispose();
            l.movingVec.dispose();
            l.multTarget.dispose();
            if (l.resultVec) l.resultVec.dispose();
        });
        layerNorm1.dispose && layerNorm1.dispose();

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