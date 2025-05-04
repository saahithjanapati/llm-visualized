import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GUI } from 'three/examples/jsm/libs/lil-gui.module.min.js';
import { WeightMatrixVisualizationInstance as WeightMatrixVisualization } from '../components/WeightMatrixVisualizationInstance.js';
import { VectorVisualizationInstanced } from '../components/VectorVisualizationInstanced.js';
import { VECTOR_LENGTH } from '../utils/constants.js';

// Maximum points per trail line
const MAX_TRAIL_POINTS = 1000;

// --- START NEW CONSTANTS ---
// Number of attention head sets (each set has Q, K, V)
const NUM_HEAD_SETS = 12;
// Horizontal gap between adjacent head sets
const HEAD_SET_GAP = 15;
// --- END NEW CONSTANTS ---

// Simple animation for MULTIPLE self-attention heads.
// TODO: Adapt logic for multiple heads beyond the first set.
// This currently just duplicates the single head animation.
// The function returns a cleanup callback to properly dispose of Three.js resources.
export function initMultiHeadAttentionAnimation(containerElement) {
    // ---------------------------------------------------------------------
    //  Scene / Camera / Renderer setup
    // ---------------------------------------------------------------------
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);

    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 2000);
    camera.position.set(0, 60, 450);

    // Accept either a <canvas> element or a generic container node like <body>
    let renderer;
    if (containerElement instanceof HTMLCanvasElement) {
        renderer = new THREE.WebGLRenderer({ canvas: containerElement, antialias: true });
    } else {
        renderer = new THREE.WebGLRenderer({ antialias: true });
        containerElement.appendChild(renderer.domElement);
    }
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);

    // ---------------------------------------------------------------------
    //  Controls & Lights
    // ---------------------------------------------------------------------
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(10, 30, 10);
    scene.add(dirLight);

    // ---------------------------------------------------------------------
    //  Parameters (defined before functions that use them)
    // ---------------------------------------------------------------------
    const matrixParams = {
        width: 37.5,
        height: 12,
        depth: 50,
        topWidthFactor: 0.47,
        cornerRadius: 1.2,
        numberOfSlits: 5,
        slitWidth: 1.85,
        slitDepthFactor: 1.0,
        slitBottomWidthFactor: 0.95,
        slitTopWidthFactor: 0.37
    };

    // ---------------------------------------------------------------------
    //  Helper Functions (defined after params, before use)
    // ---------------------------------------------------------------------

    // Stores all matrix visualization objects, structured as:
    // allMatrices[headSetIndex][0=Query, 1=Key, 2=Value]
    const allMatrices = []; // Will hold NUM_HEAD_SETS * 3 matrices

    // --- START ADDED DATA PROPERTY HELPER ---
    // Helper function to add the raw data property needed by createDuplicate
    function createVectorInstancedWithData(data, initialPosition) {
        const vecVis = new VectorVisualizationInstanced(data, initialPosition);
        vecVis.data = data; // Add the raw data property for duplication logic
        return vecVis;
    }
    // --- END ADDED DATA PROPERTY HELPER ---

    // Function to create or update all Q, K, V matrices for all head sets
    function createOrUpdateAllMatrices() {
        // Clear existing matrices from the scene if updating
        allMatrices.flat().forEach(matVis => {
            if (matVis && matVis.group) {
                scene.remove(matVis.group);
                // Dispose geometry/material later in updateMatricesGeometry or cleanup
            }
        });
        allMatrices.length = 0; // Clear the array

        let matrixPosY = matrixParams.height / 2; // bottom rests on y = 0 plane
        let matrixWidth = matrixParams.width;     // Width of a single Q/K/V matrix
        let matrixSpacing = matrixWidth;        // spacing between Q/K/V centres within a set

        // Calculate the starting X position for the center of the first set's Q matrix
        // to center the entire arrangement horizontally
        const singleSetWidth = 3 * matrixWidth;
        const totalWidth = NUM_HEAD_SETS * singleSetWidth + (NUM_HEAD_SETS - 1) * HEAD_SET_GAP;
        const firstSet_Q_Center_X = -totalWidth / 2 + matrixWidth / 2;


        for (let i = 0; i < NUM_HEAD_SETS; i++) {
            const headSetMatrices = [];

            // Calculate center X positions for Q, K, V of the current set
            const setOffset = i * (singleSetWidth + HEAD_SET_GAP);
            const x_q = firstSet_Q_Center_X + setOffset;
            const x_k = x_q + matrixSpacing;
            const x_v = x_k + matrixSpacing;

            // Create Query Matrix (Blue)
            const queryMatrix = new WeightMatrixVisualization(
                null, new THREE.Vector3(x_q, matrixPosY, 0),
                matrixParams.width, matrixParams.height, matrixParams.depth,
                matrixParams.topWidthFactor, matrixParams.cornerRadius, matrixParams.numberOfSlits,
                matrixParams.slitWidth, matrixParams.slitDepthFactor,
                matrixParams.slitBottomWidthFactor, matrixParams.slitTopWidthFactor
            );
            queryMatrix.setColor(new THREE.Color(0x0000ff));
            scene.add(queryMatrix.group);
            headSetMatrices.push(queryMatrix);

            // Create Key Matrix (Green)
            const keyMatrix = new WeightMatrixVisualization(
                null, new THREE.Vector3(x_k, matrixPosY, 0),
                matrixParams.width, matrixParams.height, matrixParams.depth,
                matrixParams.topWidthFactor, matrixParams.cornerRadius, matrixParams.numberOfSlits,
                matrixParams.slitWidth, matrixParams.slitDepthFactor,
                matrixParams.slitBottomWidthFactor, matrixParams.slitTopWidthFactor
            );
            keyMatrix.setColor(new THREE.Color(0x00ff00));
            scene.add(keyMatrix.group);
            headSetMatrices.push(keyMatrix);

            // Create Value Matrix (Red)
            const valueMatrix = new WeightMatrixVisualization(
                null, new THREE.Vector3(x_v, matrixPosY, 0),
                matrixParams.width, matrixParams.height, matrixParams.depth,
                matrixParams.topWidthFactor, matrixParams.cornerRadius, matrixParams.numberOfSlits,
                matrixParams.slitWidth, matrixParams.slitDepthFactor,
                matrixParams.slitBottomWidthFactor, matrixParams.slitTopWidthFactor
            );
            valueMatrix.setColor(new THREE.Color(0xff0000));
            scene.add(valueMatrix.group);
            headSetMatrices.push(valueMatrix);

            allMatrices.push(headSetMatrices);
        }
    }

    // Geometry update helper (called from GUI)
    function updateMatricesGeometry() {
        // When using the shared-geometry implementation disposing the geometry
        // would invalidate **all** matrices, so we simply forward the new
        // parameters to each instance.  If the params differ from the cached
        // set the first call will trigger a re-generation which every *new* instance
        // afterwards will then re-use automatically.
        allMatrices.flat().forEach(matVis => {
            if (matVis) {
                matVis.updateGeometry(matrixParams);
            }
        });

        // Recalculate positions and update scene graph objects (size may change)
        createOrUpdateAllMatrices();

        // Update controls target if matrix height changes
        controls.target.set(0, matrixParams.height / 2, 0);
        // Vector positions & tweens left untouched for simplicity.
    }

    createOrUpdateAllMatrices(); // Create matrices using the function defined above

    // ---------------------------------------------------------------------
    //  Vector & Trail Helpers (defined BEFORE vector initialization)
    // ---------------------------------------------------------------------

    // --- START MOVED FUNCTIONS ---
    const createTrailForVector = (initPos) => {
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(MAX_TRAIL_POINTS * 3);
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setDrawRange(0, 0);
        const material = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.07 });
        const line = new THREE.Line(geometry, material);
        scene.add(line);
        // seed first point
        if (initPos && initPos.length === 3) {
             geometry.getAttribute('position').setXYZ(0, ...initPos);
             geometry.setDrawRange(0, 1);
        }
        return { line, geometry, positions, points: initPos ? [initPos] : [], isFull: false };
    };

    // Utility to create a VectorVisualization plus trail and register for a specific head
    const addAnimatedVectorWithTrail = (vecVis, headIndex) => {
        scene.add(vecVis.group);
        animatedVectorsByHead[headIndex].push(vecVis);

        const trail = createTrailForVector(vecVis.group.position.toArray());
        animatedTrailLinesByHead[headIndex].push(trail);
    };
    // --- END MOVED FUNCTIONS ---

    // ---------------------------------------------------------------------
    //  Vector & Trail Initialization (uses helpers defined above)
    // ---------------------------------------------------------------------

    // Store vectors and trails per head set
    // originalVectorsByHead[headIndex] = [vecVis1, vecVis2, ...]
    // animatedVectorsByHead[headIndex] = [vecVis1, vecVis2, ...]
    // animatedTrailLinesByHead[headIndex] = [trail1, trail2, ...]
    const originalVectorsByHead = Array.from({ length: NUM_HEAD_SETS }, () => []);
    const animatedVectorsByHead = Array.from({ length: NUM_HEAD_SETS }, () => []);
    const animatedTrailLinesByHead = Array.from({ length: NUM_HEAD_SETS }, () => []);

    const vectorHeightOffset = 40; // distance beneath matrices to spawn
    const startY = -vectorHeightOffset;            // y below ground

    // Generate original vectors & trails for EACH head set
    console.log('[MultiHeadAttention] Generating initial vectors...');
    for (let h = 0; h < NUM_HEAD_SETS; h++) {
        const headSetMatrices = allMatrices[h];
        if (!headSetMatrices || headSetMatrices.length < 2) {
            console.error(`[MultiHeadAttention] Error: Matrix data missing for head set ${h}. Cannot create vectors.`);
            continue; // Skip this head set if matrices aren't ready
        }
        const keyMatrixPos = headSetMatrices[1].group.position; // Key matrix for this head
        const keyMatrixDepth = matrixParams.depth; // Assuming depth is same for all for now
        const slitSpacing = keyMatrixDepth / (matrixParams.numberOfSlits + 1); // Recalculate slit spacing here

        for (let i = 0; i < matrixParams.numberOfSlits; i++) {
            const data = Array.from({ length: VECTOR_LENGTH }, () => Math.random() * 2 - 1);
            // Use the new instanced version and the helper function
            const vecVis = createVectorInstancedWithData(data);
            // vecVis.data is now set by the helper
            const zPos = -keyMatrixDepth / 2 + slitSpacing * (i + 1);
            vecVis.group.position.set(keyMatrixPos.x, startY, zPos);
            addAnimatedVectorWithTrail(vecVis, h);
            originalVectorsByHead[h].push(vecVis);
        }
        console.log(`[MultiHeadAttention]   Added ${originalVectorsByHead[h].length} vectors for head ${h}`);
    }

    // ---------------------------------------------------------------------
    //  ANIMATION PARAMETERS (Defined BEFORE tweens that use them)
    // ---------------------------------------------------------------------
    const slideDur = 1500; // ms
    const ascendDur = 1500; // ms
    const processDur = 2000; // ms through matrix
    const stopBelowMatrix = -2; // Y target right beneath matrices (relative to Y=0 plane)

    // Define ready counter and processing phase before they're used
    let readyCount = 0;
    const totalVectorsToReady = NUM_HEAD_SETS * matrixParams.numberOfSlits * 3;
    console.log(`[MultiHeadAttention] Expecting ${totalVectorsToReady} vectors to become ready.`);

    function onVectorReady() {
        readyCount++;
        console.log(`[MultiHeadAttention] Vector ready! Count: ${readyCount}/${totalVectorsToReady}`);
        if (readyCount === totalVectorsToReady) {
            console.log('[MultiHeadAttention] All vectors ready. Starting processing phase.');
            startProcessingPhase();
        }
    }

    function startProcessingPhase() {
        console.log('[MultiHeadAttention] Executing startProcessingPhase...');
        const matrixCenterY = matrixParams.height / 2; // Calculate center Y here
        animatedVectorsByHead.flat().forEach(vecVis => {
            // flash to white - REMOVED as InstancedMesh doesn't support easy per-instance material changes this way
            // vecVis.ellipses.forEach(e => {
            //     e.material.originalColor = e.material.color.clone();
            //     e.material.color.set(0xffffff);
            //     e.material.emissive.set(0xffffff);
            //     e.material.emissiveIntensity = 1.0;
            // });


            // Animate upward through its matrix
            const targetY = matrixCenterY + matrixParams.height + 5; // Move above the matrix
            const scaler  = { t: 0 };
            new TWEEN.Tween({ y: vecVis.group.position.y })
                .to({ y: targetY }, processDur)
                .easing(TWEEN.Easing.Quadratic.InOut)
                .onUpdate(function (obj) {
                    vecVis.group.position.y = obj.y;
                })
                .start();

            // simultaneous shrink
            new TWEEN.Tween(scaler)
                .to({ t: 1 }, processDur)
                .easing(TWEEN.Easing.Quadratic.InOut)
                .onUpdate(() => {
                    const s = THREE.MathUtils.lerp(1, 0.3, scaler.t);
                    vecVis.group.scale.set(s, 1, 1);
                })
                .onComplete(() => {
                    // restore colours - REMOVED (no flash effect to restore from)
                    // vecVis.ellipses.forEach(e => {
                    //     const col = e.material.originalColor || e.material.color;
                    //     e.material.color.copy(col);
                    //     e.material.emissive.copy(col);
                    //     e.material.emissiveIntensity = 0.3;
                    // });
                })
                .start();
        });
    }

    // Create a duplicate vector at a specific X position for a specific head
    const createDuplicate = (sourceVec, headIndex, targetXPos) => {
        console.log(`[MultiHeadAttention] Creating duplicate for head ${headIndex} at x=${targetXPos.toFixed(2)}`);
        // Access data using the stored .data property
        const srcData = sourceVec.data ?? Array.from({ length: VECTOR_LENGTH }, () => Math.random() * 2 - 1);
        // Use the new instanced version via the helper
        const dup = createVectorInstancedWithData(srcData.slice());
        // dup.data is now set by the helper

        // Position duplicate at the source's Z, target X, and current Y
        dup.group.position.set(targetXPos, sourceVec.group.position.y, sourceVec.group.position.z);
        addAnimatedVectorWithTrail(dup, headIndex);

        // Ascend tween for the duplicate
        new TWEEN.Tween(dup.group.position)
            .to({ y: stopBelowMatrix }, ascendDur)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onComplete(() => {
                console.log(`[MultiHeadAttention] Duplicate at x=${targetXPos.toFixed(2)} reached bottom of matrix`);
                onVectorReady(); // Count this duplicate when it's ready
            })
            .start();
        return dup;
    };

    // --- SIMPLIFIED TWEEN SEQUENCE --- 
    console.log('[MultiHeadAttention] Starting vector animation sequence...');
    console.log('[MultiHeadAttention] Structure of originalVectorsByHead:', originalVectorsByHead);

    originalVectorsByHead.forEach((headOriginals, headIndex) => {
        const headSetMatrices = allMatrices[headIndex];
        const qPos = headSetMatrices[0].group.position;
        const kPos = headSetMatrices[1].group.position;
        const vPos = headSetMatrices[2].group.position;
        
        console.log(`[MultiHeadAttention] ---> Processing Head Index: ${headIndex}`);
        if (!headOriginals || headOriginals.length === 0) {
            console.warn(`[MultiHeadAttention] !!! No original vectors found for head ${headIndex}. Skipping this head.`);
            return; // Continue to the next head set
        }
        console.log(`[MultiHeadAttention]     Found ${headOriginals.length} original vectors for head ${headIndex}.`);
        console.log(`[MultiHeadAttention]     Target Positions: Q(${qPos.x.toFixed(2)}), K(${kPos.x.toFixed(2)}), V(${vPos.x.toFixed(2)})`);

        headOriginals.forEach((vec, vecIndex) => {
            const pos = vec.group.position;
            console.log(`[MultiHeadAttention]        -> Animating Vector ${vecIndex} starting at (${pos.x.toFixed(2)}, ${pos.y.toFixed(2)}, ${pos.z.toFixed(2)})`);
            
            // Simple linear Q->K->V->up sequence with manual update checks
            let phase = 1; // 1=to Q, 2=to K, 3=to V, 4=ascend
            
            function updateTween() {
                const now = Date.now();
                
                if (phase === 1) { // TO Q
                    pos.x += (qPos.x - pos.x) * 0.05;
                    if (Math.abs(pos.x - qPos.x) < 0.5) { // Increased threshold slightly
                        console.log(`[MultiHeadAttention]        Vector ${vecIndex} reached Q at ${pos.x.toFixed(2)}, creating duplicate`);
                        createDuplicate(vec, headIndex, qPos.x);
                        phase = 2;
                    }
                } 
                else if (phase === 2) { // TO K
                    pos.x += (kPos.x - pos.x) * 0.05;
                    if (Math.abs(pos.x - kPos.x) < 0.5) {
                        console.log(`[MultiHeadAttention]        Vector ${vecIndex} reached K at ${pos.x.toFixed(2)}, creating duplicate`);
                        createDuplicate(vec, headIndex, kPos.x);
                        phase = 3;
                    }
                }
                else if (phase === 3) { // TO V
                    pos.x += (vPos.x - pos.x) * 0.05;
                    if (Math.abs(pos.x - vPos.x) < 0.5) {
                        console.log(`[MultiHeadAttention]        Vector ${vecIndex} reached V at ${pos.x.toFixed(2)}, starting ascend`);
                        phase = 4;
                    }
                }
                else if (phase === 4) { // ASCEND
                    pos.y += (stopBelowMatrix - pos.y) * 0.05;
                    if (Math.abs(pos.y - stopBelowMatrix) < 0.5) {
                        console.log(`[MultiHeadAttention]        Vector ${vecIndex} completed ascend at ${pos.y.toFixed(2)}`);
                        phase = 5;
                        onVectorReady();
                        return; // Stop updating
                    }
                }
                
                // Continue updating until phase 5
                if (phase < 5) {
                    requestAnimationFrame(updateTween);
                }
            }
            
            // Start after a slight delay staggered by head index
            setTimeout(() => {
                requestAnimationFrame(updateTween); // Start the manual update loop
            }, headIndex * 100 + vecIndex * 50);
        });
    });
    // --- END SIMPLIFIED TWEEN SEQUENCE ---
    
    // ---------------------------------------------------------------------
    //  POST-POSITION PROCESSING & READY CHECK -----------------------------
    // ---------------------------------------------------------------------

    // ---------------------------------------------------------------------
    //  Resize handling
    // ---------------------------------------------------------------------
    function onWindowResize() {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    }
    window.addEventListener('resize', onWindowResize);

    // ---------------------------------------------------------------------
    //  Render / update loop
    // ---------------------------------------------------------------------
    const clock = new THREE.Clock();

    function animate() {
        requestAnimationFrame(animate);

        const delta = clock.getDelta();

        // Update trails for ALL animated vectors across ALL head sets
        animatedVectorsByHead.forEach((headVectors, headIndex) => {
            headVectors.forEach((vecVis, vecIndex) => {
                // Get the corresponding trail object
                const trail = animatedTrailLinesByHead[headIndex][vecIndex];
                if (!trail) return; // Safety check

                const tList = trail.points;
                const geom = trail.geometry;
                const attr = geom.getAttribute('position');

                const newPoint = [vecVis.group.position.x, vecVis.group.position.y, vecVis.group.position.z];
                if (!trail.isFull) {
                    tList.push(newPoint);
                    if (tList.length >= MAX_TRAIL_POINTS) { // Use >= for safety
                        trail.isFull = true;
                        // Optional: Implement logic to shift points if trail becomes full
                        // For now, it just stops adding new points implicitly
                    }
                } else {
                     // If full, shift existing points and add the new one at the end
                     // This creates a fixed-length moving trail
                     for (let i = 0; i < MAX_TRAIL_POINTS - 1; i++) {
                         attr.setXYZ(i, attr.getX(i + 1), attr.getY(i + 1), attr.getZ(i + 1));
                         tList[i] = tList[i+1]; // Update internal points array too
                     }
                     const lastIdx = MAX_TRAIL_POINTS - 1;
                     attr.setXYZ(lastIdx, ...newPoint);
                     tList[lastIdx] = newPoint;
                     geom.setDrawRange(0, MAX_TRAIL_POINTS); // Ensure draw range is full
                     attr.needsUpdate = true;
                     return; // Skip the normal update below if we handled the 'full' case
                }


                // Update for non-full or newly added point
                const lastIdx = tList.length - 1;
                if (lastIdx >= 0) { // Ensure there's at least one point
                     attr.setXYZ(lastIdx, ...newPoint);
                     geom.setDrawRange(0, tList.length);
                     attr.needsUpdate = true;
                }
            });
        });

        TWEEN.update();

        controls.update();
        renderer.render(scene, camera);
    }
    animate();

    // ---------------------------------------------------------------------
    //  Cleanup callback
    // ---------------------------------------------------------------------
    return () => {
        window.removeEventListener('resize', onWindowResize);
        controls.dispose();
        gui.destroy();
        renderer.dispose();

        scene.traverse(o => {
            if (o.geometry) o.geometry.dispose();
            if (o.material) {
                if (Array.isArray(o.material)) o.material.forEach(m => m.dispose());
                else o.material.dispose();
            }
        });
        // Ensure vector trails are also disposed
        animatedTrailLinesByHead.flat().forEach(trail => {
             if (trail.geometry) trail.geometry.dispose();
             if (trail.line && trail.line.material) trail.line.material.dispose();
             if (trail.line) scene.remove(trail.line);
        });
        // Ensure all matrix visualizations resources are released (handled by scene.traverse)
        // allMatrices.flat().forEach(matVis => {
        //     // Geometry/material disposed by scene.traverse if they are children
        //     // Additional cleanup specific to WeightMatrixVisualization if needed
        // });
    };
} 