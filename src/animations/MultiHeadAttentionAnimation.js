import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GUI } from 'three/examples/jsm/libs/lil-gui.module.min.js';
import { WeightMatrixVisualization } from '../components/WeightMatrixVisualization.js';
import { VectorVisualization } from '../components/VectorVisualization.js';
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
    //  Parameters & GUI setup
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

    const gui = new GUI();
    const matrixFolder = gui.addFolder('Matrix');
    matrixFolder.add(matrixParams, 'width', 10, 60, 0.5).onChange(updateMatricesGeometry);
    matrixFolder.add(matrixParams, 'height', 4, 20, 0.5).onChange(updateMatricesGeometry);
    matrixFolder.add(matrixParams, 'depth', 10, 100, 1).onChange(updateMatricesGeometry);
    matrixFolder.add(matrixParams, 'topWidthFactor', 0.2, 1.2, 0.01).onChange(updateMatricesGeometry);
    matrixFolder.add(matrixParams, 'cornerRadius', 0, 5, 0.05).onChange(updateMatricesGeometry);
    matrixFolder.add(matrixParams, 'numberOfSlits', 0, 10, 1).onChange(updateMatricesGeometry);
    matrixFolder.add(matrixParams, 'slitWidth', 0.1, 3, 0.05).onChange(updateMatricesGeometry);
    matrixFolder.add(matrixParams, 'slitDepthFactor', 0, 1, 0.01).onChange(updateMatricesGeometry);
    matrixFolder.add(matrixParams, 'slitBottomWidthFactor', 0.1, 1, 0.01).name('Slit Bottom %').onChange(updateMatricesGeometry);
    matrixFolder.add(matrixParams, 'slitTopWidthFactor', 0.1, 1, 0.01).name('Slit Top %').onChange(updateMatricesGeometry);
    matrixFolder.open();

    // --- Moved Controls Target Setting Here ---
    const initialMatrixPosY = matrixParams.height / 2; // Now safe to access
    controls.target.set(0, initialMatrixPosY, 0);
    controls.update(); // Apply the target setting
    // ------------------------------------------

    // ---------------------------------------------------------------------
    //  Create Q, K, V matrices for ALL head sets
    // ---------------------------------------------------------------------
    // Stores all matrix visualization objects, structured as:
    // allMatrices[headSetIndex][0=Query, 1=Key, 2=Value]
    const allMatrices = []; // Will hold NUM_HEAD_SETS * 3 matrices

    // Calculate positions and create matrices
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

    createOrUpdateAllMatrices(); // Initial creation

    // ---------------------------------------------------------------------
    //  Create vectors & trails (targeting the FIRST head set only)
    // ---------------------------------------------------------------------
    // These arrays now ONLY store vectors involved in the animation (first head)
    const animatedVectors = [];
    const animatedTrailLines = [];
    const originalVectors = []; // keep track of originals separately for duplication logic

    const vectorHeightOffset = 40; // distance beneath matrices to spawn
    const startY = -vectorHeightOffset;            // y below ground

    // Get positions for the FIRST head set's matrices
    const firstHeadSet = allMatrices[0];
    const firstQPos = firstHeadSet[0].group.position;
    const firstKPos = firstHeadSet[1].group.position;
    const firstVPos = firstHeadSet[2].group.position;

    // Compute slit spacing based on the *first* head's Key matrix depth
    // Note: Assumes all matrices share the same depth initially
    const firstKeyMatrixDepth = matrixParams.depth; // Use initial param
    const slitSpacing = firstKeyMatrixDepth / (matrixParams.numberOfSlits + 1);

    // --- START ADDED FUNCTION ---
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
    // --- END ADDED FUNCTION ---

    // Utility to create a VectorVisualization plus trail and register in ANIMATED lists
    const addAnimatedVectorWithTrail = (vecVis) => {
        scene.add(vecVis.group);
        animatedVectors.push(vecVis);

        const trail = createTrailForVector(vecVis.group.position.toArray());
        animatedTrailLines.push(trail);
    };

    // Generate 5 original vectors & trails, positioned relative to the FIRST Key matrix
    for (let i = 0; i < matrixParams.numberOfSlits; i++) {
        const data = Array.from({ length: VECTOR_LENGTH }, () => Math.random() * 2 - 1);
        const vecVis = new VectorVisualization(data);
        vecVis.data = data; // preserve raw values for later duplication

        // Calculate Z position based on slit index
        const zPos = -firstKeyMatrixDepth / 2 + slitSpacing * (i + 1);
        // Initial position: below the center (X) of the first Key matrix
        vecVis.group.position.set(firstKPos.x, startY, zPos);

        addAnimatedVectorWithTrail(vecVis);
        originalVectors.push(vecVis); // Still track originals separately
    }
    // No need for updateVectorsPositions() here anymore

    // ---------------------------------------------------------------------
    //  Geometry update helper (called from GUI)
    // ---------------------------------------------------------------------
    function updateMatricesGeometry() {
        // Recreate geometry for ALL matrices across ALL sets
        allMatrices.flat().forEach(matVis => {
            if (matVis) {
                // Dispose old geometry/material before creating new
                if (matVis.mesh) {
                     if (matVis.mesh.geometry) matVis.mesh.geometry.dispose();
                     if (matVis.mesh.material) {
                         if (Array.isArray(matVis.mesh.material)) {
                             matVis.mesh.material.forEach(m => m.dispose());
                         } else {
                             matVis.mesh.material.dispose();
                         }
                     }
                }
                matVis.updateGeometry(matrixParams);
            }
        });

        // Recalculate positions and update scene graph objects
        // (Reusing the creation logic handles positioning based on new params)
        createOrUpdateAllMatrices();

        // --- IMPORTANT ---
        // Since vectors are positioned relative to the *first* head set,
        // we need to update their initial Z positions if depth changes.
        // The running animation state might become inconsistent if params change mid-flight.
        // For simplicity, we won't try to dynamically reposition animated vectors here.
        // A full reset might be needed for robust parameter updates during animation.

        // We also need to update the target positions used in tweens if width changes.
        // Re-fetch positions of the *newly created* first head matrices.
        // This assumes tweens haven't started or can adapt; may cause jumps.
        const updatedFirstHead = allMatrices[0];
        firstQPos.copy(updatedFirstHead[0].group.position);
        firstKPos.copy(updatedFirstHead[1].group.position);
        firstVPos.copy(updatedFirstHead[2].group.position);

        // Update controls target if matrix height changes
        controls.target.set(0, matrixParams.height / 2, 0);

        // Restore intended colours (createOrUpdateAllMatrices already does this)
    }

    // ---------------------------------------------------------------------
    //  COPY-AND-ROUTE SEQUENCE (TARGETING FIRST HEAD SET) ------------------
    // ---------------------------------------------------------------------
    const slideDur   = 1500; // ms
    const ascendDur  = 1500; // ms
    const processDur = 2000; // ms through matrix

    const stopBelowMatrix = -2; // Y target right beneath matrices (relative to Y=0 plane)
    const matrixCenterY = matrixParams.height / 2; // Reference Y for processing target

    let readyCount = 0; // when reaches (originals + duplicates for first head) under matrices

    function onVectorReady() {
        readyCount++;
        // Only start processing when all vectors for the *first head* are ready
        if (readyCount === matrixParams.numberOfSlits * 3) {
            startProcessingPhase();
        }
    }

    // Create a duplicate vector at a specific X position (for Q or K of the first head)
    const createDuplicate = (sourceVec, targetXPos) => {
        const srcData = sourceVec.data ?? Array.from({ length: VECTOR_LENGTH }, () => Math.random() * 2 - 1);
        const dup = new VectorVisualization(srcData.slice());
        dup.data = srcData.slice();
        // Position duplicate initially at the source's Z, but the target X, and current Y
        dup.group.position.set(targetXPos, sourceVec.group.position.y, sourceVec.group.position.z);
        addAnimatedVectorWithTrail(dup); // Add duplicate to animated list

        // Ascend tween for the duplicate
        new TWEEN.Tween(dup.group.position)
            .to({ y: stopBelowMatrix }, ascendDur)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onComplete(onVectorReady) // Count this duplicate when it's ready
            .start();
        return dup;
    };

    // Build phase tweens for each original vector, targeting the first head set's matrices
    originalVectors.forEach((vec) => {
        const currentPos = vec.group.position; // original vector's position tween target

        // Slide left to FIRST Q position
        const slideLeft = new TWEEN.Tween(currentPos)
            .to({ x: firstQPos.x }, slideDur)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onComplete(() => {
                // Create duplicate at Q position when original arrives
                createDuplicate(vec, firstQPos.x);
            });

        // Slide centre to FIRST K position
        const slideCentre = new TWEEN.Tween(currentPos)
            .to({ x: firstKPos.x }, slideDur)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onComplete(() => {
                 // Create duplicate at K position when original arrives
                createDuplicate(vec, firstKPos.x);
            });

        // Slide right to FIRST V position
        const slideRight = new TWEEN.Tween(currentPos)
            .to({ x: firstVPos.x }, slideDur)
            .easing(TWEEN.Easing.Quadratic.InOut);
            // No duplicate created here, the original continues

        // Ascend original after reaching V position
        const ascendOriginal = new TWEEN.Tween(currentPos)
            .to({ y: stopBelowMatrix }, ascendDur)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onComplete(onVectorReady); // Count original when ready at V

        // Chain the tweens for the original vector's movement
        slideLeft.chain(slideCentre);
        slideCentre.chain(slideRight);
        slideRight.chain(ascendOriginal);

        // Start the sequence for this original vector
        slideLeft.start();
    });

    // ---------------------------------------------------------------------
    //  POST-POSITION PROCESSING THROUGH MATRICES (FIRST HEAD SET ONLY) -----
    // ---------------------------------------------------------------------
    function startProcessingPhase() {
        // Only process vectors associated with the first head animation
        animatedVectors.forEach(vecVis => {
            // flash to white
            vecVis.ellipses.forEach(e => {
                e.material.originalColor = e.material.color.clone();
                e.material.color.set(0xffffff);
                e.material.emissive.set(0xffffff);
                e.material.emissiveIntensity = 1.0;
            });

            // Animate upward through its matrix (use shared height/position params)
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
                    // restore colours (smaller vector new mapping could apply)
                    vecVis.ellipses.forEach(e => {
                        const col = e.material.originalColor || e.material.color;
                        e.material.color.copy(col);
                        e.material.emissive.copy(col);
                        e.material.emissiveIntensity = 0.3;
                    });
                })
                .start();
        });
    }

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

        // Update trails ONLY for the animated vectors (first head set)
        animatedVectors.forEach((vecVis, idx) => {
            // Get the trail object directly using the index from the animated list
            const trail = animatedTrailLines[idx];
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
        animatedTrailLines.forEach(trail => {
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