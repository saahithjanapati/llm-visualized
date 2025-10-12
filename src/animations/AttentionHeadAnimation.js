import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GUI } from 'three/examples/jsm/libs/lil-gui.module.min.js';
import { WeightMatrixVisualization } from '../components/WeightMatrixVisualization.js';
import { VectorVisualization } from '../components/VectorVisualization.js';
import { VECTOR_LENGTH } from '../utils/constants.js';
import { scaleOpacityForDisplay } from '../utils/trailConstants.js';

// Maximum points per trail line
const MAX_TRAIL_POINTS = 1000;

// Simple animation for a single self-attention head consisting of separate
// query, key and value projection weight matrices.  A single vector rises up
// along the centre (key) matrix and stops slightly below the matrix ready for
// future interaction/attention visualisation.  The function returns a cleanup
// callback to properly dispose of Three.js resources when the animation is
// no longer needed.
export function initAttentionHeadAnimation(containerElement) {
    // ---------------------------------------------------------------------
    //  Scene / Camera / Renderer setup
    // ---------------------------------------------------------------------
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);

    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 30, 90);

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

    // ---------------------------------------------------------------------
    //  Create Q, K, V matrices (left-to-right) – order QKV; colours B,R,G
    // ---------------------------------------------------------------------
    let matrixPosY = matrixParams.height / 2; // bottom rests on y = 0 plane
    let spacingX = matrixParams.width;        // centres separated so bottoms touch

    const queryMatrix = new WeightMatrixVisualization(
        null,
        new THREE.Vector3(-spacingX, matrixPosY, 0),
        matrixParams.width,
        matrixParams.height,
        matrixParams.depth,
        matrixParams.topWidthFactor,
        matrixParams.cornerRadius,
        matrixParams.numberOfSlits,
        matrixParams.slitWidth,
        matrixParams.slitDepthFactor,
        matrixParams.slitBottomWidthFactor,
        matrixParams.slitTopWidthFactor
    );
    queryMatrix.setColor(new THREE.Color(0x0000ff)); // Blue
    scene.add(queryMatrix.group);

    const keyMatrix = new WeightMatrixVisualization(
        null,
        new THREE.Vector3(0, matrixPosY, 0),
        matrixParams.width,
        matrixParams.height,
        matrixParams.depth,
        matrixParams.topWidthFactor,
        matrixParams.cornerRadius,
        matrixParams.numberOfSlits,
        matrixParams.slitWidth,
        matrixParams.slitDepthFactor,
        matrixParams.slitBottomWidthFactor,
        matrixParams.slitTopWidthFactor
    );
    keyMatrix.setColor(new THREE.Color(0xff0000)); // Red
    scene.add(keyMatrix.group);

    const valueMatrix = new WeightMatrixVisualization(
        null,
        new THREE.Vector3(spacingX, matrixPosY, 0),
        matrixParams.width,
        matrixParams.height,
        matrixParams.depth,
        matrixParams.topWidthFactor,
        matrixParams.cornerRadius,
        matrixParams.numberOfSlits,
        matrixParams.slitWidth,
        matrixParams.slitDepthFactor,
        matrixParams.slitBottomWidthFactor,
        matrixParams.slitTopWidthFactor
    );
    valueMatrix.setColor(new THREE.Color(0x00ff00)); // Green
    scene.add(valueMatrix.group);

    const allMatrices = [queryMatrix, keyMatrix, valueMatrix];

    // ---------------------------------------------------------------------
    //  Create vectors & trails (5 lanes through slits in the KEY matrix)
    // ---------------------------------------------------------------------
    const allVectors = [];      // originals + duplicates
    const allTrailLines = [];

    const originalVectors = []; // keep track of originals separately

    const vectorHeightOffset = 40; // distance beneath matrices to spawn
    const startY = -vectorHeightOffset;            // y below ground
    const stopY = matrixPosY - matrixParams.height - 2; // just beneath key matrix bottom (y=0 - 2)

    // Compute slit spacing for current depth
    const updateVectorsPositions = () => {
        const slitSpacing = matrixParams.depth / (matrixParams.numberOfSlits + 1);
        allVectors.forEach((vecVis, i) => {
            const zPos = -matrixParams.depth / 2 + slitSpacing * (i + 1);
            vecVis.group.position.set(0, vecVis.group.position.y, zPos);
        });
    };

    const createTrailForVector = (initPos) => {
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(MAX_TRAIL_POINTS * 3);
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setDrawRange(0, 0);
        const material = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: scaleOpacityForDisplay(0.07) });
        const line = new THREE.Line(geometry, material);
        scene.add(line);
        // seed first point
        geometry.getAttribute('position').setXYZ(0, ...initPos);
        geometry.setDrawRange(0, 1);
        return { line, geometry, positions, points: [initPos], isFull: false };
    };

    // Utility to create a VectorVisualization plus trail and register arrays
    const addVectorWithTrail = (vecVis) => {
        scene.add(vecVis.group);
        allVectors.push(vecVis);

        const trail = createTrailForVector([vecVis.group.position.x, vecVis.group.position.y, vecVis.group.position.z]);
        allTrailLines.push(trail);
    };

    // Generate 5 original vectors & trails
    for (let i = 0; i < matrixParams.numberOfSlits; i++) {
        const data = Array.from({ length: VECTOR_LENGTH }, () => Math.random() * 2 - 1);
        const vecVis = new VectorVisualization(data);
        vecVis.data = data; // preserve raw values for later duplication
        vecVis.group.position.set(0, startY, 0); // z will be set below
        addVectorWithTrail(vecVis);
        originalVectors.push(vecVis);
    }
    updateVectorsPositions(); // place Z positions

    // ---------------------------------------------------------------------
    //  Geometry update helper (called from GUI)
    // ---------------------------------------------------------------------
    function updateMatricesGeometry() {
        // Recompute derived placement values
        spacingX = matrixParams.width;
        matrixPosY = matrixParams.height / 2;

        // Recreate geometry for each matrix
        allMatrices.forEach(matVis => {
            matVis.updateGeometry(matrixParams);
        });
        // Re-position matrices so bottoms touch
        queryMatrix.setPosition(-spacingX, matrixPosY, 0);
        keyMatrix.setPosition(0,          matrixPosY, 0);
        valueMatrix.setPosition(spacingX,  matrixPosY, 0);

        // Restore intended colours in case recreated materials reset
        queryMatrix.setColor(0x0000ff);
        keyMatrix.setColor(0xff0000);
        valueMatrix.setColor(0x00ff00);

        // Update vectors Z alignment
        updateVectorsPositions();
    }

    // ---------------------------------------------------------------------
    //  COPY-AND-ROUTE SEQUENCE USING TWEEN ---------------------------------
    // ---------------------------------------------------------------------
    const slideDur   = 1500; // ms
    const ascendDur  = 1500; // ms
    const processDur = 2000; // ms through matrix

    const stopBelowMatrix = -2; // Y target right beneath matrices

    let readyCount = 0; // when reaches (originals + duplicates) under matrices

    function onVectorReady() {
        readyCount++;
        if (readyCount === matrixParams.numberOfSlits * 3) {
            // All 15 vectors positioned; start processing
            startProcessingPhase();
        }
    }

    const createDuplicate = (sourceVec, xPos) => {
        const srcData = sourceVec.data ?? Array.from({ length: VECTOR_LENGTH }, () => Math.random() * 2 - 1);
        const dup = new VectorVisualization(srcData.slice());
        dup.data = srcData.slice();
        dup.group.position.set(xPos, sourceVec.group.position.y, sourceVec.group.position.z);
        addVectorWithTrail(dup);

        // Ascend tween
        new TWEEN.Tween(dup.group.position)
            .to({ y: stopBelowMatrix }, ascendDur)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onComplete(onVectorReady)
            .start();
        return dup;
    };

    // Build phase tweens for each original
    originalVectors.forEach((vec) => {
        // Slide left to Q position
        const slideLeft = new TWEEN.Tween(vec.group.position)
            .to({ x: -spacingX }, slideDur)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onComplete(() => {
                createDuplicate(vec, -spacingX); // duplicate to Q
            });

        // Slide centre
        const slideCentre = new TWEEN.Tween(vec.group.position)
            .to({ x: 0 }, slideDur)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onComplete(() => {
                createDuplicate(vec, 0); // duplicate to K
            });

        // Slide right to V
        const slideRight = new TWEEN.Tween(vec.group.position)
            .to({ x: spacingX }, slideDur)
            .easing(TWEEN.Easing.Quadratic.InOut);

        // Ascend originals after reaching V
        const ascendOriginal = new TWEEN.Tween(vec.group.position)
            .to({ y: stopBelowMatrix }, ascendDur)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onComplete(onVectorReady);

        slideLeft.chain(slideCentre);
        slideCentre.chain(slideRight);
        slideRight.chain(ascendOriginal);

        slideLeft.start();
    });

    // ---------------------------------------------------------------------
    //  POST-POSITION PROCESSING THROUGH MATRICES ---------------------------
    // ---------------------------------------------------------------------
    function startProcessingPhase() {
        allVectors.forEach(vecVis => {
            // flash to white
            vecVis.ellipses.forEach(e => {
                e.material.originalColor = e.material.color.clone();
                e.material.color.set(0xffffff);
                e.material.emissive.set(0xffffff);
                e.material.emissiveIntensity = 1.0;
            });

            // Animate upward through its matrix
            const targetY = matrixPosY + matrixParams.height + 5;
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

        // Update trails for all vectors (including duplicates)
        allVectors.forEach((vecVis, idx) => {
            // Get the trail object directly
            const trail = allTrailLines[idx];
            // Access points list via trail object
            const tList = trail.points;
            const geom = trail.geometry;
            const attr = geom.getAttribute('position');

            const newPoint = [vecVis.group.position.x, vecVis.group.position.y, vecVis.group.position.z];
            // Check if the trail is full before pushing
            if (!trail.isFull) {
                tList.push(newPoint);
                // Check if it just became full
                if (tList.length === MAX_TRAIL_POINTS) {
                    trail.isFull = true;
                }
            }

            const lastIdx = tList.length - 1;
            attr.setXYZ(lastIdx, ...newPoint);
            geom.setDrawRange(0, tList.length);
            attr.needsUpdate = true;
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
    };
}
