import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GUI } from 'three/examples/jsm/libs/lil-gui.module.min.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'; // Import EffectComposer
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';     // Import RenderPass
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'; // Import UnrealBloomPass
import { VectorVisualizationInstanced } from '../components/VectorVisualizationInstanced'; // Updated import
import { scaleOpacityForDisplay } from '../utils/trailConstants.js';
import { WeightMatrixVisualization } from '../components/WeightMatrixVisualization';
import { VECTOR_LENGTH, HIDE_INSTANCE_Y_OFFSET } from '../utils/constants.js'; // Import VECTOR_LENGTH and HIDE_INSTANCE_Y_OFFSET
import { mapValueToColor } from '../utils/colors.js'; // For vector addition color updates
import TWEEN from '@tweenjs/tween.js'; // Tweening library for addition animation

// Toggle to true to see verbose logs; false for production performance
const DEBUG = false;

// Maximum points per trail line (adjust for performance/length)
const MAX_TRAIL_POINTS = 2000; // Further increased buffer size

export function initMainScene(canvas) { // Renamed function here
    // --- Basic Three.js setup ---
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111); // Dark background

    const starField = createRotatingStarField();
    scene.add(starField);

    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 3000);
    camera.position.set(0, 10, 25); // Position camera to view both objects

    const renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    // --- Post Processing Setup ---
    const composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));

    const bloomParams = {
        strength: 1.5, // Initial strength
        radius: 0.4,   // Initial radius
        threshold: 0.85 // Initial threshold
    };
    const bloomPass = new UnrealBloomPass(
        new THREE.Vector2(window.innerWidth, window.innerHeight),
        bloomParams.strength,
        bloomParams.radius,
        bloomParams.threshold
    );
    composer.addPass(bloomPass);


    // --- Lighting ---
    // Hemisphere Light (Sky/Ground) -
    const hemisphereLight = new THREE.HemisphereLight(0xffffff, 0x888888, 1.0);
    hemisphereLight.position.set(0, 50, 0); // Position doesn't affect diffuse lighting, but can be useful for helpers
    scene.add(hemisphereLight);

    // --- Controls ---
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxDistance = 2500; // prevent zooming beyond far plane

    // --- Keyboard Controls State ---
    const keysPressed = {};
    const moveSpeed = 0.5; // Original speed, can be reused or adjusted
    const panSpeed = 0.2; // Speed for panning
    const zoomSpeed = 0.5; // Speed for zooming
    const rotateSpeed = 0.02; // Adjust rotation speed as needed

    // --- Visualizations ---
    const allVectorVisualizations = []; // Array to hold all vector instances
    const allTrailLines = []; // Array to hold { line, geometry, material } for trails
    const allTrailPoints = []; // Array of arrays to hold points for each trail line [[x,y,z], ...]
    const vectorHeightOffset = 60; // Start even lower so vectors appear further beneath the matrix

    // --- Branching / Duplicate Vector Setup ---
    // Arrays to hold duplicate vectors and their trails
    const branchedVectorVisualizations = [];
    const branchedTrailLines = [];
    const branchedTrailPoints = [];
    const additionPlayedFlags = [];
    const originalTrailFrozenFlags = []; // Flags to stop updating original trails once merged
    // Smooth trail extension state tracking
    const trailExtendActive = [];
    const trailExtendStartTimes = [];
    const trailExtendStartYs = [];
    const trailExtendTargetYs = [];
    // Calculate total duration of addition animation for trail sync
    const additionDuration = 400; // ms from startAdditionAnimation
    const additionFlashDuration = 80; // ms from startAdditionAnimation
    const additionDelayBetweenCubes = 15; // ms from startAdditionAnimation
    const additionVectorLength = VECTOR_LENGTH; // Ensure this matches data length
    const totalAdditionDuration = additionDuration + additionFlashDuration + (additionVectorLength - 1) * additionDelayBetweenCubes;
    const trailExtendDuration = totalAdditionDuration; // sync trail extension with addition animation

    // Configuration for the duplicate vector path (all durations are expressed as fraction of the full loop 0..1)
    const branchConfig = {
        branchStartT: 0.15,   // When (0..1) the branch occurs (still below the matrix)
        horzDurationT: 0.20,  // Fraction of the loop spent moving right
        vertDurationT: 0.30,  // Fraction spent moving up
        leftDurationT: 0.25,  // Fraction spent moving left to meet originals
        branchX: 90,          // How far to move to the right (in world units) – move further to avoid the matrix
        meetYOffset: 5        // Final Y offset above the original vector path when they meet
    };

    // --- New simple two‑matrix merge configuration (branching temporarily disabled) ---
    const mergeConfig = {
        rightX: branchConfig.branchX, // X position of right‑hand matrix and its vectors
        vertDurationT: 0.7,           // Portion of the timeline the vectors move vertically
        moveLeftDurationT: 0.3        // Portion of timeline they slide left to merge
    };

    // Material Definitions (Shared by Matrix & Vectors)
    const availableMaterials = {
        MeshBasicMaterial: THREE.MeshBasicMaterial,
        MeshLambertMaterial: THREE.MeshLambertMaterial,
        MeshPhongMaterial: THREE.MeshPhongMaterial,
        MeshStandardMaterial: THREE.MeshStandardMaterial,
        MeshNormalMaterial: THREE.MeshNormalMaterial,
        MeshDepthMaterial: THREE.MeshDepthMaterial,
        MeshToonMaterial: THREE.MeshToonMaterial,
    };
    const defaultMatrixMaterialName = 'MeshStandardMaterial';
    const defaultVectorMaterialName = 'MeshStandardMaterial'; // Default for vectors

    // Configure matrix with user specified defaults
    const matrixParams = {
        width: 57.5,
        height: 8, // Increased thickness
        depth: 72,
        topWidthFactor: 0.76,
        cornerRadius: 2, // Default corner radius
        numberOfSlits: 5,
        slitWidth: 2.0, // Increased slit width
        slitDepthFactor: 1.0,
        slitWidthFactor: 0.6,
        opacity: 1.0, // Added opacity parameter
        material: defaultMatrixMaterialName,
        color: 0xaaaaaa, // Default color
    };
    // --- Weight Matrices ---
    // Original matrix in the centre (for the original vectors)
    const matrixVis = new WeightMatrixVisualization(
        null,
        new THREE.Vector3(0, 0, 0),
        matrixParams.width,
        matrixParams.height,
        matrixParams.depth,
        matrixParams.topWidthFactor,
        matrixParams.cornerRadius,
        matrixParams.numberOfSlits,
        matrixParams.slitWidth,
        matrixParams.slitDepthFactor,
        matrixParams.slitWidthFactor
    );

    // Additional matrix that sits along the branched path so the duplicate vectors
    // also pass through a weight matrix.  It shares the *same* parameters so that
    // GUI controls affect both identically.  We simply offset it in X so that the
    // two matrices do not intersect.
    const branchedMatrixVis = new WeightMatrixVisualization(
        null,
        new THREE.Vector3(branchConfig.branchX, 0, 0), // position it directly on the branched path
        matrixParams.width,
        matrixParams.height,
        matrixParams.depth,
        matrixParams.topWidthFactor,
        matrixParams.cornerRadius,
        matrixParams.numberOfSlits,
        matrixParams.slitWidth,
        matrixParams.slitDepthFactor,
        matrixParams.slitWidthFactor
    );

    // Keep both matrices in a single collection for easy iteration later
    const allMatrixVisualizations = [matrixVis, branchedMatrixVis];

    // Function to update matrix material
    function updateMatrixMaterial() {
        // Iterate over BOTH matrices so GUI changes affect them equally
        allMatrixVisualizations.forEach(matVis => {
            if (!matVis.group) return;

            // Find the first mesh to get its old material (for disposal)
            let oldMaterial = null;
            matVis.group.children.forEach(child => {
                if (child instanceof THREE.Mesh && child.material && !oldMaterial) {
                    oldMaterial = child.material;
                }
            });

            const MaterialConstructor = availableMaterials[matrixParams.material];
            if (!MaterialConstructor) {
                console.error('Invalid material selected:', matrixParams.material);
                return;
            }

            const newMaterial = new MaterialConstructor({
                color: matrixParams.color,
                transparent: true, // Keep transparent for opacity control
                opacity: matrixParams.opacity,
            });

            // Apply the new material to every mesh child of the current matrix
            matVis.group.children.forEach(child => {
                if (child instanceof THREE.Mesh) {
                    child.material = newMaterial;
                    child.material.needsUpdate = true;
                }
            });

            if (oldMaterial && oldMaterial !== newMaterial && typeof oldMaterial.dispose === 'function') {
                oldMaterial.dispose();
            }
        });
    }

    // Utility to rebuild BOTH matrices when a geometry-changing GUI control
    // changes.  After we recreate their geometries we call updateMatrixMaterial
    // to restore the currently selected material type, colour and opacity.
    const updateBothMatrixGeometry = () => {
        matrixVis.updateGeometry(matrixParams);
        branchedMatrixVis.updateGeometry(matrixParams);

        // Ensure appearance stays in sync
        updateMatrixMaterial();
    };

    // Initial material setup (applies to both matrices)
    updateMatrixMaterial();

    // Add both matrices to the scene
    scene.add(matrixVis.group);
    scene.add(branchedMatrixVis.group);

    // Define colors for animation (kept for potential future use)
    const brightYellow = new THREE.Color(0xFFFF00);
    const darkYellow = new THREE.Color(0xCCA000);
    const matrixCurrentColor = new THREE.Color(); // Color object for the central matrix
    const matrixCurrentColor2 = new THREE.Color(); // Color object for the branched‑path matrix
    const baseMatrixColor = new THREE.Color(); // To store the GUI base color

    // --- Vector Control Parameters ---
    const vectorControlParams = {
        opacity: 1.0,
        metalness: 0.3, // Default from VectorVisualization
        roughness: 0.5, // Default from VectorVisualization
        emissiveIntensity: 0.3, // Default from VectorVisualization
        material: defaultVectorMaterialName, // Add material selection for vectors
        // Add a base color for instanced materials, as vertex colors will tint this
        baseColor: 0xffffff
    };

    // Function to update all vector materials
    function updateAllVectorMaterials(property, value) {
        [...allVectorVisualizations, ...branchedVectorVisualizations].forEach(vectorVis => {
            if (!vectorVis.mesh || !vectorVis.mesh.material) return;

            const material = vectorVis.mesh.material; // Get the shared material

            if (property === 'material') {
                // Change material type
                const MaterialConstructor = availableMaterials[value];
                if (!MaterialConstructor) {
                    console.error('Invalid vector material selected:', value);
                    return;
                }

                const oldMaterial = material;
                const newMaterial = new MaterialConstructor({
                    vertexColors: true, // Crucial for instanced rendering
                    color: new THREE.Color(vectorControlParams.baseColor), // Use base color
                    transparent: vectorControlParams.opacity < 1.0,
                    opacity: vectorControlParams.opacity,
                    metalness: vectorControlParams.metalness,
                    roughness: vectorControlParams.roughness,
                    emissive: new THREE.Color(vectorControlParams.baseColor), // Base emissive color
                    emissiveIntensity: vectorControlParams.emissiveIntensity,
                });
                vectorVis.mesh.material = newMaterial; // Assign new material to the instanced mesh
                if (oldMaterial && typeof oldMaterial.dispose === 'function') {
                    oldMaterial.dispose();
                }
            } else {
                // Update specific property on existing material
                if (property === 'opacity') {
                    material.opacity = value;
                    material.transparent = value < 1.0;
                } else if (property === 'metalness') {
                    material.metalness = value;
                } else if (property === 'roughness') {
                    material.roughness = value;
                } else if (property === 'emissiveIntensity') {
                    material.emissiveIntensity = value;
                } else if (property === 'baseColor') { // For instanced mesh base color
                    material.color.set(value);
                    material.emissive.set(value); // Keep emissive tint aligned with base color
                }
            }
            material.needsUpdate = true;
        });
    }

    // --- Vector Addition Animation (adapted for VectorVisualizationInstanced) ---
    function startAdditionAnimation(vec1, vec2) { // vec1 is source (branched), vec2 is target (original)
        const duration = 400;
        const flashDuration = 80;
        const delayBetweenCubes = 15;
        const vectorLength = VECTOR_LENGTH; // vec1.normalizedData.length;

        // Mark addition animation playing
        additionPlaying.active = true;
        additionPlaying.endTime = getVirtualNow() + duration + flashDuration + vectorLength * delayBetweenCubes;

        if (DEBUG) console.log('[startAdditionAnimation] active tweens before creation:', TWEEN.getAll().length);

        for (let i = 0; i < vectorLength; i++) {
            // For InstancedMesh, we don't have individual ellipse objects.
            // We animate the instance's matrix directly (or a property controlling it).
            // The "targetPosition" logic needs to be rethought.
            // vec1 is the branched (source) vector, vec2 is the original (target) vector.
            // The branched vector's instances will move towards the original vector's instances.

            // Get world position of the target instance (vec2, instance i)
            const targetWorldPos = new THREE.Vector3();
            const tempMatrixTarget = new THREE.Matrix4();
            vec2.mesh.getMatrixAt(i, tempMatrixTarget);
            targetWorldPos.setFromMatrixPosition(tempMatrixTarget);
            targetWorldPos.applyMatrix4(vec2.group.matrixWorld); // Apply group's world transform

            // Convert target world position to local space of vec1's group
            const localTargetPosInVec1Group = vec1.group.worldToLocal(targetWorldPos.clone());

            // We need the current local Y of instance 'i' in vec1
            const currentInstanceMatrix = new THREE.Matrix4();
            vec1.mesh.getMatrixAt(i, currentInstanceMatrix);
            const currentInstanceLocalPos = new THREE.Vector3().setFromMatrixPosition(currentInstanceMatrix);


            const moveTween = new TWEEN.Tween({ y: currentInstanceLocalPos.y }, true) // Tween a dummy object
                .to({ y: localTargetPosInVec1Group.y }, duration)
                .easing(TWEEN.Easing.Quadratic.InOut)
                .delay(i * delayBetweenCubes)
                .onUpdate((obj) => {
                    vec1.setInstanceYOffset(i, obj.y); // Update instance's Y position
                    if (DEBUG && i === 0) {
                        // console.log('[MoveTween onUpdate] vec1 instance 0 local y =', obj.y);
                    }
                })
                .onStart(() => {
                    if (DEBUG && i === 0 && !additionPlaying.tweenLogged) {
                        // console.log('[MoveTween onStart] First tween started. From y=', currentInstanceLocalPos.y, 'to', localTargetPosInVec1Group.y);
                        additionPlaying.tweenLogged = true;
                    }
                })
                .onComplete(() => {
                    if (DEBUG && i === 0) console.log('[MoveTween onComplete] First tween completed for instance 0');

                    // Flash effect on target instance (vec2, instance i)
                    const originalColor = new THREE.Color();
                    vec2.mesh.getColorAt(i, originalColor); // Get current color

                    // Set to white for flash
                    vec2.setInstanceColor(i, new THREE.Color(0xffffff));

                    new TWEEN.Tween({}, true) // Dummy tween for timing
                        .to({}, flashDuration)
                        .onComplete(() => {
                            // Calculate new summed value & color
                            // Note: vec1.rawData[i] should be the value from the *branched* vector
                            //       vec2.rawData[i] is from the *original* vector
                            const sum = vec1.rawData[i] + vec2.rawData[i];
                            vec2.rawData[i] = sum; // Update raw data on target
                            // Normalized data is not directly updated here, relies on updateData if needed later
                            // For immediate visual, calculate new color from sum
                            const newColor = mapValueToColor( (sum - vec2.normalizedData.reduce((a,b)=>a+b,0)/VECTOR_LENGTH) / Math.sqrt(vec2.normalizedData.reduce((acc, val, idx, arr) => acc + Math.pow(val - vec2.rawData.reduce((a,b)=>a+b,0)/VECTOR_LENGTH, 2), 0)/VECTOR_LENGTH + 1e-5)); // A quick map based on sum
                            // More correctly, we should re-normalize vec2.rawData and then map colors.
                            // For now, use the simple approach:
                            const remappedSumColor = mapValueToColor(sum); // Or use a re-normalized value


                            // Restore instance material to new color (on vec2)
                            vec2.setInstanceColor(i, remappedSumColor); // Use the color from sum

                            // "Hide" the source instance (vec1, instance i) by moving it far away
                            vec1.setInstanceYOffset(i, HIDE_INSTANCE_Y_OFFSET);
                        })
                        .start(getVirtualNow());
                });

            moveTween.start(getVirtualNow());
            if (DEBUG && i === 0) {
                // console.log("moveTween.start() called for i=0");
            }
        }

        if (DEBUG) console.log('[startAdditionAnimation] active tweens after creation:', TWEEN.getAll().length);
    }


    function resetVectorAddition(vec1, vec2) { // vec1 is source (branched), vec2 is target (original)
        const len = VECTOR_LENGTH; // vec1.normalizedData.length;
        for (let i = 0; i < len; i++) {
            // Reset source instance (vec1)
            vec1.setInstanceYOffset(i, 0); // Reset Y offset
            // Color is already set from its normalizedData, no need to change unless data changes

            // Reset target instance (vec2)
            // Recalculate color based on its current (potentially summed) normalizedData
            const color = mapValueToColor(vec2.normalizedData[i]);
            vec2.setInstanceColor(i, color);
        }
        // Ensure raw data of vec2 is updated if its normalizedData was what we conceptually summed into
        // This part might need more thought if rawData is the source of truth.
        // For now, assume normalizedData drives colors, and rawData was updated in startAdditionAnimation.
    }

    // --- Create and Position Vectors over Slits ---
    const slitSpacing = matrixParams.depth / (matrixParams.numberOfSlits + 1);
    // Define animation bounds
    const startY = -matrixParams.height / 2 - vectorHeightOffset; // Start below the matrix
    const endY = matrixParams.height / 2 + vectorHeightOffset;   // End above the matrix
    const animationDistance = endY - startY;

    for (let i = 0; i < matrixParams.numberOfSlits; i++) {
        // Generate unique data for each vector, matching VECTOR_LENGTH
        const vectorData = Array.from({ length: VECTOR_LENGTH }, () => (Math.random() - 0.5) * 10);

        // Calculate Z position based on slit index
        const vectorZPos = -matrixParams.depth / 2 + slitSpacing * (i + 1);
        // Set initial position (start of animation)
        const initialPosition = new THREE.Vector3(0, startY, vectorZPos);

        // Use VectorVisualizationInstanced constructor
        const vectorVis = new VectorVisualizationInstanced(vectorData, initialPosition);
        // vectorVis.data is now vectorVis.rawData or vectorVis.normalizedData

        // Apply initial vector control parameters
        if (vectorVis.mesh && vectorVis.mesh.material) {
            const mat = vectorVis.mesh.material;
            mat.opacity = vectorControlParams.opacity;
            mat.transparent = vectorControlParams.opacity < 1.0;

            // Safely set properties, checking if they exist on the material
            if ('metalness' in mat) mat.metalness = vectorControlParams.metalness;
            if ('roughness' in mat) mat.roughness = vectorControlParams.roughness;
            if ('emissiveIntensity' in mat) mat.emissiveIntensity = vectorControlParams.emissiveIntensity;
            
            if (mat.color) { // mat.color should exist on MeshBasicMaterial
                mat.color.set(vectorControlParams.baseColor); // Set base color
            }
            if (mat.emissive) { // mat.emissive does not exist on MeshBasicMaterial by default
                mat.emissive.set(vectorControlParams.baseColor); // Set base emissive color
            }
            // Material type is set by default in VectorVisualizationInstanced,
            // can be changed via updateAllVectorMaterials if a different default is needed initially.
            // For now, we assume the default MeshStandardMaterial is fine and apply properties.
        }


        scene.add(vectorVis.group);
        allVectorVisualizations.push(vectorVis);

        // --- Create Duplicate Branched Vector ---
        const branchedInitialPosition = new THREE.Vector3(branchConfig.branchX, startY, vectorZPos);
        const branchedVectorVis = new VectorVisualizationInstanced(vectorData.slice(), branchedInitialPosition); // duplicate data

        // Apply same initial material overrides as originals
        if (branchedVectorVis.mesh && branchedVectorVis.mesh.material && vectorVis.mesh && vectorVis.mesh.material) {
            // Clone properties, not the material instance itself, as it's shared within InstancedMesh
            const bMat = branchedVectorVis.mesh.material;
            const oMat = vectorVis.mesh.material;
            bMat.opacity = oMat.opacity;
            bMat.transparent = oMat.transparent;

            // Safely copy properties, checking if they exist on both materials
            if ('metalness' in oMat && 'metalness' in bMat) bMat.metalness = oMat.metalness;
            if ('roughness' in oMat && 'roughness' in bMat) bMat.roughness = oMat.roughness;
            if ('emissiveIntensity' in oMat && 'emissiveIntensity' in bMat) bMat.emissiveIntensity = oMat.emissiveIntensity;
            
            if (oMat.color && bMat.color) {
                bMat.color.copy(oMat.color);
            }
            if (oMat.emissive && bMat.emissive) {
                bMat.emissive.copy(oMat.emissive);
            }
        }

        scene.add(branchedVectorVis.group);
        branchedVectorVisualizations.push(branchedVectorVis);
        additionPlayedFlags.push(false);

        // Trail for duplicate
        branchedTrailPoints.push([]);
        const branchedTrailGeometry = new THREE.BufferGeometry();
        const branchedPositions = new Float32Array(MAX_TRAIL_POINTS * 3);
        branchedTrailGeometry.setAttribute('position', new THREE.BufferAttribute(branchedPositions, 3));
        const branchedTrailMaterial = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: scaleOpacityForDisplay(0.05) });
        const branchedTrailLine = new THREE.Line(branchedTrailGeometry, branchedTrailMaterial);
        scene.add(branchedTrailLine);
        // Seed branched trail starting directly beneath the right‑hand matrix
        const initialBPos = [branchConfig.branchX, startY, vectorZPos];
        branchedTrailPoints[i].push(initialBPos);
        branchedTrailGeometry.getAttribute('position').setXYZ(0, ...initialBPos);
        branchedTrailGeometry.setDrawRange(0, 1);
        branchedTrailGeometry.computeBoundingSphere();
        branchedTrailLines.push({ line: branchedTrailLine, geometry: branchedTrailGeometry, material: branchedTrailMaterial });

        // Initialize frozen flag for original trail
        originalTrailFrozenFlags.push(false);
        trailExtendActive.push(false);
        trailExtendStartTimes.push(0);
        trailExtendStartYs.push(startY);
        trailExtendTargetYs.push(startY);

        // --- Create Trail Line for this Vector ---
        allTrailPoints.push([]); // Initialize points array for this trail
        const trailGeometry = new THREE.BufferGeometry();
        // Pre-allocate buffer large enough for max points
        const positions = new Float32Array(MAX_TRAIL_POINTS * 3);
        trailGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

        const trailMaterial = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: scaleOpacityForDisplay(0.05) });
        const trailLine = new THREE.Line(trailGeometry, trailMaterial);
        scene.add(trailLine);
        // Seed original trail with first point as well
        const initialPos = [0, startY, vectorZPos];
        allTrailPoints[i].push(initialPos);
        trailGeometry.getAttribute('position').setXYZ(0, ...initialPos);
        trailGeometry.setDrawRange(0, 1);
        trailGeometry.computeBoundingSphere();
        allTrailLines.push({ line: trailLine, geometry: trailGeometry, material: trailMaterial });
    }

    // --- GUI ---
    const gui = new GUI({ closeFolders: true });
    gui.close();

    // New Hemisphere Light GUI Folder
    const hemiLightFolder = gui.addFolder('Hemisphere Light');
    hemiLightFolder.addColor(hemisphereLight, 'color').name('Sky Color');
    hemiLightFolder.addColor(hemisphereLight, 'groundColor').name('Ground Color');
    hemiLightFolder.add(hemisphereLight, 'intensity', 0, 3, 0.05).name('Intensity');
    // hemiLightFolder.open();

    // Bloom Effect Folder
    const bloomFolder = gui.addFolder('Bloom Effect');
    bloomFolder.add(bloomParams, 'strength', 0, 3, 0.05).name('Strength').onChange((value) => { bloomPass.strength = value; });
    bloomFolder.add(bloomParams, 'radius', 0, 1, 0.01).name('Radius').onChange((value) => { bloomPass.radius = value; });
    bloomFolder.add(bloomParams, 'threshold', 0, 1, 0.01).name('Threshold').onChange((value) => { bloomPass.threshold = value; });
    // bloomFolder.open();

    // Matrix Folder
    const matrixFolder = gui.addFolder('Matrix');
    matrixFolder.add(matrixParams, 'material', Object.keys(availableMaterials)).name('Material').onChange(updateMatrixMaterial);
    matrixFolder.addColor(matrixParams, 'color').name('Color').onChange(updateMatrixMaterial);
    matrixFolder.add(matrixParams, 'numberOfSlits', 0, 20, 1)
        .name('Number of Slits')
        .onChange(updateBothMatrixGeometry);
    matrixFolder.add(matrixParams, 'slitWidth', 0.1, 5.0, 0.05).name('Slit Width').onChange(updateBothMatrixGeometry);
    matrixFolder.add(matrixParams, 'slitDepthFactor', 0, 1, 0.01).name('Slit Depth Factor').onChange(updateBothMatrixGeometry);
    matrixFolder.add(matrixParams, 'slitWidthFactor', 0.1, 1, 0.01).name('Slit Width Factor').onChange(updateBothMatrixGeometry);
    matrixFolder.add(matrixParams, 'width', 1, 100, 0.5).name('Base Width').onChange(updateBothMatrixGeometry);
    matrixFolder.add(matrixParams, 'topWidthFactor', 0.1, 2, 0.01).name('Top Width Factor').onChange(updateBothMatrixGeometry);
    matrixFolder.add(matrixParams, 'cornerRadius', 0, 5, 0.05).name('Corner Radius').onChange(updateBothMatrixGeometry);
    matrixFolder.add(matrixParams, 'depth', 5, 100, 1)
        .name('Depth')
        .onChange(updateBothMatrixGeometry);
    matrixFolder.add(matrixParams, 'opacity', 0, 1, 0.01).name('Opacity').onChange((value) => {
        matrixParams.opacity = value; // Store value
        updateMatrixMaterial(); // Recreate material with new opacity
    });
    // matrixFolder.open(); // Keep closed by default

    // Vectors Folder
    const vectorFolder = gui.addFolder('Vectors');
    vectorFolder.add(vectorControlParams, 'material', Object.keys(availableMaterials)).name('Material').onChange(value => updateAllVectorMaterials('material', value));
    vectorFolder.add(vectorControlParams, 'opacity', 0, 1, 0.01).name('Opacity').onChange(value => updateAllVectorMaterials('opacity', value));
    vectorFolder.add(vectorControlParams, 'metalness', 0, 1, 0.01).name('Metalness').onChange(value => updateAllVectorMaterials('metalness', value));
    vectorFolder.add(vectorControlParams, 'roughness', 0, 1, 0.01).name('Roughness').onChange(value => updateAllVectorMaterials('roughness', value));
    vectorFolder.add(vectorControlParams, 'emissiveIntensity', 0, 1, 0.01).name('Emissive Intensity').onChange(value => updateAllVectorMaterials('emissiveIntensity', value));
    vectorFolder.add(vectorControlParams, 'baseColor', 0x000000, 0xffffff).name('Base Color (Instanced)').onChange(value => updateAllVectorMaterials('baseColor', value));
    // vectorFolder.open(); // Keep closed by default

    // --- Resize Handling ---
    window.addEventListener('resize', onWindowResize, false);
    function onWindowResize() {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
        composer.setSize(window.innerWidth, window.innerHeight); // Update composer size
    }

    // --- Keyboard Event Listeners ---
    function onKeyDown(event) {
        keysPressed[event.key.toLowerCase()] = true;
        keysPressed[event.code] = true; // Also store by code for arrow keys
    }

    function onKeyUp(event) {
        keysPressed[event.key.toLowerCase()] = false;
        keysPressed[event.code] = false;
    }

    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);

    const clock = new THREE.Clock(); // Clock for animation timing
    let previousY = startY; // Initialize previous Y position tracking

    const additionPlaying = { active: false, endTime: 0, logged: false, tweenLogged: false }; // Global flag to pause reset while addition runs

    // --- Page Visibility Pause/Resume Support ---
    let isPaused = false; // Tracks if the animation is currently paused because the tab is hidden
    let accumulatedElapsedBeforePause = 0; // Total elapsedTime before the current pause
    let tweenTimeOffset = 0; // Total time removed from TWEEN due to pauses
    let pauseStartTime = 0;  // Absolute time when pause began

    // Helper to get the "virtual" timeline that excludes paused durations
    const getVirtualNow = () => performance.now() - tweenTimeOffset;

    function pauseAnimation() {
        if (isPaused) return;
        isPaused = true;
        // Add the time up to this moment into our accumulator
        accumulatedElapsedBeforePause += clock.getElapsedTime(); // Preserve elapsed up to the pause moment
        // Stop the clock so it does not advance further
        clock.stop();

        // Mark when this pause started so we can offset TWEEN timing
        pauseStartTime = performance.now();
    }

    function resumeAnimation() {
        if (!isPaused) return;
        isPaused = false;
        // Restart clock from zero; we'll add the accumulated offset each frame
        clock.start();

        // Increase the TWEEN time offset by the actual pause duration so tweens resume seamlessly
        if (pauseStartTime !== 0) {
            tweenTimeOffset += performance.now() - pauseStartTime;
            pauseStartTime = 0;
        }
    }

    // Listen for tab visibility changes to trigger pause / resume
    const visibilityHandler = () => {
        if (document.hidden) {
            pauseAnimation();
        } else {
            resumeAnimation();
        }
    };
    document.addEventListener('visibilitychange', visibilityHandler);

    // --- Post-Addition Rise Parameters ---
    const extraRiseDistance = 15; // How far to continue rising after combination
    const extraRiseDuration = 3;  // Seconds it takes to reach the extra height
    let extraRiseStartTime = null;

    // --- Animation Loop ---
    function animate() {
        requestAnimationFrame(animate);

        // If the tab is hidden and the animation is paused, skip all updates to
        // avoid large jumps in simulation state when the user returns.
        if (isPaused) {
            return; // Nothing else to do this frame
        }

        // Rotate the ambient star field slowly for subtle motion parallax.
        starField.rotation.y += 0.00025;
        starField.rotation.x += 0.0001;

        // --- Handle Keyboard Movement ---
        const cameraRight = new THREE.Vector3();
        camera.getWorldDirection(cameraRight); // Get forward direction first
        cameraRight.cross(camera.up).normalize(); // Calculate right vector
        const cameraUp = new THREE.Vector3().copy(camera.up); // Use world up for panning usually
        const cameraForward = new THREE.Vector3(); // Vector to store forward direction
        camera.getWorldDirection(cameraForward); // Get the camera's forward direction

        // WASD for Panning (Move camera and target together)
        if (keysPressed['w']) {
            const panOffset = cameraUp.clone().multiplyScalar(panSpeed);
            camera.position.add(panOffset);
            controls.target.add(panOffset);
        }
        if (keysPressed['s']) {
            const panOffset = cameraUp.clone().multiplyScalar(-panSpeed);
            camera.position.add(panOffset);
            controls.target.add(panOffset);
        }
        if (keysPressed['a']) {
            const panOffset = cameraRight.clone().multiplyScalar(-panSpeed);
            camera.position.add(panOffset);
            controls.target.add(panOffset);
        }
        if (keysPressed['d']) {
            const panOffset = cameraRight.clone().multiplyScalar(panSpeed);
            camera.position.add(panOffset);
            controls.target.add(panOffset);
        }

        // Arrow Up/Down for Rotating View Angle (Adjusting OrbitControls target vertically)
        if (keysPressed['ArrowUp']) {
            const targetOffset = cameraUp.clone().multiplyScalar(rotateSpeed * 10); // Tilt view upward
            controls.target.add(targetOffset);
        }
        if (keysPressed['ArrowDown']) {
            const targetOffset = cameraUp.clone().multiplyScalar(-rotateSpeed * 10); // Tilt view downward
            controls.target.add(targetOffset);
        }

        // Arrow Left/Right for Rotating View Angle (Adjusting OrbitControls target horizontally)
        if (keysPressed['ArrowLeft']) {
            const targetOffset = cameraRight.clone().multiplyScalar(-rotateSpeed * 10); // Adjust target left
            controls.target.add(targetOffset);
        }
        if (keysPressed['ArrowRight']) {
            const targetOffset = cameraRight.clone().multiplyScalar(rotateSpeed * 10); // Adjust target right
            controls.target.add(targetOffset);
        }

        controls.update(); // Required if enableDamping is true

        // Update tweens (addition animations) using virtual time that excludes pauses
        const updated = TWEEN.update(getVirtualNow());

        // Calculate scene elapsed time (for vector motion) which also stops during pauses
        const elapsedTime = accumulatedElapsedBeforePause + clock.getElapsedTime();

        // Log position *after* tween update
        if (DEBUG && additionPlaying.active) {
            // Log position of the first ellipse of the first vector during the animation
            // This needs to be adapted for InstancedMesh if specific instance data is needed.
            // For now, we can log the group position or skip detailed instance logging here.
            if (branchedVectorVisualizations[0] && branchedVectorVisualizations[0].group) {
                 // console.log("Animate loop: branchedVectorVisualizations[0].group.position.y = ", branchedVectorVisualizations[0].group.position.y);
                 // To get specific instance data, it's more involved:
                 const tempMatrix = new THREE.Matrix4();
                 branchedVectorVisualizations[0].mesh.getMatrixAt(0, tempMatrix);
                 const instancePos = new THREE.Vector3().setFromMatrixPosition(tempMatrix);
                 // console.log("Animate loop: branched instance[0] local y =", instancePos.y);
            }
        }

        const animationDuration = 5; // seconds to reach the meeting point once
        const loopTime = elapsedTime / animationDuration; // allow to grow beyond 1 so vectors keep moving

        // Calculate current Y position using linear interpolation within the loop
        const currentY = startY + loopTime * animationDistance;

        // Mark addition finished when its expected time passes
        if (additionPlaying.active && getVirtualNow() > additionPlaying.endTime) {
            additionPlaying.active = false;
        }

        // If all addition animations have played and finished, start the extra rise (once)
        if (!additionPlaying.active && extraRiseStartTime === null && additionPlayedFlags.every(f => f)) {
            // Store based on the unified elapsedTime value that includes pre-pause accumulator
            extraRiseStartTime = elapsedTime;
        }

        // Calculate any extra rise offset based on time since extraRiseStartTime
        let extraRiseOffset = 0;
        if (extraRiseStartTime !== null) {
            const riseElapsed = elapsedTime - extraRiseStartTime;
            const tRise = Math.min(riseElapsed / extraRiseDuration, 1); // 0..1
            extraRiseOffset = THREE.MathUtils.lerp(0, extraRiseDistance, tRise);
        }

        // --- Update Trail Lines for Original Vectors ---
        allTrailLines.forEach((trail, index) => {
            const currentPoints = allTrailPoints[index];
            const vectorZPos = allVectorVisualizations[index].group.position.z; // Z is constant per vector
            // Before computing trailY, insert retrieval of center ellipses
            const centerIndex = Math.floor(VECTOR_LENGTH / 2);
            const origCenterWorld = new THREE.Vector3();
            const branchedCenterWorld = new THREE.Vector3();

            // Get world positions for the center instances
            if (allVectorVisualizations[index] && allVectorVisualizations[index].mesh) {
                const originalVec = allVectorVisualizations[index];
                const tempMatrixOrig = new THREE.Matrix4();
                originalVec.mesh.getMatrixAt(centerIndex, tempMatrixOrig);
                origCenterWorld.setFromMatrixPosition(tempMatrixOrig);
                origCenterWorld.applyMatrix4(originalVec.group.matrixWorld);
            }
            if (branchedVectorVisualizations[index] && branchedVectorVisualizations[index].mesh) {
                const branchedVec = branchedVectorVisualizations[index];
                const tempMatrixBranched = new THREE.Matrix4();
                branchedVec.mesh.getMatrixAt(centerIndex, tempMatrixBranched);
                branchedCenterWorld.setFromMatrixPosition(tempMatrixBranched);
                branchedCenterWorld.applyMatrix4(branchedVec.group.matrixWorld);
            }

            // Determine Y position based on state (normal, extending, frozen)
            let trailY;
            if (originalTrailFrozenFlags[index]) {
                // After freeze, follow branched center ellipse
                trailY = branchedCenterWorld.y;
            } else if (trailExtendActive[index]) {
                const now = performance.now();
                const tExt = Math.min((now - trailExtendStartTimes[index]) / trailExtendDuration, 1);
                trailY = THREE.MathUtils.lerp(trailExtendStartYs[index], trailExtendTargetYs[index], tExt);
                if (tExt >= 1) {
                    trailY = trailExtendTargetYs[index];
                    trailExtendActive[index] = false;
                    originalTrailFrozenFlags[index] = true;
                }
            } else {
                // Use original center ellipse position
                trailY = origCenterWorld.y;
            }

            const newPoint = [0, trailY, vectorZPos];

            if (originalTrailFrozenFlags[index]) {
                // Just update the very last point so the tip keeps following the merged path
                if (currentPoints.length > 0) {
                    currentPoints[currentPoints.length - 1][1] = trailY;
                }
            } else {
                // Record new point while trail is still growing
                if (currentPoints.length === 0 ||
                    newPoint[0] !== currentPoints[currentPoints.length - 1][0] ||
                    newPoint[1] !== currentPoints[currentPoints.length - 1][1] ||
                    newPoint[2] !== currentPoints[currentPoints.length - 1][2]) {
                    if (currentPoints.length < MAX_TRAIL_POINTS) currentPoints.push(newPoint);
                }
            }

            // Update geometry attribute
            const positionAttribute = trail.geometry.getAttribute('position');
            const posAttr = positionAttribute;
            const prevDrawCount = trail.geometry.drawRange.count || 0;

            const drawCount = currentPoints.length;

            // Update only the newest point (last index)
            const lastIdx = drawCount - 1;
            posAttr.setXYZ(lastIdx, newPoint[0], newPoint[1], newPoint[2]);

            trail.geometry.setDrawRange(0, drawCount);
            posAttr.needsUpdate = true;

            // Recompute bounding sphere only when we actually added a vertex this frame
            if (!originalTrailFrozenFlags[index] && drawCount !== prevDrawCount) {
                trail.geometry.computeBoundingSphere();
            }
        });

        /* ==============================================================
         *  BRANCHING PATH DISABLED – Using simple two‑matrix merge flow
         *  ------------------------------------------------------------
         *  1. Both vector sets (left & right) ascend vertically through
         *     their respective matrices until `mergeConfig.vertDurationT`.
         *  2. Right‑hand vectors then slide horizontally to X = 0 over
         *     `mergeConfig.moveLeftDurationT` of the timeline, lining up
         *     with the left vectors.
         *  3. When the slide finishes we trigger the addition animation
         *     (right → left).
         * ============================================================*/

        const vertPhaseT   = mergeConfig.vertDurationT;
        const slidePhaseT  = mergeConfig.moveLeftDurationT;
        const yEnd         = startY + animationDistance; // final vertical height (just above matrices)

        branchedVectorVisualizations.forEach((bVecVis, index) => {
            const zPos = bVecVis.group.position.z;

            let xPos   = mergeConfig.rightX;
            let yPos   = yEnd; // default final height

            if (loopTime < vertPhaseT) {
                // Ascend vertically only
                const tVert = loopTime / vertPhaseT;
                yPos = THREE.MathUtils.lerp(startY, yEnd, tVert);
                xPos = mergeConfig.rightX;
            } else {
                // Slide horizontally towards centre
                const tSlide = Math.min((loopTime - vertPhaseT) / slidePhaseT, 1);
                yPos = yEnd;
                xPos = THREE.MathUtils.lerp(mergeConfig.rightX, 0, tSlide);

                // Trigger addition once (when slide finishes)
                if (!additionPlayedFlags[index] && tSlide >= 1) {
                    // Move units FROM the lower/left vector set (allVectorVisualizations[index])
                    // TO the upper/right one (bVecVis)
                    // So, source is allVectorVisualizations[index], target is bVecVis
                    // BUT, the current animation has branched moving to original.
                    // startAdditionAnimation expects (sourceInstanceContainer, targetInstanceContainer)
                    // vec1 is source (moves), vec2 is target (receives value, flashes)
                    // In the current setup: branchedVectorVisualizations[index] (bVecVis) is the source,
                    // and allVectorVisualizations[index] is the target.
                    startAdditionAnimation(allVectorVisualizations[index], bVecVis);
                    additionPlayedFlags[index] = true;

                    // Initialize smooth trail extension parameters (grow original trail during addition)
                    trailExtendActive[index] = true;
                    trailExtendStartTimes[index] = performance.now();
                    trailExtendStartYs[index] = allVectorVisualizations[index].group.position.y + extraRiseOffset; // Current original position
                    trailExtendTargetYs[index] = yPos + extraRiseOffset; // Meeting point
                }
            }

            // Apply extra rise offset so merged vectors continue moving up
            bVecVis.group.position.set(xPos, yPos + extraRiseOffset, zPos);

            // --- Simple trail update for branched vectors ---
            const bTrail = branchedTrailLines[index];
            const bPoints = branchedTrailPoints[index];
            const bNew = [xPos, yPos + extraRiseOffset, zPos];
            const prevLen = bPoints.length;
            if (prevLen === 0 || bNew[0] !== bPoints[prevLen - 1][0] || bNew[1] !== bPoints[prevLen - 1][1] || bNew[2] !== bPoints[prevLen - 1][2]) {
                if (bPoints.length < MAX_TRAIL_POINTS) bPoints.push(bNew);
            }

            const newLen = bPoints.length;
            const bPosAttr = bTrail.geometry.getAttribute('position');
            const newIdx = newLen - 1;
            bPosAttr.setXYZ(newIdx, bNew[0], bNew[1], bNew[2]);

            bTrail.geometry.setDrawRange(0, newLen);
            bPosAttr.needsUpdate = true;

            // Update bounding sphere only when we actually grew the trail
            if (newLen !== prevLen) {
                bTrail.geometry.computeBoundingSphere();
            }
        });

        // --- Matrix Color Animation ---
        // Check if the current material uses standard color property
        const canAnimateColor = matrixParams.material !== 'MeshNormalMaterial' && matrixParams.material !== 'MeshDepthMaterial';

        if (canAnimateColor && matrixVis.group) {
            const matrixBottomY = -matrixParams.height / 2;
            const matrixMidY = 0; // Center of the matrix
            const matrixTopY = matrixParams.height / 2;

            // Get base color from GUI and derive dark version
            baseMatrixColor.set(matrixParams.color);
            const outsideColor = baseMatrixColor.clone().multiplyScalar(0.2);

            let t = 0;
            if (currentY < matrixBottomY) {
                matrixCurrentColor.copy(outsideColor);
            } else if (currentY >= matrixBottomY && currentY < matrixMidY) {
                // Entering: Dark Base to Bright Yellow
                t = (currentY - matrixBottomY) / (matrixMidY - matrixBottomY);
                matrixCurrentColor.lerpColors(outsideColor, brightYellow, t);
            } else if (currentY >= matrixMidY && currentY < matrixTopY) {
                // Leaving: Bright Yellow to Dark Yellow
                t = (currentY - matrixMidY) / (matrixTopY - matrixMidY);
                matrixCurrentColor.lerpColors(brightYellow, darkYellow, t);
            } else { // currentY >= matrixTopY
                matrixCurrentColor.copy(darkYellow);
            }

            // Apply the calculated color to all mesh materials of BOTH matrices
            matrixVis.group.children.forEach(child => {
                if (child instanceof THREE.Mesh && child.material && child.material.color) {
                    child.material.color.copy(matrixCurrentColor);
                }
            });

            // --- Colour animation for the BRANCHED matrix based on the first branched vector's Y ---
            let branchedVectorY = branchedVectorVisualizations.length > 0 ? branchedVectorVisualizations[0].group.position.y : startY;
            if (branchedVectorY < matrixBottomY) {
                matrixCurrentColor2.copy(outsideColor);
            } else if (branchedVectorY >= matrixBottomY && branchedVectorY < matrixMidY) {
                const t2 = (branchedVectorY - matrixBottomY) / (matrixMidY - matrixBottomY);
                matrixCurrentColor2.lerpColors(outsideColor, brightYellow, t2);
            } else if (branchedVectorY >= matrixMidY && branchedVectorY < matrixTopY) {
                const t2 = (branchedVectorY - matrixMidY) / (matrixTopY - matrixMidY);
                matrixCurrentColor2.lerpColors(brightYellow, darkYellow, t2);
            } else {
                matrixCurrentColor2.copy(darkYellow);
            }

            branchedMatrixVis.group.children.forEach(child => {
                if (child instanceof THREE.Mesh && child.material && child.material.color) {
                    child.material.color.copy(matrixCurrentColor2);
                }
            });
        }

        // Update original vector positions – stop once they reach below branched height
        const yOriginalStop = yEnd - branchConfig.meetYOffset;
        allVectorVisualizations.forEach(vectorVis => {
            // Allow vertical movement until reaching stop height
            let baseY;
            if (currentY < yOriginalStop) {
                baseY = currentY;
            } else {
                baseY = yOriginalStop;
            }

            vectorVis.group.position.y = baseY + extraRiseOffset;
        });

        // Update previous Y for next frame
        previousY = currentY;

        // Use composer to render with post-processing effects
        composer.render();
        // renderer.render(scene, camera); // Replaced by composer.render()
    }

    function createRotatingStarField() {
        const group = new THREE.Group();

        const starCount = 1000;
        const positions = new Float32Array(starCount * 3);
        const colors = new Float32Array(starCount * 3);

        const colorOuter = new THREE.Color(0x4f91ff);
        const colorInner = new THREE.Color(0xffffff);

        for (let i = 0; i < starCount; i++) {
            // Distribute stars within a spherical shell to surround the scene.
            const radius = THREE.MathUtils.randFloat(120, 420);
            const theta = THREE.MathUtils.randFloat(0, Math.PI * 2);
            const phi = Math.acos(THREE.MathUtils.randFloatSpread(2));

            const sinPhi = Math.sin(phi);
            const x = radius * sinPhi * Math.cos(theta);
            const y = radius * Math.cos(phi);
            const z = radius * sinPhi * Math.sin(theta);

            positions[i * 3] = x;
            positions[i * 3 + 1] = y;
            positions[i * 3 + 2] = z;

            // Blend star colours so distant stars skew cooler.
            const lerpT = THREE.MathUtils.inverseLerp(120, 420, radius);
            const starColor = colorInner.clone().lerp(colorOuter, lerpT);
            colors[i * 3] = starColor.r;
            colors[i * 3 + 1] = starColor.g;
            colors[i * 3 + 2] = starColor.b;
        }

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({
            size: 1.6,
            sizeAttenuation: true,
            transparent: true,
            opacity: 0.85,
            depthWrite: false,
            vertexColors: true,
        });

        const points = new THREE.Points(geometry, material);
        group.add(points);

        return group;
    }

    animate(); // Start the animation loop

    // Return cleanup function
    return () => {
        window.removeEventListener('resize', onWindowResize);
        window.removeEventListener('keydown', onKeyDown); // Remove key listeners
        window.removeEventListener('keyup', onKeyUp);     // Remove key listeners
        document.removeEventListener('visibilitychange', visibilityHandler);
        controls.dispose();
        gui.destroy();

        // Dispose Bloom Pass Render Targets if they exist
        if (bloomPass && bloomPass.renderTargetBright) bloomPass.renderTargetBright.dispose();
        if (bloomPass && bloomPass.renderTargetBlur) bloomPass.renderTargetBlur.dispose();
        // We don't explicitly dispose the composer itself, but ensure passes are handled.

        // Dispose geometries and materials
        allVectorVisualizations.forEach(vec => vec.dispose()); // Dispose all vectors (handles instanced mesh geometry/material)
        allTrailLines.forEach(trail => { // Dispose original trails
            trail.geometry.dispose();
            trail.material.dispose();
        });

        branchedVectorVisualizations.forEach(vec => vec.dispose()); // Same for branched
        branchedTrailLines.forEach(trail => {
            trail.geometry.dispose();
            trail.material.dispose();
        });

        // Dispose materials and geometries for both matrices
        allMatrixVisualizations.forEach(matVis => {
            let matMaterial = null;
            if (matVis.group) {
                matVis.group.children.forEach(child => {
                    if (child instanceof THREE.Mesh && child.material && !matMaterial) {
                        matMaterial = child.material;
                    }
                });
            }
            if (matMaterial && typeof matMaterial.dispose === 'function') {
                matMaterial.dispose();
            }
            matVis._clearMesh();
        });

        if (starField) {
            starField.children.forEach(child => {
                if (child.geometry && typeof child.geometry.dispose === 'function') {
                    child.geometry.dispose();
                }
                if (child.material) {
                    if (Array.isArray(child.material)) {
                        child.material.forEach(mat => mat.dispose && mat.dispose());
                    } else if (typeof child.material.dispose === 'function') {
                        child.material.dispose();
                    }
                }
            });
            scene.remove(starField);
        }

        renderer.dispose();
    };
}
