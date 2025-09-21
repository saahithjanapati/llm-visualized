import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GUI } from 'three/examples/jsm/libs/lil-gui.module.min.js';
import { VectorVisualization } from '../components/VectorVisualization';
import { WeightMatrixVisualization } from '../components/WeightMatrixVisualization';
import { VECTOR_LENGTH } from '../utils/constants.js'; // Import VECTOR_LENGTH
import { scaleOpacityForDisplay } from '../utils/trailConstants.js';

const JOYFUL_PALETTE = [
    0xff6f61, // lively coral
    0xffd166, // warm sunshine
    0x06d6a0, // mint green
    0x118ab2, // vibrant blue
    0x9d4edd, // playful violet
];

function easeInOutCubic(t) {
    if (t < 0.5) {
        return 4 * t * t * t;
    }
    const f = -2 * t + 2;
    return 1 - (f * f * f) / 2;
}

function clamp01(value) {
    return Math.min(1, Math.max(0, value));
}

// Maximum points per trail line (adjust for performance/length)
const MAX_TRAIL_POINTS = 1000; // Further increased buffer size

export function initVectorMatrixScene(canvas) {
    // --- Basic Three.js setup ---
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0e1433); // Deep navy backdrop for contrast

    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 10, 25); // Position camera to view both objects

    const renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

    // --- Lighting ---
    const ambientLight = new THREE.AmbientLight(0xfff3e0, 0.65);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.6);
    directionalLight.position.set(5, 12, 9);
    scene.add(directionalLight);
    const magentaLight = new THREE.PointLight(0xff85a1, 0.6, 120);
    magentaLight.position.set(-18, 6, 18);
    scene.add(magentaLight);
    const cyanLight = new THREE.PointLight(0x5af3ff, 0.55, 140);
    cyanLight.position.set(18, 10, -16);
    scene.add(cyanLight);

    // --- Controls ---
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

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
    const vectorAnimationStates = []; // Store per-vector joyful animation data
    const vectorHeightOffset = 40; // Significantly increased offset

    // Configure matrix with user specified defaults
    const matrixParams = {
        width: 57.5,
        height: 8, // Increased thickness
        depth: 72,
        topWidthFactor: 0.76,
        cornerRadius: 0.1, // Keeping previous default, adjust if needed
        numberOfSlits: 5,
        slitWidth: 2.0, // Increased slit width
        slitDepthFactor: 1.0,
        slitWidthFactor: 0.6,
    };
    const matrixVis = new WeightMatrixVisualization(null, new THREE.Vector3(0, 0, 0), matrixParams.width, matrixParams.height, matrixParams.depth, matrixParams.topWidthFactor, matrixParams.cornerRadius, matrixParams.numberOfSlits, matrixParams.slitWidth, matrixParams.slitDepthFactor, matrixParams.slitWidthFactor);
    scene.add(matrixVis.group);

    // Define colors for animation
    const darkGray = new THREE.Color(0x333333);
    const brightYellow = new THREE.Color(0xFFFF00);
    const darkYellow = new THREE.Color(0xCCA000);
    const matrixCurrentColor = new THREE.Color(); // Color object to reuse for lerping

    // --- Create and Position Vectors over Slits ---
    const slitSpacing = matrixParams.depth / (matrixParams.numberOfSlits + 1);
    // Define animation bounds
    const startY = -matrixParams.height / 2 - vectorHeightOffset; // Start below the matrix
    const endY = matrixParams.height / 2 + vectorHeightOffset;   // End above the matrix
    const animationDistance = endY - startY;

    for (let i = 0; i < matrixParams.numberOfSlits; i++) {
        // Generate unique data for each vector, matching VECTOR_LENGTH
        const vectorData = Array.from({ length: VECTOR_LENGTH }, () => (Math.random() - 0.5) * 10);
        const vectorParams = {
            numberOfElements: VECTOR_LENGTH, // Use VECTOR_LENGTH
            elementWidth: 1, // Match vector element width if needed
            spacing: 1.5,    // Match vector spacing if needed
            // Add other params if needed, using defaults otherwise
        };
        const vectorVis = new VectorVisualization(vectorData, vectorParams);

        // Calculate Z position based on slit index
        const vectorZPos = -matrixParams.depth / 2 + slitSpacing * (i + 1);
        // Set initial position (start of animation)
        vectorVis.group.position.set(0, startY, vectorZPos);

        scene.add(vectorVis.group);
        allVectorVisualizations.push(vectorVis); // Add to array for cleanup

        const accentColor = new THREE.Color(JOYFUL_PALETTE[i % JOYFUL_PALETTE.length]);
        vectorVis.group.userData.accentColor = accentColor;
        vectorVis.ellipses.forEach((ellipse) => {
            const tinted = ellipse.material.color.clone().lerp(accentColor, 0.35);
            ellipse.material.color.copy(tinted);
            ellipse.material.emissive.copy(accentColor);
            ellipse.material.emissiveIntensity = 0.35;
            ellipse.material.needsUpdate = true;
        });

        const highlightIndices = [
            Math.floor(VECTOR_LENGTH * 0.2) + (i % 3),
            Math.floor(VECTOR_LENGTH * 0.5),
            Math.floor(VECTOR_LENGTH * 0.8) - (i % 4),
        ].map(index => THREE.MathUtils.clamp(index, 0, VECTOR_LENGTH - 1));

        vectorAnimationStates.push({
            accentColor,
            highlightIndices,
            previousY: startY,
            wobblePhase: Math.random() * Math.PI * 2,
            wobbleSpeed: 1.4 + Math.random() * 0.6,
            bobAmplitude: 1.2 + Math.random() * 1.0,
            swayAmplitude: 0.4 + Math.random() * 0.3,
        });

        // --- Create Trail Line for this Vector ---
        allTrailPoints.push([]); // Initialize points array for this trail
        const trailGeometry = new THREE.BufferGeometry();
        // Pre-allocate buffer large enough for max points
        const positions = new Float32Array(MAX_TRAIL_POINTS * 3);
        trailGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

        const trailMaterial = new THREE.LineBasicMaterial({ color: accentColor, transparent: true, opacity: scaleOpacityForDisplay(0.22) });
        const trailLine = new THREE.Line(trailGeometry, trailMaterial);
        scene.add(trailLine);
        allTrailLines.push({ line: trailLine, geometry: trailGeometry, material: trailMaterial });
    }

    // --- GUI (Optional - for parameter tuning) ---
    const gui = new GUI();
    // const vectorFolder = gui.addFolder('Vector'); // Remove vector folder

    const matrixFolder = gui.addFolder('Matrix');
    matrixFolder.add(matrixParams, 'numberOfSlits', 0, 20, 1).name('Number of Slits').onChange(() => { /* Need complex update logic now */ }); // Note: Changing slits requires recreating vectors
    matrixFolder.add(matrixParams, 'slitWidth', 0.1, 5.0, 0.05).name('Slit Width').onChange(() => matrixVis.updateGeometry(matrixParams)); // Increased max slit width
    matrixFolder.add(matrixParams, 'slitDepthFactor', 0, 1, 0.01).name('Slit Depth Factor').onChange(() => matrixVis.updateGeometry(matrixParams));
    matrixFolder.add(matrixParams, 'slitWidthFactor', 0.1, 1, 0.01).name('Slit Width Factor').onChange(() => matrixVis.updateGeometry(matrixParams));
    matrixFolder.add(matrixParams, 'width', 1, 100, 0.5).name('Base Width').onChange(() => matrixVis.updateGeometry(matrixParams));
    matrixFolder.add(matrixParams, 'topWidthFactor', 0.1, 2, 0.01).name('Top Width Factor').onChange(() => matrixVis.updateGeometry(matrixParams));
    matrixFolder.add(matrixParams, 'cornerRadius', 0, 5, 0.05).name('Corner Radius').onChange(() => matrixVis.updateGeometry(matrixParams));
    matrixFolder.add(matrixParams, 'depth', 5, 100, 1).name('Depth').onChange(() => { /* Need complex update logic now */ }); // Note: Changing depth requires repositioning vectors
    matrixFolder.open(); // Open matrix folder by default

    // --- Resize Handling ---
    window.addEventListener('resize', onWindowResize, false);
    function onWindowResize() {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
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
    let accumulatedTime = 0;
    let previousProgress = 0;

    // --- Animation Loop ---
    function animate() {
        requestAnimationFrame(animate);

        // --- Handle Keyboard Movement ---
        const cameraRight = new THREE.Vector3();
        camera.getWorldDirection(cameraRight); // Get forward direction first
        cameraRight.cross(camera.up).normalize(); // Calculate right vector
        const cameraUp = new THREE.Vector3().copy(camera.up); // Use world up for panning usually

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

        // Arrow Up/Down for Tilting (Pitch - Adjusting OrbitControls target vertically)
        if (keysPressed['ArrowUp']) {
            const targetOffset = cameraUp.clone().multiplyScalar(rotateSpeed * 10); // Adjust target up
            controls.target.add(targetOffset);
        }
        if (keysPressed['ArrowDown']) {
            const targetOffset = cameraUp.clone().multiplyScalar(-rotateSpeed * 10); // Adjust target down
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

        const deltaTime = clock.getDelta();
        accumulatedTime += deltaTime;
        const animationDuration = 5; // seconds for one full loop (up and down implied by modulo)
        const loopTime = (accumulatedTime % animationDuration) / animationDuration; // Normalized loop time (0 to 1)
        const easedLoopTime = easeInOutCubic(loopTime);

        let currentY = startY + easedLoopTime * animationDistance;

        const anticipationWindow = 0.12;
        if (loopTime < anticipationWindow) {
            const t = loopTime / anticipationWindow;
            currentY -= Math.sin(t * Math.PI) * 3.5;
        }
        const followThroughWindow = 0.14;
        if (loopTime > 1 - followThroughWindow) {
            const t = (loopTime - (1 - followThroughWindow)) / followThroughWindow;
            currentY += Math.sin(t * Math.PI) * 2.8;
        }

        // --- Check for Animation Reset ---
        const animationReset = loopTime < previousProgress;
        if (animationReset) {
            // Clear points for all trails
            allTrailPoints.forEach(points => points.length = 0);
        }
        previousProgress = loopTime;

        const animatedYs = [];

        allVectorVisualizations.forEach((vectorVis, index) => {
            const state = vectorAnimationStates[index];
            if (!state) return;

            state.wobblePhase += deltaTime * state.wobbleSpeed;
            const bob = Math.sin(state.wobblePhase) * state.bobAmplitude;
            const lift = Math.sin((loopTime + index * 0.12) * Math.PI * 2) * 0.6;
            const animatedY = currentY + bob + lift;
            const velocity = deltaTime > 0 ? (animatedY - state.previousY) / deltaTime : 0;
            const stretch = THREE.MathUtils.clamp(1 + velocity * 0.03, 0.78, 1.35);
            const squash = 1 / Math.sqrt(stretch);

            vectorVis.group.scale.set(squash, stretch, squash);
            const sway = Math.sin(state.wobblePhase * 0.8 + loopTime * Math.PI * 2) * state.swayAmplitude;
            vectorVis.group.position.set(sway, animatedY, vectorVis.group.position.z);
            vectorVis.group.rotation.z = Math.sin(state.wobblePhase + loopTime * Math.PI * 2) * 0.15;

            state.highlightIndices.forEach((highlightIdx, hi) => {
                const ellipse = vectorVis.ellipses[highlightIdx];
                if (!ellipse || !ellipse.material) return;
                const pulse = (Math.sin(state.wobblePhase + hi * Math.PI * 0.75 + loopTime * Math.PI * 2) + 1) * 0.5;
                ellipse.material.emissiveIntensity = THREE.MathUtils.lerp(0.35, 0.95, pulse);
                ellipse.material.needsUpdate = true;
            });

            animatedYs[index] = animatedY;
            state.previousY = animatedY;
        });

        // --- Update Trail Lines ---
        allTrailLines.forEach((trail, index) => {
            const currentPoints = allTrailPoints[index];
            const vectorVis = allVectorVisualizations[index];
            const vectorZPos = vectorVis.group.position.z; // Get Z for this vector
            const newPoint = [vectorVis.group.position.x, animatedYs[index] ?? currentY, vectorZPos];

            // Add new point
            currentPoints.push(newPoint);

            // Update geometry attribute
            const positionAttribute = trail.geometry.getAttribute('position');
            for (let j = 0; j < currentPoints.length; j++) {
                positionAttribute.setXYZ(j, currentPoints[j][0], currentPoints[j][1], currentPoints[j][2]);
            }

            trail.geometry.setDrawRange(0, currentPoints.length);
            positionAttribute.needsUpdate = true; 
        });

        // --- Matrix Color Animation ---
        const referenceY = animatedYs[0] ?? currentY;
        const matrixBottomY = -matrixParams.height / 2;
        const matrixMidY = 0; // Center of the matrix
        const matrixTopY = matrixParams.height / 2;

        let t = 0;
        if (referenceY < matrixBottomY) {
            matrixCurrentColor.copy(darkGray);
        } else if (referenceY >= matrixBottomY && referenceY < matrixMidY) {
            // Entering: Gray to Bright Yellow with playful tint
            t = clamp01((referenceY - matrixBottomY) / (matrixMidY - matrixBottomY));
            const enterColor = brightYellow.clone().lerp(new THREE.Color(0xff8fab), 0.35);
            matrixCurrentColor.lerpColors(darkGray, enterColor, t);
        } else if (referenceY >= matrixMidY && referenceY < matrixTopY) {
            // Leaving: Bright Yellow to Dark Yellow
            t = clamp01((referenceY - matrixMidY) / (matrixTopY - matrixMidY));
            const exitColor = darkYellow.clone().lerp(new THREE.Color(0x7c3aed), 0.4);
            matrixCurrentColor.lerpColors(brightYellow, exitColor, t);
        } else { // referenceY >= matrixTopY
            const finalColor = darkYellow.clone().lerp(new THREE.Color(0x5eead4), 0.45);
            matrixCurrentColor.copy(finalColor);
        }
        matrixVis.setColor(matrixCurrentColor);

        renderer.render(scene, camera);
    }

    animate(); // Start the animation loop

    // Return cleanup function
    return () => {
        window.removeEventListener('resize', onWindowResize);
        window.removeEventListener('keydown', onKeyDown); // Remove key listeners
        window.removeEventListener('keyup', onKeyUp);     // Remove key listeners
        controls.dispose();
        gui.destroy();
        // Dispose geometries and materials
        allVectorVisualizations.forEach(vec => vec.dispose()); // Dispose all vectors
        allTrailLines.forEach(trail => { // Dispose trails
            trail.geometry.dispose();
            trail.material.dispose();
        });
        matrixVis._clearMesh(); // Use existing clear method
        // Add disposal for other objects if needed
        renderer.dispose();
    };
} 