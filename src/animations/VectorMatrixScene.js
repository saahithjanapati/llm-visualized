import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GUI } from 'three/examples/jsm/libs/lil-gui.module.min.js';
import { VectorVisualization } from '../components/VectorVisualization';
import { WeightMatrixVisualization } from '../components/WeightMatrixVisualization';
import { VECTOR_LENGTH } from '../utils/constants.js'; // Import VECTOR_LENGTH

export function initVectorMatrixScene(canvas) {
    // --- Basic Three.js setup ---
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111); // Dark background

    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 10, 25); // Position camera to view both objects

    const renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);

    // --- Lighting ---
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(5, 10, 7.5);
    scene.add(directionalLight);

    // --- Controls ---
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // --- Visualizations ---
    const allVectorVisualizations = []; // Array to hold all vector instances
    const vectorHeightOffset = 20; // Further increased offset for larger animation range

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
    matrixFolder.add(matrixParams, 'depth', 5, 100, 1).name('Depth').onChange(() => { /* Need complex update logic now */ }); // Note: Changing depth requires repositioning vectors
    matrixFolder.open(); // Open matrix folder by default

    // --- Resize Handling ---
    window.addEventListener('resize', onWindowResize, false);
    function onWindowResize() {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    }

    const clock = new THREE.Clock(); // Clock for animation timing

    // --- Animation Loop ---
    function animate() {
        requestAnimationFrame(animate);
        controls.update(); // Required if enableDamping is true

        const elapsedTime = clock.getElapsedTime();
        const animationDuration = 5; // seconds for one full loop (up and down implied by modulo)
        const loopTime = (elapsedTime % animationDuration) / animationDuration; // Normalized loop time (0 to 1)

        // Calculate current Y position using linear interpolation within the loop
        const currentY = startY + loopTime * animationDistance;

        // --- Matrix Color Animation ---
        const matrixBottomY = -matrixParams.height / 2;
        const matrixMidY = 0; // Center of the matrix
        const matrixTopY = matrixParams.height / 2;

        let t = 0;
        if (currentY < matrixBottomY) {
            matrixCurrentColor.copy(darkGray);
        } else if (currentY >= matrixBottomY && currentY < matrixMidY) {
            // Entering: Gray to Bright Yellow
            t = (currentY - matrixBottomY) / (matrixMidY - matrixBottomY);
            matrixCurrentColor.lerpColors(darkGray, brightYellow, t);
        } else if (currentY >= matrixMidY && currentY < matrixTopY) {
            // Leaving: Bright Yellow to Dark Yellow
            t = (currentY - matrixMidY) / (matrixTopY - matrixMidY);
            matrixCurrentColor.lerpColors(brightYellow, darkYellow, t);
        } else { // currentY >= matrixTopY
            matrixCurrentColor.copy(darkYellow);
        }
        matrixVis.setColor(matrixCurrentColor);

        // Update vector positions
        allVectorVisualizations.forEach(vectorVis => {
            vectorVis.group.position.y = currentY;
        });

        renderer.render(scene, camera);
    }

    animate(); // Start the animation loop

    // Return cleanup function
    return () => {
        window.removeEventListener('resize', onWindowResize);
        controls.dispose();
        gui.destroy();
        // Dispose geometries and materials
        allVectorVisualizations.forEach(vec => vec.dispose()); // Dispose all vectors
        matrixVis._clearMesh(); // Use existing clear method
        // Add disposal for other objects if needed
        renderer.dispose();
    };
} 