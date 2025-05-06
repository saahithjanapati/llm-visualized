import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GUI } from 'three/examples/jsm/libs/lil-gui.module.min.js';
import { LayerNormalizationVisualization } from '../components/LayerNormalizationVisualization.js';
import { WeightMatrixVisualization } from '../components/WeightMatrixVisualization.js';

// Build a static (non-animated) visualisation of an entire 12-layer GPT block.
// The layout is kept deliberately simple: every layer is placed one above the
// other along +Y with a fixed vertical spacing.  Each layer contains, from
// bottom to top:
//     LayerNorm → Multi-Head Attention → LayerNorm → MLP (two matrices)
//
// For the attention block we reuse a *single* geometry and render it via three
// InstancedMesh objects (one for Q, K, V) so that every attention head across
// *all* layers shares GPU buffers.  This keeps draw-calls to just three for the
// whole transformer stack.
// -----------------------------------------------------------------------------
export function initGPTModelVisualization(container) {
    // ───────────────────────────────────────────────────────────────────────────
    // Scene / camera / renderer
    // ───────────────────────────────────────────────────────────────────────────
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);

    // We need a fairly tall view – position far on Z and shift a bit up on Y so
    // that the user immediately sees the whole tower.
    const NUM_LAYERS    = 12;
    const LAYER_HEIGHT  = 180; // increased per-layer vertical span (world units)
    const totalHeight   = NUM_LAYERS * LAYER_HEIGHT;

    const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 5000);
    camera.position.set(0, totalHeight * 0.55, 1600);

    let renderer;
    if (container instanceof HTMLCanvasElement) {
        renderer = new THREE.WebGLRenderer({ canvas: container, antialias: true });
    } else {
        renderer = new THREE.WebGLRenderer({ antialias: true });
        container.appendChild(renderer.domElement);
    }
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);

    // ───────────────────────────────────────────────────────────────────────────
    // Controls & lights
    // ───────────────────────────────────────────────────────────────────────────
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, totalHeight * 0.55, 0);

    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const dir = new THREE.DirectionalLight(0xffffff, 0.9);
    dir.position.set(25, totalHeight, 40);
    scene.add(dir);

    // ───────────────────────────────────────────────────────────────────────────
    // Shared geometries & materials
    // ───────────────────────────────────────────────────────────────────────────
    // Weight-matrix base parameters (matches MultiHeadAttentionAnimation)
    const matrixParams = {
        width: 37.5,
        height: 12,
        depth: 200,
        topWidthFactor: 0.47,
        cornerRadius: 1.2,
        numberOfSlits: 5,
        slitWidth: 1.85,
        slitDepthFactor: 1.0,
        slitBottomWidthFactor: 0.95,
        slitTopWidthFactor: 0.37
    };

    // Build *one* WeightMatrixVisualization then harvest its geometry so all
    // instances share the buffer.  We intentionally skip adding the helper's
    // group to the scene – it's only used as a geometry factory.
    const geomFactory = new WeightMatrixVisualization(null, new THREE.Vector3(),
        matrixParams.width, matrixParams.height, matrixParams.depth,
        matrixParams.topWidthFactor, matrixParams.cornerRadius, matrixParams.numberOfSlits,
        matrixParams.slitWidth, matrixParams.slitDepthFactor,
        matrixParams.slitBottomWidthFactor, matrixParams.slitTopWidthFactor
    );

    const baseGeometry = geomFactory.mesh.geometry; // shared across all heads

    // Materials – bright distinct colours
    const qMat = new THREE.MeshStandardMaterial({ color: 0x3388ff, metalness: 0.2, roughness: 0.6, transparent: true, opacity: 0.85 });
    const kMat = new THREE.MeshStandardMaterial({ color: 0x33ff88, metalness: 0.2, roughness: 0.6, transparent: true, opacity: 0.85 });
    const vMat = new THREE.MeshStandardMaterial({ color: 0xff3355, metalness: 0.2, roughness: 0.6, transparent: true, opacity: 0.85 });

    // Instanced meshes — one for each of Q, K, V
    const MAX_HEADS   = 12; // GPT size (fewer for bigger models)
    const MAX_LAYERS  = NUM_LAYERS; // constant for this viz

    // GUI-controlled state
    const state = {
        layers: MAX_LAYERS,
        heads:  MAX_HEADS
    };

    const totalInstances = MAX_LAYERS * MAX_HEADS;

    const qMesh = new THREE.InstancedMesh(baseGeometry, qMat, totalInstances);
    const kMesh = new THREE.InstancedMesh(baseGeometry, kMat, totalInstances);
    const vMesh = new THREE.InstancedMesh(baseGeometry, vMat, totalInstances);

    // Store original transforms so we can restore quickly when toggling visibility
    const originalQ = new Array(totalInstances);
    const originalK = new Array(totalInstances);
    const originalV = new Array(totalInstances);
    const hiddenMatrix     = new THREE.Matrix4().makeScale(0, 0, 0);

    // Helper to compute per-instance transform matrices
    const tmpMat   = new THREE.Matrix4();
    const tmpPos   = new THREE.Vector3();
    const tmpQuat  = new THREE.Quaternion();
    const tmpScale = new THREE.Vector3(1, 1, 1);

    // Horizontal layout identical to MultiHeadAttentionAnimation
    const singleSetWidth = 3 * matrixParams.width;
    const HEAD_SET_GAP   = 60;
    const totalWidth     = MAX_HEADS * singleSetWidth + (MAX_HEADS - 1) * HEAD_SET_GAP;
    const firstSet_Q_Center_X = -totalWidth / 2 + matrixParams.width / 2;

    // Dimensions for non-instanced blocks (defined BEFORE use)
    const lnWidth = 40;
    const lnHeight = 23;
    const lnDepth = 72;

    const mlpWidth  = 37.5;
    const mlpHeight = 14;
    const mlpDepth  = 50;

    const attentionYOffset = 40; // Y-offset of attention block within a layer

    // X position where the *left* edge of the attention block starts
    const leftEdgeX = -totalWidth / 2;
    // Pre-computed centres so LayerNorm/MLP left-edges align with the attention block
    const lnCentreX  = leftEdgeX + lnWidth / 2;
    const mlpCentreX = leftEdgeX + mlpWidth / 2;

    let instanceIdx = 0;
    for (let layer = 0; layer < NUM_LAYERS; layer++) {
        const layerBaseY = layer * LAYER_HEIGHT;

        for (let head = 0; head < MAX_HEADS; head++) {
            const setOffset = head * (singleSetWidth + HEAD_SET_GAP);
            const x_q = firstSet_Q_Center_X + setOffset;
            const x_k = x_q + matrixParams.width;
            const x_v = x_k + matrixParams.width;

            // Q
            tmpPos.set(x_q, layerBaseY + attentionYOffset, 0);
            tmpMat.compose(tmpPos, tmpQuat, tmpScale);
            qMesh.setMatrixAt(instanceIdx, tmpMat.clone());
            originalQ[instanceIdx] = tmpMat.clone();

            // K
            tmpPos.set(x_k, layerBaseY + attentionYOffset, 0);
            tmpMat.compose(tmpPos, tmpQuat, tmpScale);
            kMesh.setMatrixAt(instanceIdx, tmpMat.clone());
            originalK[instanceIdx] = tmpMat.clone();

            // V
            tmpPos.set(x_v, layerBaseY + attentionYOffset, 0);
            tmpMat.compose(tmpPos, tmpQuat, tmpScale);
            vMesh.setMatrixAt(instanceIdx, tmpMat.clone());
            originalV[instanceIdx] = tmpMat.clone();

            instanceIdx++;
        }
    }

    qMesh.instanceMatrix.needsUpdate = true;
    kMesh.instanceMatrix.needsUpdate = true;
    vMesh.instanceMatrix.needsUpdate = true;

    scene.add(qMesh, kMesh, vMesh);

    // ───────────────────────────────────────────────────────────────────────────
    // Per-layer: LayerNorms & MLP (non-instanced; lighter count)
    // ───────────────────────────────────────────────────────────────────────────
    const lnColour1 = new THREE.Color(0x00e0ff);
    const lnColour2 = new THREE.Color(0xff00ff);
    const mlpColour1 = new THREE.Color(0xffcc00);
    const mlpColour2 = new THREE.Color(0xff8800);

    // We reuse geometry by holding one base LayerNormVis & one base MLP matrix
    const lnFactory = new LayerNormalizationVisualization(new THREE.Vector3(), lnWidth, lnHeight, lnDepth, 1.0, 5, 2.5, 3.75);
    const lnBaseGeom = lnFactory.mesh.geometry;

    const mlpUpFactory   = new WeightMatrixVisualization(null, new THREE.Vector3(), mlpWidth, mlpHeight, mlpDepth, 1.4, 1.0, 4, 1.6, 1.0, 1.0, 0.3);
    const mlpDownFactory = new WeightMatrixVisualization(null, new THREE.Vector3(), mlpWidth, mlpHeight, mlpDepth, 0.4, 1.0, 4, 1.6, 1.0, 0.3, 1.0);

    // Material & instancing for LayerNorms (two per layer) – we could instance,
    // but the count is small (24) so separate meshes are fine.

    for (let layer = 0; layer < NUM_LAYERS; layer++) {
        const baseY = layer * LAYER_HEIGHT;

        // Bottom LayerNorm ------------------------------------------------------
        {
            const mesh = new THREE.Mesh(lnBaseGeom, new THREE.MeshStandardMaterial({ color: lnColour1, metalness: 0.2, roughness: 0.6, transparent: true, opacity: 0.85 }));
            mesh.position.set(lnCentreX, baseY + 10, 0);
            scene.add(mesh);
        }

        // Top LayerNorm (after attention) --------------------------------------
        {
            const mesh = new THREE.Mesh(lnBaseGeom, new THREE.MeshStandardMaterial({ color: lnColour2, metalness: 0.2, roughness: 0.6, transparent: true, opacity: 0.85 }));
            mesh.position.set(lnCentreX, baseY + attentionYOffset + 35, 0);
            scene.add(mesh);
        }

        // MLP block (two matrices) ---------------------------------------------
        {
            // Bottom (upsample) matrix
            const upMesh = new THREE.Mesh(mlpUpFactory.mesh.geometry, new THREE.MeshStandardMaterial({ color: mlpColour1, metalness: 0.15, roughness: 0.65, transparent: true, opacity: 0.85 }));
            const upCenterY = baseY + attentionYOffset + 60;
            upMesh.position.set(mlpCentreX, upCenterY, 0);
            scene.add(upMesh);

            // Top (downsample) matrix – placed directly above the upsample block
            const downMesh = new THREE.Mesh(mlpDownFactory.mesh.geometry, new THREE.MeshStandardMaterial({ color: mlpColour2, metalness: 0.15, roughness: 0.65, transparent: true, opacity: 0.85 }));
            downMesh.position.set(mlpCentreX, upCenterY + mlpHeight, 0);
            scene.add(downMesh);
        }
    }

    // ───────────────────────────────────────────────────────────────────────────
    // GUI: dynamic head-count control
    // ────────────────────────────────────────────────────────────────────
    const gui = new GUI();
    gui.add(state, 'layers', 1, MAX_LAYERS, 1).name('Visible Layers').onChange(updateVisibility);
    gui.add(state, 'heads',  1, MAX_HEADS,  1).name('Heads / Layer').onChange(updateVisibility);

    function updateVisibility() {
        let idx = 0;
        for (let l = 0; l < MAX_LAYERS; l++) {
            const layerVisible = l < state.layers;
            for (let h = 0; h < MAX_HEADS; h++) {
                const show = layerVisible && h < state.heads;
                qMesh.setMatrixAt(idx, show ? originalQ[idx] : hiddenMatrix);
                kMesh.setMatrixAt(idx, show ? originalK[idx] : hiddenMatrix);
                vMesh.setMatrixAt(idx, show ? originalV[idx] : hiddenMatrix);
                idx++;
            }
        }
        qMesh.instanceMatrix.needsUpdate = true;
        kMesh.instanceMatrix.needsUpdate = true;
        vMesh.instanceMatrix.needsUpdate = true;
    }

    // Apply initial visibility (in case defaults < max)
    updateVisibility();

    // ───────────────────────────────────────────────────────────────────────────
    // Resize handler & render loop
    // ───────────────────────────────────────────────────────────────────────────
    function onWindowResize() {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    }
    window.addEventListener('resize', onWindowResize);

    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }
    animate();

    // Cleanup callback ---------------------------------------------------------
    return () => {
        window.removeEventListener('resize', onWindowResize);
        controls.dispose();
        gui.destroy();
        renderer.dispose();
        scene.traverse(obj => {
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) {
                if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
                else obj.material.dispose();
            }
        });
    };
} 