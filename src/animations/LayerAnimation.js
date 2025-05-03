import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { LayerNormalizationVisualization } from '../components/LayerNormalizationVisualization.js';
import { VectorVisualization } from '../components/VectorVisualization.js';
import { VectorNormalizationVisualization } from '../components/VectorNormalizationVisualization.js';
import { VECTOR_LENGTH } from '../utils/constants.js';
import { mapValueToColor } from '../utils/colors.js';

// NOTE: Requires global TWEEN.js (loaded separately via <script>)

export function initLayerAnimation(container) {
    // -------------------------------------------------------------------------
    //  Basic Three.js setup
    // -------------------------------------------------------------------------
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);

    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 20, 70);

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

    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(10, 30, 10);
    scene.add(dirLight);

    // -------------------------------------------------------------------------
    //  Visualisation blocks (only right-hand LayerNorm)
    // -------------------------------------------------------------------------
    const lnParams = {
        width: 40,
        height: 25,
        depth: 72,
        wallThickness: 1.0,
        numberOfHoles: 5,
        holeWidth: 2.5,
        holeWidthFactor: 3.75
    };
    const branchX = 80; // Horizontal offset for right-hand LayerNorm
    const layerNorm = new LayerNormalizationVisualization(
        new THREE.Vector3(branchX, 0, 0),
        lnParams.width,
        lnParams.height,
        lnParams.depth,
        lnParams.wallThickness,
        lnParams.numberOfHoles,
        lnParams.holeWidth,
        lnParams.holeWidthFactor
    );
    scene.add(layerNorm.group);

    // -------------------------------------------------------------------------
    //  Main path parameters (no central matrix)
    // -------------------------------------------------------------------------
    const offsetY = 10;
    const startY = -lnParams.height / 2 - offsetY;   // Spawn below LayerNorm height
    const meetYOffset = 5;                           // Meeting point above LayerNorm top
    const meetY = lnParams.height / 2 + meetYOffset; // Y where originals wait for merge

    const branchStartY = startY + 5; // trigger branch when originals have risen just 5 units

    // Z-lane spacing same as number of holes
    const numVectors = lnParams.numberOfHoles;
    const slitSpacing = lnParams.depth / (numVectors + 1);

    // Motion speeds
    const riseSpeedOriginal = 3;   // slower up speed for originals
    const horizSpeed = 15;         // horizontal move speed for duplicates/result
    const riseSpeedInsideLN = 6;   // duplicate vertical speed inside LN
    const mergeGap = 7; // branched result stops this much below originals before merge

    // Per-lane collections
    const originals = [];
    const lanes = []; // each lane object will mirror LayerNormPipeline logic and extra states

    // --- Trail line support --------------------------------------------------------
    const MAX_TRAIL_POINTS = 1500;
    function createTrailLine(color) {
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(MAX_TRAIL_POINTS * 3);
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setDrawRange(0, 0);
        const material = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.12 });
        const line = new THREE.Line(geometry, material);
        scene.add(line);
        return { line, geometry, positions, points: [] };
    }

    for (let i = 0; i < numVectors; i++) {
        const zPos = -lnParams.depth / 2 + slitSpacing * (i + 1);

        // ---------- Original vector on main (centre) path ----------
        const data = Array.from({ length: VECTOR_LENGTH }, () => Math.random() * 2 - 1);
        const origVec = new VectorVisualization(data, new THREE.Vector3(0, startY, zPos));
        origVec.data = [...data];
        scene.add(origVec.group);
        originals.push(origVec);

        // ---------- Duplicate moving vector (will branch) ----------
        const movingVec = new VectorNormalizationVisualization(new THREE.Vector3(0, startY, zPos));
        movingVec.originalData = [...data];
        movingVec.normalizedData = movingVec.layerNormalize(data);
        movingVec.data = movingVec.normalizedData;
        scene.add(movingVec.group);

        // Start hidden – will appear once branch begins
        movingVec.group.visible = false;

        // ---------- Static vectors inside LayerNorm ----------
        const multTarget = new VectorVisualization(data.slice(), new THREE.Vector3(branchX, 0, zPos));
        multTarget.data = [...data];
        scene.add(multTarget.group);

        const addStartInsideY = lnParams.height / 4; // quarter height above centre
        const addVec = new VectorVisualization(data.slice(), new THREE.Vector3(branchX, addStartInsideY, zPos));
        addVec.data = [...data];
        scene.add(addVec.group);

        // Create trails
        const origTrail = createTrailLine(0xffffff);
        const branchTrail = createTrailLine(0xffffff);

        lanes.push({
            zPos,
            originalVec: origVec,
            movingVec,
            multTarget,
            addVec,
            // Pipeline flags
            normStarted: false,
            multStarted: false,
            multDone: false,
            addStarted: false,
            addDone: false,
            // Horizontal / merge states
            horizPhase: 'waiting', // waiting | right | insideLN | moveLeft | merged
            resultVec: null,
            mergeStarted: false,
            origTrail,
            branchTrail
        });
    }

    // -------------------------------------------------------------------------
    //  Helper: multiplication animation (copied from pipeline)
    // -------------------------------------------------------------------------
    function startMultiplicationAnimation(vec1, vec2, onComplete) {
        const duration = 750;
        const vectorLength = vec1.ellipses.length;
        let moveTweensCompleted = 0;

        for (let i = 0; i < vectorLength; i++) {
            const e1 = vec1.ellipses[i];
            const e2 = vec2.ellipses[i];
            if (!e1 || !e2) continue;
            const targetPosWorld = new THREE.Vector3();
            e2.getWorldPosition(targetPosWorld);
            const localTarget = e1.parent.worldToLocal(targetPosWorld.clone());

            new TWEEN.Tween(e1.position)
                .to({ y: localTarget.y }, duration)
                .easing(TWEEN.Easing.Quadratic.InOut)
                .onStart(() => { if (e1.material) e1.material.depthWrite = false; })
                .onComplete(() => {
                    e1.visible = false;
                    if (e1.material) e1.material.depthWrite = true;
                    moveTweensCompleted++;
                    if (moveTweensCompleted === vectorLength) triggerFlash();
                })
                .start();
        }

        function triggerFlash() {
            const flashDuration = 150;
            const original = [];
            for (let i = 0; i < vectorLength; i++) {
                const e2 = vec2.ellipses[i];
                original[i] = {
                    color: e2.material.color.clone(),
                    emissive: e2.material.emissive.clone(),
                    emissiveIntensity: e2.material.emissiveIntensity
                };
                e2.material.color.set(0xffffff);
                e2.material.emissive.set(0xffffff);
                e2.material.emissiveIntensity = 1.5;
            }
            new TWEEN.Tween({}).to({}, flashDuration).onComplete(() => {
                for (let i = 0; i < vectorLength; i++) {
                    const e1 = vec1.ellipses[i];
                    const e2 = vec2.ellipses[i];
                    const product = vec1.data[i] * vec2.data[i];
                    vec2.data[i] = product;
                    const col = mapValueToColor(product);
                    e2.material.color.copy(col);
                    e2.material.emissive.copy(col);
                    e2.material.emissiveIntensity = original[i].emissiveIntensity;
                    if (e1) e1.visible = false;
                }
                if (onComplete) onComplete();
            }).start();
        }
    }

    // -------------------------------------------------------------------------
    //  Helper: addition animation (same as pipeline)
    // -------------------------------------------------------------------------
    function startAdditionAnimation(vec1, vec2, onComplete) {
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
    }

    // helper to push position into trail
    function updateTrail(trailObj, pos) {
        const pts = trailObj.points;
        // Only add if changed
        if (pts.length === 0 || pos.x !== pts[pts.length - 1][0] || pos.y !== pts[pts.length - 1][1] || pos.z !== pts[pts.length - 1][2]) {
            if (pts.length < MAX_TRAIL_POINTS) pts.push([pos.x, pos.y, pos.z]);
            const idx = pts.length - 1;
            trailObj.geometry.attributes.position.setXYZ(idx, pos.x, pos.y, pos.z);
            trailObj.geometry.setDrawRange(0, pts.length);
            trailObj.geometry.attributes.position.needsUpdate = true;
            if (idx % 20 === 0) trailObj.geometry.computeBoundingSphere();
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
    function animate() {
        requestAnimationFrame(animate);
        const deltaTime = clock.getDelta();
        const timeNow = performance.now();

        lanes.forEach((lane, idx) => {
            const { originalVec, movingVec, multTarget, addVec } = lane;

            // -------------------- ORIGINAL VEC RISE --------------------
            const branchFinalY = meetY; // branched vectors end at the merge height
            const originalStopY = meetY - mergeGap; // originals stop below branched vectors
            if (originalVec.group.position.y < originalStopY) {
                originalVec.group.position.y += riseSpeedOriginal * deltaTime;
                if (originalVec.group.position.y > originalStopY) originalVec.group.position.y = originalStopY;
            }

            // Update original trail: track center ellipse during merge, else use group position
            if (lane.mergeStarted) {
                const centerIndex = Math.floor(VECTOR_LENGTH / 2);
                const centerEllipse = originalVec.ellipses[centerIndex];
                const worldPos = new THREE.Vector3();
                centerEllipse.getWorldPosition(worldPos);
                updateTrail(lane.origTrail, worldPos);
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
                    const dx = horizSpeed * deltaTime;
                    movingVec.group.position.x = Math.min(branchX, movingVec.group.position.x + dx);
                    if (movingVec.group.position.x >= branchX) {
                        movingVec.group.position.x = branchX;
                        lane.horizPhase = 'insideLN';
                    }
                    break;
                }
                case 'insideLN': {
                    // ---------------- LayerNorm pipeline behaviour ----------------
                    const bottomY = -lnParams.height / 2;
                    const midY = 0;
                    const topY = lnParams.height / 2;

                    // Start normalization when reaching 35% height above bottom
                    const normStartY = bottomY + (midY - bottomY) * 0.35;
                    if (!lane.normStarted && movingVec.group.position.y >= normStartY) {
                        movingVec.startAnimation();
                        lane.normStarted = true;
                    }

                    // Update normalization visuals
                    movingVec.update(timeNow);

                    // Move up (only when not actively normalizing)
                    const normAnimating = lane.normStarted && movingVec.animationState.isAnimating;
                    if (!lane.multStarted && !normAnimating) {
                        movingVec.group.position.y += riseSpeedInsideLN * deltaTime;
                    }

                    // Trigger multiplication at centre
                    if (!lane.multStarted && movingVec.group.position.y >= midY) {
                        lane.multStarted = true;
                        startMultiplicationAnimation(movingVec, multTarget, () => {
                            lane.multDone = true;
                            movingVec.group.visible = false;
                        });
                    }

                    // After multiplication, trigger addition
                    if (lane.multDone && !lane.addStarted) {
                        lane.addStarted = true;
                        startAdditionAnimation(multTarget, addVec, () => {
                            lane.addDone = true;
                            addVec.group.visible = false; // we will create result copy
                        });
                    }

                    // After addition completes create rising result once per lane
                    if (lane.addDone && !lane.resultVec) {
                        // Create rising copy like pipeline
                        const resultData = [...addVec.data];
                        const resultVec = new VectorVisualization(resultData, addVec.group.position.clone());
                        resultVec.data = [...resultData];
                        scene.add(resultVec.group);
                        // Copy material appearance from addVec
                        for (let i = 0; i < VECTOR_LENGTH; i++) {
                            if (addVec.ellipses[i] && resultVec.ellipses[i]) {
                                resultVec.ellipses[i].material.color.copy(addVec.ellipses[i].material.color);
                                resultVec.ellipses[i].material.emissive.copy(addVec.ellipses[i].material.emissive);
                                resultVec.ellipses[i].material.emissiveIntensity = addVec.ellipses[i].material.emissiveIntensity;
                            }
                        }

                        lane.resultVec = resultVec;

                        // Rise just above LN top
                        const finalY = branchFinalY; // fully clear LayerNorm
                        const distance = finalY - resultVec.group.position.y;
                        const riseDuration = (distance / riseSpeedInsideLN) * 1000;

                        new TWEEN.Tween(resultVec.group.position)
                            .to({ y: finalY }, riseDuration)
                            .easing(TWEEN.Easing.Linear.None)
                            .onComplete(() => {
                                // Start horizontal slide left to centre
                                const slideDuration = (branchX / horizSpeed) * 1000;
                                new TWEEN.Tween(resultVec.group.position)
                                    .to({ x: 0 }, slideDuration)
                                    .easing(TWEEN.Easing.Quadratic.InOut)
                                    .onComplete(() => {
                                        // Trigger merge addition once at centre
                                        if (!lane.mergeStarted) {
                                            lane.mergeStarted = true;
                                            startAdditionAnimation(originalVec, resultVec, () => {
                                                resultVec.group.visible = false;
                                                lane.horizPhase = 'merged';
                                            });
                                        }
                                    })
                                    .start();
                            })
                            .start();
                    }
                    break;
                }
                case 'moveLeft': {
                    if (!lane.mergeStarted && lane.resultVec && lane.resultVec.group.position.x <= 0.01) {
                        lane.mergeStarted = true;
                        startAdditionAnimation(lane.resultVec, originalVec, () => {
                            // After merge hide resultVec
                            if (lane.resultVec) {
                                lane.resultVec.group.visible = false;
                            }
                            lane.horizPhase = 'merged';
                        });
                    }
                    break;
                }
                case 'merged':
                default:
                    break;
            }

            // Determine which branched object is visible for trail
            let branchFollower = null;
            if (lane.resultVec && lane.resultVec.group.visible) branchFollower = lane.resultVec.group;
            else if (lane.movingVec.group.visible) branchFollower = lane.movingVec.group;
            if (branchFollower) {
                updateTrail(lane.branchTrail, branchFollower.position);
            }
        });

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
            l.addVec.dispose();
            if (l.resultVec) l.resultVec.dispose();
        });
        layerNorm.dispose && layerNorm.dispose();
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