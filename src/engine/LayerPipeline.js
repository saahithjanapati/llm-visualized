import * as THREE from 'three';
import { CoreEngine } from './CoreEngine.js';
import Gpt2Layer from './layers/Gpt2Layer.js';
import { createRandomSource } from '../data/RandomActivationSource.js';
import {
    MLP_MATRIX_PARAMS_DOWN,
    EMBEDDING_MATRIX_PARAMS_VOCAB,
    TOP_EMBED_Y_GAP_ABOVE_TOWER,
    TOP_EMBED_Y_ADJUST,
    GLOBAL_ANIM_SPEED_MULT,
    ANIM_RISE_SPEED_ORIGINAL,
    LN_PARAMS,
    LN_NORM_START_FRACTION_FROM_BOTTOM,
    PRISM_ADD_ANIM_BASE_DURATION,
    PRISM_ADD_ANIM_BASE_FLASH_DURATION,
    PRISM_ADD_ANIM_BASE_DELAY_BETWEEN_PRISMS,
    PRISM_ADD_ANIM_SPEED_MULT,
    VECTOR_LENGTH_PRISM
} from '../utils/constants.js';
import { VectorVisualizationInstancedPrism } from '../components/VectorVisualizationInstancedPrism.js';
import { startPrismAdditionAnimation } from '../utils/additionUtils.js';
import { PrismLayerNormAnimation } from '../animations/PrismLayerNormAnimation.js';

function simplePrismMultiply(srcVec, tgtVec, onComplete) {
    for (let i = 0; i < VECTOR_LENGTH_PRISM; i++) {
        tgtVec.rawData[i] = (srcVec.rawData[i] || 0) * (tgtVec.rawData[i] || 0);
    }
    tgtVec.updateKeyColorsFromData(tgtVec.rawData, 30);
    if (onComplete) onComplete();
}

const COLOR_DARK_GRAY = new THREE.Color(0x333333);
const COLOR_LIGHT_YELLOW = new THREE.Color(0xffffff);
const COLOR_BRIGHT_YELLOW = new THREE.Color(0xffffff);

/**
 * LayerPipeline orchestrates a single bundle of vectors ("lanes") through an
 * arbitrary stack of GPT-2 transformer layers.  Unlike the old approach that
 * created a fresh bundle for every layer at T=0, this controller initialises
 * ONE set of vectors, then – once a layer finishes – hands those same Three
 * objects off to the next Gpt2Layer positioned above the previous.
 */
export class LayerPipeline extends EventTarget {
    /**
     * @param {HTMLCanvasElement} canvas – Render target for the CoreEngine.
     * @param {number}           numLayers – Total layers in the stack.
     * @param {Object}           [opts] – Additional CoreEngine options (camera, speed, etc.).
     * @param {() => any}        [opts.randomFactory] – Factory that produces a fresh random-source for each layer.
     */
    constructor(canvas, numLayers = 12, opts = {}) {
        super();
        if (!canvas) throw new Error('LayerPipeline requires a renderCanvas element');
        this._numLayers = Math.max(1, numLayers);
        this._canvas    = canvas;
        this._opts      = opts;
        this._randFactory = typeof opts.randomFactory === 'function' ? opts.randomFactory : createRandomSource;

        this._layers = [];
        this._currentLayerIdx = 0;

        this._autoCameraFollow = opts.autoCameraFollow !== false;
        this._autoCameraLerp = (typeof opts.autoCameraLerp === 'number' && Number.isFinite(opts.autoCameraLerp))
            ? THREE.MathUtils.clamp(opts.autoCameraLerp, 0.01, 1)
            : 0.12;
        // Bias the automatic camera focus so it hugs the left edge of the tower and
        // looks slightly above the geometric centre.  These defaults were tuned to
        // align the frame with the MLP matrices while letting the vertical motion
        // feel more like a continuous rise instead of a series of jumps.
        this._autoCameraHorizontalBias = (typeof opts.autoCameraHorizontalBias === 'number')
            ? THREE.MathUtils.clamp(opts.autoCameraHorizontalBias, 0, 1)
            : 0.18;
        this._autoCameraVerticalBias = (typeof opts.autoCameraVerticalBias === 'number')
            ? THREE.MathUtils.clamp(opts.autoCameraVerticalBias, 0, 1)
            : 0.62;
        this._autoCameraMaxDownStep = (typeof opts.autoCameraMaxDownStep === 'number' && Number.isFinite(opts.autoCameraMaxDownStep))
            ? Math.max(0, opts.autoCameraMaxDownStep)
            : 45;
        this._autoCameraLastUpdate = 0;
        this._autoCameraMinIntervalMs = 80;
        this._autoCameraBox = new THREE.Box3();
        this._autoCameraCenter = new THREE.Vector3();
        this._autoCameraTargetScratch = new THREE.Vector3();
        this._autoCameraOffsetScratch = new THREE.Vector3();
        this._autoCameraDesiredTargetScratch = new THREE.Vector3();
        this._autoCameraDesiredCameraPosScratch = new THREE.Vector3();
        this._autoCameraSmoothedTarget = new THREE.Vector3();
        this._autoCameraSmoothedCameraPos = new THREE.Vector3();
        this._autoCameraHasSmoothedState = false;
        this._onAutoCameraProgress = () => { this._maybeAutoCameraFocus(); };
        this.addEventListener('progress', this._onAutoCameraProgress);

        // ------------------------------------------------------------------
        // Pre-create *all* layers so their static visuals are visible upfront.
        // Only the first layer is active immediately; higher layers remain
        // dormant until their turn, at which point we inject residual lanes.
        // ------------------------------------------------------------------

        const engineOpts = { ...opts };
        if (typeof engineOpts.cameraFarMargin !== 'number') {
            const DEFAULT_CAMERA_FAR_MARGIN = 40000;
            // Provide additional depth so tall transformer stacks remain visible
            // when the user zooms far away from the tower. The allowance scales
            // with layer count but never drops below the default margin.
            const approxTowerAllowance = Math.max(DEFAULT_CAMERA_FAR_MARGIN, this._numLayers * 1800);
            engineOpts.cameraFarMargin = approxTowerAllowance;
        }
        this._engine = new CoreEngine(canvas, [], engineOpts);

        for (let i = 0; i < this._numLayers; i++) {
            const rand = this._randFactory();
            const isActive = i === 0; // only first layer active initially
            const layer = new Gpt2Layer(i, rand, 0, /*externalLanes*/ null, /*onFinished*/ null, isActive);

            // Assign onFinished callback for chaining once layer becomes active
            layer.setOnFinished(() => this._advanceToNextLayer());
            layer.setProgressEmitter(this);

            layer.init(this._engine.scene);
            if (typeof this._engine.registerRaycastRoot === 'function') {
                this._engine.registerRaycastRoot(layer.root);
            }
            this._layers.push(layer);
            this._engine._layers.push(layer); // add to engine update list
        }

        // Ensure first layer has active callback wired before start
        this._layers[0].setOnFinished(() => this._advanceToNextLayer());
        this._layers[0].setProgressEmitter(this);

        this._maybeAutoCameraFocus({ immediate: true });
    }

    /** Dispose and tear down Three resources */
    dispose() {
        if (this._onAutoCameraProgress) {
            this.removeEventListener('progress', this._onAutoCameraProgress);
            this._onAutoCameraProgress = null;
        }
        if (this._engine) {
            this._engine.dispose();
        }
    }

    /** Return reference to internal CoreEngine (for advanced use-cases). */
    get engine() { return this._engine; }

    /** Enable or disable automatic camera tracking of the active layer. */
    setAutoCameraFollow(enabled, { immediate = false } = {}) {
        const nextValue = !!enabled;
        if (nextValue === this._autoCameraFollow) {
            if (nextValue && immediate) {
                this._maybeAutoCameraFocus({ immediate: true });
            }
            return;
        }
        this._autoCameraFollow = nextValue;
        this._autoCameraHasSmoothedState = false;
        if (this._autoCameraFollow) {
            this._maybeAutoCameraFocus({ immediate: true });
        }
    }

    /** Check whether automatic camera tracking is enabled. */
    isAutoCameraFollowEnabled() {
        return !!this._autoCameraFollow;
    }

    // ----------------------------------------------------------------------
    // Private helpers
    // ----------------------------------------------------------------------

    /**
     * Called when the currently active layer reports completion via its
     * `onFinished` callback.  This creates the next Gpt2Layer, injects the
     * existing lane bundle and registers its own completion hook.
     */
    _advanceToNextLayer() {
        this._currentLayerIdx += 1;
        this.dispatchEvent(new Event('progress'));
        if (this._currentLayerIdx >= this._numLayers) {
            // All layers processed – trigger final rise into top embedding
            try { this._animateRiseIntoTopEmbedding(); } catch (_) { /* optional */ }
            return;
        }

        // Grab lanes from the previous (just-completed) layer
        const prevLayer = this._layers[this._currentLayerIdx - 1];
        // Deactivate the previous layer to stop it updating the shared vectors
        if (prevLayer) {
            prevLayer.isActive = false;
        }
        const externalLanes = prevLayer.lanes;
        // Ensure residual trails remain continuous by reparenting any
        // world-space trails to the new engine scene (safety no-op if same).
        if (externalLanes && externalLanes.length) {
            externalLanes.forEach(lane => {
                // Prefer the dedicated world-space residual trail carried across lanes
                const trailRef = (lane && lane.originalTrail)
                    || (lane && lane.originalVec && lane.originalVec.userData && lane.originalVec.userData.trail);
                if (trailRef && typeof trailRef.reparent === 'function') {
                    trailRef.reparent(this._engine.scene);
                }
            });
        }

        const nextLayer = this._layers[this._currentLayerIdx];
        if (!nextLayer) return;

        nextLayer.activateWithLanes(externalLanes);

        this._maybeAutoCameraFocus({ immediate: true });

        // Now that the original residual vectors have been transferred, we can safely
        // hide the remaining heavy geometry in the previous layer to save GPU work.
        if (prevLayer && typeof prevLayer.hideDynamicGeometry === 'function') {
            prevLayer.hideDynamicGeometry();
        }
    }

    /**
     * After the last layer completes, raise residual vectors up into the top
     * vocabulary embedding position using the same placement logic as the test page.
     */
    _animateRiseIntoTopEmbedding() {
        const lastLayer = this._layers[this._numLayers - 1];
        if (!lastLayer || !Array.isArray(lastLayer.lanes) || !lastLayer.lanes.length) return;

        if (!lastLayer.mlpDown || !lastLayer.mlpDown.group) return;

        const targetYLocal = this._calculateTopEmbeddingTargetY(lastLayer);

        const lnInfo = this._findTopLayerNorm(lastLayer);

        this._animateResidualVectors(lastLayer, targetYLocal, lnInfo);
    }

    /**
     * Determine the target local Y position for residual vectors entering the
     * top vocabulary embedding and update MHSA animation boundaries.
     * @param {Gpt2Layer} lastLayer
     * @returns {number} Local-space Y coordinate where vectors should stop.
     */
    _calculateTopEmbeddingTargetY(lastLayer) {
        let targetYLocal = null;
        try {
            const scene = this._engine && this._engine.scene;
            let topEmbedObj = null;
            if (scene && typeof scene.traverse === 'function') {
                scene.traverse((obj) => {
                    if (topEmbedObj) return;
                    if (obj && obj.userData && obj.userData.label === 'Vocab Embedding (Top)') {
                        topEmbedObj = obj;
                    }
                });
            }
            if (topEmbedObj) {
                const centerWorld = new THREE.Vector3();
                topEmbedObj.getWorldPosition(centerWorld);
                const stopWorldY = centerWorld.y - EMBEDDING_MATRIX_PARAMS_VOCAB.height / 2 + 5;
                const localVec = new THREE.Vector3(0, stopWorldY, 0);
                lastLayer.root.worldToLocal(localVec);
                targetYLocal = localVec.y;
            }
        } catch (_) { /* fallback to formula below */ }

        if (targetYLocal == null) {
            const towerTopYLocal = lastLayer.mlpDown.group.position.y + MLP_MATRIX_PARAMS_DOWN.height / 2;
            const topVocabCenterYLocal = towerTopYLocal + TOP_EMBED_Y_GAP_ABOVE_TOWER + EMBEDDING_MATRIX_PARAMS_VOCAB.height / 2 + TOP_EMBED_Y_ADJUST;
            targetYLocal = topVocabCenterYLocal - EMBEDDING_MATRIX_PARAMS_VOCAB.height / 2 + 5;
        }

        try {
            if (lastLayer.mhsaAnimation) {
                lastLayer.mhsaAnimation.finalOriginalY = targetYLocal;
                lastLayer.mhsaAnimation.topEmbeddingStopY = targetYLocal;
                lastLayer.mhsaAnimation.postSplitRiseSpeed = ANIM_RISE_SPEED_ORIGINAL;
            }
            lastLayer.__topEmbedStopYLocal = targetYLocal;
        } catch (_) { /* no-op */ }

        return targetYLocal;
    }

    /**
     * Locate the optional top LayerNorm in the scene and compute useful
     * positional data for animation.
     * @param {Gpt2Layer} lastLayer
     * @returns {{lnTopGroup: THREE.Object3D, lnCenterY: number, lnBottomY: number}|null}
     *          LayerNorm group and position info if found.
     */
    _findTopLayerNorm(lastLayer) {
        let lnTopGroup = null;
        try {
            const scene = this._engine && this._engine.scene;
            if (scene && typeof scene.traverse === 'function') {
                scene.traverse(obj => {
                    if (!lnTopGroup && obj && obj.userData && obj.userData.label === 'LayerNorm (Top)') {
                        lnTopGroup = obj;
                    }
                });
            }
        } catch (_) { /* optional */ }

        if (!lnTopGroup) return null;

        const lnCenterWorld = new THREE.Vector3();
        lnTopGroup.getWorldPosition(lnCenterWorld);
        const lnCenterLocal = lnCenterWorld.clone();
        lastLayer.root.worldToLocal(lnCenterLocal);
        const lnCenterY = lnCenterLocal.y;
        const lnBottomY = lnCenterY - LN_PARAMS.height / 2;

        return { lnTopGroup, lnCenterY, lnBottomY };
    }

    /**
     * Apply the bright activated appearance to a LayerNorm group.
     * @param {THREE.Object3D} lnTopGroup
     */
    _activateLayerNormColor(lnTopGroup) {
        const white = new THREE.Color(0xffffff);
        lnTopGroup.traverse(obj => {
            if (obj.isMesh && obj.material) {
                const apply = mat => { mat.color.copy(white); mat.emissive.copy(white); mat.emissiveIntensity = 0.5; mat.transparent = false; mat.opacity = 1.0; };
                if (Array.isArray(obj.material)) obj.material.forEach(apply); else apply(obj.material);
            }
        });
    }

    /**
     * Animate residual vectors toward the vocab embedding, optionally passing
     * through the top LayerNorm pipeline when present.
     * @param {Gpt2Layer} lastLayer
     * @param {number} targetYLocal
     * @param {{lnTopGroup: THREE.Object3D, lnCenterY: number, lnBottomY: number}|null} lnInfo
     */
    _animateResidualVectors(lastLayer, targetYLocal, lnInfo) {
        if (lnInfo && lnInfo.lnTopGroup) {
            const { lnTopGroup, lnCenterY, lnBottomY } = lnInfo;
            const lnMeshes = [];
            lnTopGroup.traverse(obj => {
                if (obj && obj.isMesh && obj.material) {
                    lnMeshes.push(obj);
                }
            });

            const lnColorState = {
                highestY: -Infinity,
                locked: false,
                lockedColor: new THREE.Color(COLOR_BRIGHT_YELLOW),
                currentColor: new THREE.Color(COLOR_DARK_GRAY),
                currentOpacity: 1.0
            };
            const tempColor = new THREE.Color();
            const applyTopLnColor = () => {
                lnMeshes.forEach(mesh => {
                    const applyMaterial = mat => {
                        if (!mat) return;
                        if (mat.color) mat.color.copy(lnColorState.currentColor);
                        if (mat.emissive) mat.emissive.copy(lnColorState.currentColor);
                        mat.transparent = lnColorState.currentOpacity < 1.0;
                        mat.opacity = lnColorState.currentOpacity;
                        mat.needsUpdate = true;
                    };
                    if (Array.isArray(mesh.material)) {
                        mesh.material.forEach(applyMaterial);
                    } else {
                        applyMaterial(mesh.material);
                    }
                });
            };
            applyTopLnColor();

            const lnHeight = LN_PARAMS.height || 0;
            const lnTopY = lnCenterY + lnHeight / 2;
            const normStartY = lnBottomY + lnHeight * LN_NORM_START_FRACTION_FROM_BOTTOM;
            const exitTransitionRange = 5;
            const tmpWorldPos = new THREE.Vector3();
            const tmpLocalPos = new THREE.Vector3();

            const updateTopLnColor = (y) => {
                if (!Number.isFinite(y)) return;
                if (y > lnColorState.highestY) {
                    lnColorState.highestY = y;
                }

                const highest = lnColorState.highestY;
                if (lnColorState.locked) {
                    lnColorState.currentColor.copy(lnColorState.lockedColor);
                    lnColorState.currentOpacity = 1.0;
                } else {
                    if (highest >= lnBottomY && highest < lnCenterY) {
                        const denom = Math.max(lnCenterY - lnBottomY, 1e-6);
                        const t = (highest - lnBottomY) / denom;
                        tempColor.copy(COLOR_DARK_GRAY).lerp(COLOR_LIGHT_YELLOW, t);
                        lnColorState.currentColor.copy(tempColor);
                        lnColorState.currentOpacity = THREE.MathUtils.lerp(1.0, 0.6, t);
                    } else if (highest >= lnCenterY && highest < lnTopY) {
                        lnColorState.currentColor.copy(COLOR_LIGHT_YELLOW);
                        lnColorState.currentOpacity = 0.6;
                    } else if (highest >= lnTopY) {
                        const tRaw = (highest - lnTopY) / exitTransitionRange;
                        const t = Math.min(1, Math.max(0, tRaw));
                        tempColor.copy(COLOR_LIGHT_YELLOW).lerp(COLOR_BRIGHT_YELLOW, t);
                        lnColorState.currentColor.copy(tempColor);
                        lnColorState.currentOpacity = THREE.MathUtils.lerp(0.6, 1.0, t);
                    } else {
                        lnColorState.currentColor.copy(COLOR_DARK_GRAY);
                        lnColorState.currentOpacity = 1.0;
                    }

                    if (highest >= lnTopY + exitTransitionRange) {
                        lnColorState.locked = true;
                        lnColorState.lockedColor.copy(COLOR_BRIGHT_YELLOW);
                        lnColorState.currentColor.copy(lnColorState.lockedColor);
                        lnColorState.currentOpacity = 1.0;
                    }
                }

                applyTopLnColor();
            };

            const additionDuration = (PRISM_ADD_ANIM_BASE_DURATION + PRISM_ADD_ANIM_BASE_FLASH_DURATION + VECTOR_LENGTH_PRISM * PRISM_ADD_ANIM_BASE_DELAY_BETWEEN_PRISMS) / PRISM_ADD_ANIM_SPEED_MULT;

            const updateTrailPosition = (vector) => {
                if (!vector || !vector.userData || !vector.userData.trail) return;
                const trail = vector.userData.trail;
                if (typeof trail.update !== 'function') return;
                vector.group.getWorldPosition(tmpWorldPos);
                if (vector.userData.trailWorld) {
                    tmpLocalPos.copy(tmpWorldPos);
                    trail.update(tmpLocalPos);
                } else {
                    tmpLocalPos.copy(tmpWorldPos);
                    try {
                        const parentObject = (trail._line && trail._line.parent) || trail._scene || null;
                        if (parentObject && typeof parentObject.worldToLocal === 'function') {
                            parentObject.worldToLocal(tmpLocalPos);
                        }
                    } catch (_) {
                        // fall back to world position already copied into tmpLocalPos
                    }
                    trail.update(tmpLocalPos);
                }
            };

            lastLayer.lanes.forEach(lane => {
                const vec = lane && lane.originalVec;
                if (!vec || !vec.group) return;
                const startY = vec.group.position.y;
                if (!Number.isFinite(startY)) return;

                const zPos = lane.zPos || 0;
                const multVec = new VectorVisualizationInstancedPrism(vec.rawData.slice(), new THREE.Vector3(0, lnCenterY, zPos));
                lastLayer.root.add(multVec.group);
                multVec.group.visible = false;

                const addVec = new VectorVisualizationInstancedPrism(vec.rawData.slice(), new THREE.Vector3(0, lnCenterY + LN_PARAMS.height / 4, zPos));
                lastLayer.root.add(addVec.group);
                addVec.group.visible = false;

                const normAnim = new PrismLayerNormAnimation(vec);
                let normLoopActive = false;

                if (startY >= lnBottomY) {
                    multVec.group.visible = true;
                    addVec.group.visible = true;
                    lane.__topLnEntered = true;
                }

                const startFinalRise = (resVec) => {
                    const riseDist = Math.max(0, targetYLocal - resVec.group.position.y);
                    const durMs = (riseDist / (ANIM_RISE_SPEED_ORIGINAL * GLOBAL_ANIM_SPEED_MULT)) * 1000;
                    new TWEEN.Tween(resVec.group.position)
                        .to({ y: targetYLocal }, Math.max(100, durMs))
                        .easing(TWEEN.Easing.Quadratic.InOut)
                        .onUpdate(() => {
                            updateTopLnColor(resVec.group.position.y);
                            updateTrailPosition(resVec);
                            this.dispatchEvent(new Event('progress'));
                        })
                        .onComplete(() => {
                            updateTopLnColor(targetYLocal + exitTransitionRange);
                            updateTrailPosition(resVec);
                            this.dispatchEvent(new Event('progress'));
                        })
                        .start();
                };

                const beginMultiply = () => {
                    if (lane.__topLnMultStarted) return;
                    lane.__topLnMultStarted = true;
                    multVec.group.visible = true;
                    addVec.group.visible = true;

                    simplePrismMultiply(vec, multVec, () => {
                        updateTopLnColor(multVec.group.position.y);
                        vec.group.visible = false;
                        multVec.group.visible = false;

                        const resVec = new VectorVisualizationInstancedPrism(multVec.rawData.slice(), multVec.group.position.clone());
                        lastLayer.root.add(resVec.group);

                        if (multVec.group && multVec.group.parent) {
                            multVec.group.parent.remove(multVec.group);
                        }

                        resVec.userData = resVec.userData || {};
                        if (vec.userData && vec.userData.trail) {
                            resVec.userData.trail = vec.userData.trail;
                            resVec.userData.trailWorld = vec.userData.trailWorld;
                        }
                        lane.originalVec = resVec;

                        updateTopLnColor(resVec.group.position.y);
                        this.dispatchEvent(new Event('progress'));

                        startPrismAdditionAnimation(addVec, resVec, null, () => {
                            if (addVec.group) addVec.group.visible = false;
                            if (addVec.group && addVec.group.parent) {
                                addVec.group.parent.remove(addVec.group);
                            }
                        });

                        new TWEEN.Tween({ t: 0 })
                            .to({ t: 1 }, additionDuration)
                            .onUpdate(() => {
                                updateTopLnColor(resVec.group.position.y);
                                updateTrailPosition(resVec);
                                this.dispatchEvent(new Event('progress'));
                            })
                            .start();

                        setTimeout(() => {
                            updateTopLnColor(resVec.group.position.y);
                            startFinalRise(resVec);
                        }, additionDuration + 100);
                    });
                };

                const riseToCenter = () => {
                    const targetY = Math.max(vec.group.position.y, lnCenterY);
                    if (vec.group.position.y >= targetY - 0.01) {
                        vec.group.position.y = targetY;
                        beginMultiply();
                        return;
                    }

                    const distance = Math.max(0, targetY - vec.group.position.y);
                    const duration = (distance / (ANIM_RISE_SPEED_ORIGINAL * GLOBAL_ANIM_SPEED_MULT)) * 1000;
                    new TWEEN.Tween(vec.group.position)
                        .to({ y: targetY }, Math.max(100, duration))
                        .easing(TWEEN.Easing.Quadratic.InOut)
                        .onUpdate(() => {
                            updateTopLnColor(vec.group.position.y);
                            this.dispatchEvent(new Event('progress'));
                        })
                        .onComplete(() => {
                            updateTopLnColor(vec.group.position.y);
                            this.dispatchEvent(new Event('progress'));
                            beginMultiply();
                        })
                        .start();
                };

                const startNormalization = () => {
                    if (normLoopActive) return;
                    normLoopActive = true;
                    try {
                        normAnim.start(vec.rawData.slice());
                    } catch (_) {
                        normLoopActive = false;
                        riseToCenter();
                        return;
                    }

                    const runLoop = () => {
                        normAnim.update(0);
                        updateTopLnColor(vec.group.position.y);
                        this.dispatchEvent(new Event('progress'));
                        if (normAnim.isAnimating) {
                            requestAnimationFrame(runLoop);
                        } else {
                            normLoopActive = false;
                            riseToCenter();
                        }
                    };
                    runLoop();
                };

                const moveToNormStart = () => {
                    updateTopLnColor(vec.group.position.y);
                    const stageTarget = Math.max(vec.group.position.y, normStartY);

                    if (vec.group.position.y >= stageTarget - 0.01) {
                        vec.group.position.y = stageTarget;
                        startNormalization();
                        return;
                    }

                    const distance = Math.max(0, stageTarget - vec.group.position.y);
                    const duration = (distance / (ANIM_RISE_SPEED_ORIGINAL * GLOBAL_ANIM_SPEED_MULT)) * 1000;
                    new TWEEN.Tween(vec.group.position)
                        .to({ y: stageTarget }, Math.max(100, duration))
                        .easing(TWEEN.Easing.Quadratic.InOut)
                        .onUpdate(() => {
                            updateTopLnColor(vec.group.position.y);
                            if (!lane.__topLnEntered && vec.group.position.y >= lnBottomY) {
                                lane.__topLnEntered = true;
                                multVec.group.visible = true;
                                addVec.group.visible = true;
                            }
                            this.dispatchEvent(new Event('progress'));
                        })
                        .onComplete(() => {
                            updateTopLnColor(vec.group.position.y);
                            this.dispatchEvent(new Event('progress'));
                            startNormalization();
                        })
                        .start();
                };

                moveToNormStart();
            });

            return;
        }

        lastLayer.lanes.forEach(lane => {
            const vec = lane && lane.originalVec;
            if (!vec || !vec.group) return;
            const startY = vec.group.position.y;
            if (typeof startY !== 'number' || !isFinite(startY)) return;
            if (startY >= targetYLocal - 0.01) return;

            const riseDist = Math.max(0, targetYLocal - startY);
            const durMs = (riseDist / (ANIM_RISE_SPEED_ORIGINAL * GLOBAL_ANIM_SPEED_MULT)) * 1000;

            new TWEEN.Tween(vec.group.position)
                .to({ y: targetYLocal }, Math.max(100, durMs))
                .easing(TWEEN.Easing.Quadratic.InOut)
                .onUpdate(() => this.dispatchEvent(new Event('progress')))
                .onComplete(() => this.dispatchEvent(new Event('progress')))
                .start();
        });
    }

    _maybeAutoCameraFocus({ immediate = false } = {}) {
        if (!this._autoCameraFollow) return;
        const engine = this._engine;
        if (!engine || !engine.camera || !engine.controls) return;
        if (typeof engine.isUserNavigating === 'function' && engine.isUserNavigating()) return;

        const now = (typeof performance !== 'undefined' && performance?.now)
            ? performance.now()
            : Date.now();
        if (!immediate && now - this._autoCameraLastUpdate < this._autoCameraMinIntervalMs) return;
        this._autoCameraLastUpdate = now;

        const layerIndex = Math.min(this._currentLayerIdx, this._layers.length - 1);
        const layer = this._layers[layerIndex];
        if (!layer || !layer.root) return;

        const bbox = this._autoCameraBox;
        bbox.makeEmpty();
        bbox.setFromObject(layer.root);
        if (bbox.isEmpty()) return;

        const center = bbox.getCenter(this._autoCameraCenter);
        if (!Number.isFinite(center.x) || !Number.isFinite(center.y) || !Number.isFinite(center.z)) return;

        const controls = engine.controls;
        const camera = engine.camera;
        if (!controls || !camera) return;

        const currentTarget = this._autoCameraTargetScratch.copy(controls.target);
        const offset = this._autoCameraOffsetScratch.copy(camera.position).sub(currentTarget);

        const desiredTarget = this._autoCameraDesiredTargetScratch.set(
            THREE.MathUtils.lerp(bbox.min.x, bbox.max.x, this._autoCameraHorizontalBias),
            THREE.MathUtils.lerp(bbox.min.y, bbox.max.y, this._autoCameraVerticalBias),
            THREE.MathUtils.lerp(bbox.min.z, bbox.max.z, 0.5)
        );

        if (!immediate && this._autoCameraHasSmoothedState) {
            const minY = this._autoCameraSmoothedTarget.y - this._autoCameraMaxDownStep;
            if (desiredTarget.y < minY) {
                desiredTarget.y = minY;
            }
        }

        const desiredCameraPos = this._autoCameraDesiredCameraPosScratch.copy(desiredTarget).add(offset);
        const alpha = immediate ? 1 : this._autoCameraLerp;

        if (!this._autoCameraHasSmoothedState || immediate) {
            this._autoCameraSmoothedTarget.copy(desiredTarget);
            this._autoCameraSmoothedCameraPos.copy(desiredCameraPos);
            this._autoCameraHasSmoothedState = true;
        } else {
            this._autoCameraSmoothedTarget.lerp(desiredTarget, alpha);
            this._autoCameraSmoothedCameraPos.lerp(desiredCameraPos, alpha);
        }

        controls.target.copy(this._autoCameraSmoothedTarget);
        camera.position.copy(this._autoCameraSmoothedCameraPos);

        if (typeof engine.notifyCameraUpdated === 'function') {
            engine.notifyCameraUpdated();
        }
    }
}

/** Convenience helper mirroring CoreEngine.startEngine signature */
export function startPipeline(canvas, numLayers = 12, opts = {}) {
    const pipeline = new LayerPipeline(canvas, numLayers, opts);
    return () => pipeline.dispose();
} 
