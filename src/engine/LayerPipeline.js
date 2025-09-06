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
    PRISM_ADD_ANIM_BASE_DURATION,
    PRISM_ADD_ANIM_BASE_FLASH_DURATION,
    PRISM_ADD_ANIM_BASE_DELAY_BETWEEN_PRISMS,
    PRISM_ADD_ANIM_SPEED_MULT,
    VECTOR_LENGTH_PRISM
} from '../utils/constants.js';
import { VectorVisualizationInstancedPrism } from '../components/VectorVisualizationInstancedPrism.js';
import { startPrismAdditionAnimation } from '../utils/additionUtils.js';

function simplePrismMultiply(srcVec, tgtVec, onComplete) {
    for (let i = 0; i < VECTOR_LENGTH_PRISM; i++) {
        tgtVec.rawData[i] = (srcVec.rawData[i] || 0) * (tgtVec.rawData[i] || 0);
    }
    tgtVec.updateKeyColorsFromData(tgtVec.rawData, 30);
    if (onComplete) onComplete();
}

/**
 * LayerPipeline orchestrates a single bundle of vectors ("lanes") through an
 * arbitrary stack of GPT-2 transformer layers.  Unlike the old approach that
 * created a fresh bundle for every layer at T=0, this controller initialises
 * ONE set of vectors, then – once a layer finishes – hands those same Three
 * objects off to the next Gpt2Layer positioned above the previous.
 */
export class LayerPipeline {
    /**
     * @param {HTMLCanvasElement} canvas – Render target for the CoreEngine.
     * @param {number}           numLayers – Total layers in the stack.
     * @param {Object}           [opts] – Additional CoreEngine options (camera, speed, etc.).
     * @param {() => any}        [opts.randomFactory] – Factory that produces a fresh random-source for each layer.
     */
    constructor(canvas, numLayers = 12, opts = {}) {
        if (!canvas) throw new Error('LayerPipeline requires a renderCanvas element');
        this._numLayers = Math.max(1, numLayers);
        this._canvas    = canvas;
        this._opts      = opts;
        this._randFactory = typeof opts.randomFactory === 'function' ? opts.randomFactory : createRandomSource;

        this._layers = [];
        this._currentLayerIdx = 0;

        // ------------------------------------------------------------------
        // Pre-create *all* layers so their static visuals are visible upfront.
        // Only the first layer is active immediately; higher layers remain
        // dormant until their turn, at which point we inject residual lanes.
        // ------------------------------------------------------------------

        const engineOpts = { ...opts };
        this._engine = new CoreEngine(canvas, [], engineOpts);

        for (let i = 0; i < this._numLayers; i++) {
            const rand = this._randFactory();
            const isActive = i === 0; // only first layer active initially
            const layer = new Gpt2Layer(i, rand, 0, /*externalLanes*/ null, /*onFinished*/ null, isActive);

            // Assign onFinished callback for chaining once layer becomes active
            layer.setOnFinished(() => this._advanceToNextLayer());

            layer.init(this._engine.scene);
            this._layers.push(layer);
            this._engine._layers.push(layer); // add to engine update list
        }

        // Ensure first layer has active callback wired before start
        this._layers[0].setOnFinished(() => this._advanceToNextLayer());
    }

    /** Dispose and tear down Three resources */
    dispose() { this._engine && this._engine.dispose(); }

    /** Return reference to internal CoreEngine (for advanced use-cases). */
    get engine() { return this._engine; }

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

        // Compute the target Y (in the LAST LAYER'S LOCAL SPACE) corresponding to
        // the entrance of the flipped top vocab embedding matrix.  Using LOCAL
        // coordinates avoids mixing world-space values with the lane vectors'
        // local positions.
        if (!lastLayer.mlpDown || !lastLayer.mlpDown.group) return;
        // Prefer using the ACTUAL top embedding object in the scene if present
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
            // Fallback: compute via relative geometry positions (local space)
            const towerTopYLocal = lastLayer.mlpDown.group.position.y + MLP_MATRIX_PARAMS_DOWN.height / 2;
            const topVocabCenterYLocal = towerTopYLocal + TOP_EMBED_Y_GAP_ABOVE_TOWER + EMBEDDING_MATRIX_PARAMS_VOCAB.height / 2 + TOP_EMBED_Y_ADJUST;
            targetYLocal = topVocabCenterYLocal - EMBEDDING_MATRIX_PARAMS_VOCAB.height / 2 + 5;
        }
        // Clamp MHSA's continuous residual rise target so vectors stop exactly at the
        // entrance of the top embedding instead of drifting past it.
        try {
            if (lastLayer.mhsaAnimation) {
                lastLayer.mhsaAnimation.finalOriginalY = targetYLocal;
                lastLayer.mhsaAnimation.topEmbeddingStopY = targetYLocal;
                lastLayer.mhsaAnimation.postSplitRiseSpeed = ANIM_RISE_SPEED_ORIGINAL;
            }
            lastLayer.__topEmbedStopYLocal = targetYLocal;
        } catch (_) { /* no-op */ }

        // Locate the top LayerNorm, if present
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

        // If a top LayerNorm exists, run a mini LayerNorm pipeline before entering the vocab embedding
        if (lnTopGroup) {
            const lnCenterWorld = new THREE.Vector3();
            lnTopGroup.getWorldPosition(lnCenterWorld);
            const lnCenterLocal = lnCenterWorld.clone();
            lastLayer.root.worldToLocal(lnCenterLocal);
            const lnCenterY = lnCenterLocal.y;
            const lnBottomY = lnCenterY - LN_PARAMS.height / 2;

            const activateLnColor = () => {
                const white = new THREE.Color(0xffffff);
                lnTopGroup.traverse(obj => {
                    if (obj.isMesh && obj.material) {
                        const apply = mat => { mat.color.copy(white); mat.emissive.copy(white); mat.emissiveIntensity = 0.5; mat.transparent = false; mat.opacity = 1.0; };
                        if (Array.isArray(obj.material)) obj.material.forEach(apply); else apply(obj.material);
                    }
                });
            };

            lastLayer.lanes.forEach(lane => {
                const vec = lane && lane.originalVec;
                if (!vec || !vec.group) return;
                const startY = vec.group.position.y;
                if (typeof startY !== 'number' || !isFinite(startY)) return;

                const zPos = lane.zPos || 0;
                const multVec = new VectorVisualizationInstancedPrism(vec.rawData.slice(), new THREE.Vector3(0, lnCenterY, zPos));
                lastLayer.root.add(multVec.group);
                multVec.group.visible = false;
                const addVec = new VectorVisualizationInstancedPrism(vec.rawData.slice(), new THREE.Vector3(0, lnCenterY + LN_PARAMS.height / 4, zPos));
                lastLayer.root.add(addVec.group);
                addVec.group.visible = false;

                const distToCenter = Math.max(0, lnCenterY - startY);
                const durToCenter = (distToCenter / (ANIM_RISE_SPEED_ORIGINAL * GLOBAL_ANIM_SPEED_MULT)) * 1000;

                new TWEEN.Tween(vec.group.position)
                    .to({ y: lnCenterY }, Math.max(100, durToCenter))
                    .easing(TWEEN.Easing.Quadratic.InOut)
                    .onUpdate(() => {
                        if (!lane.__topLnEntered && vec.group.position.y >= lnBottomY) {
                            lane.__topLnEntered = true;
                            multVec.group.visible = true;
                            addVec.group.visible = true;
                            activateLnColor();
                        }
                    })
                    .onComplete(() => {
                        simplePrismMultiply(vec, multVec, () => {
                            vec.group.visible = false;
                            multVec.group.visible = false;

                            const resVec = new VectorVisualizationInstancedPrism(multVec.rawData.slice(), multVec.group.position.clone());
                            lastLayer.root.add(resVec.group);

                            const addDur = (PRISM_ADD_ANIM_BASE_DURATION + PRISM_ADD_ANIM_BASE_FLASH_DURATION + VECTOR_LENGTH_PRISM * PRISM_ADD_ANIM_BASE_DELAY_BETWEEN_PRISMS) / PRISM_ADD_ANIM_SPEED_MULT;
                            startPrismAdditionAnimation(addVec, resVec);

                            setTimeout(() => {
                                const riseDist = Math.max(0, targetYLocal - resVec.group.position.y);
                                const durMs = (riseDist / (ANIM_RISE_SPEED_ORIGINAL * GLOBAL_ANIM_SPEED_MULT)) * 1000;
                                new TWEEN.Tween(resVec.group.position)
                                    .to({ y: targetYLocal }, Math.max(100, durMs))
                                    .easing(TWEEN.Easing.Quadratic.InOut)
                                    .start();
                            }, addDur + 100);
                        });
                    })
                    .start();
            });
            return;
        }

        // Fallback: no top LayerNorm – simply rise into the embedding
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
                .start();
        });
    }
}

/** Convenience helper mirroring CoreEngine.startEngine signature */
export function startPipeline(canvas, numLayers = 12, opts = {}) {
    const pipeline = new LayerPipeline(canvas, numLayers, opts);
    return () => pipeline.dispose();
} 