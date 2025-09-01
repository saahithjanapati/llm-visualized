import fs from 'fs';
import * as THREE from 'three';
import { Buffer } from 'buffer';
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js';

// ---------------------------------------------------------------------------
// Minimal FileReader polyfill so GLTFExporter works in Node.js.
// ---------------------------------------------------------------------------

if (typeof globalThis.FileReader === 'undefined') {
    class FileReader {
        constructor() {
            this.onloadend = null; // callback
            this.result = null;
        }

        _finish(result) {
            this.result = result;
            if (typeof this.onloadend === 'function') {
                this.onloadend({ target: this });
            }
        }

        readAsArrayBuffer(blob) {
            blob.arrayBuffer().then(buf => this._finish(buf));
        }

        readAsDataURL(blob) {
            blob.arrayBuffer().then(buf => {
                const mime = blob.type || 'application/octet-stream';
                const base64 = Buffer.from(buf).toString('base64');
                this._finish(`data:${mime};base64,${base64}`);
            });
        }
    }

    globalThis.FileReader = FileReader;
}

// ---------------------------------------------------------------------------
// Import component classes and constants – we purposefully **do not** pull the
// global NUM_VECTOR_LANES value because we want a *single*-lane cross-section.
// ---------------------------------------------------------------------------
import { LayerNormalizationVisualization } from '../src/components/LayerNormalizationVisualization.js';
import { WeightMatrixVisualization } from '../src/components/WeightMatrixVisualization.js';
import {
    LN_PARAMS as LN_ALL_PARAMS,
    MHA_MATRIX_PARAMS as MHA_ALL_PARAMS,
    MLP_MATRIX_PARAMS_UP as MLP_UP_ALL,
    MLP_MATRIX_PARAMS_DOWN as MLP_DOWN_ALL,
    EMBEDDING_MATRIX_PARAMS_VOCAB,
    EMBEDDING_MATRIX_PARAMS_POSITION,
    VECTOR_DEPTH_SPACING
} from '../src/utils/constants.js';

import { MHA_OUTPUT_PROJECTION_MATRIX_PARAMS } from '../src/animations/LayerAnimationConstants.js';

// ---------------------------------------------------------------------------
// Helper – replicate cache-key generators from the component code so each
// exported BufferGeometry ends up with the same key the runtime expects.
// ---------------------------------------------------------------------------
function wmKey(p) {
    // `high` = QUALITY_PRESET injected by WeightMatrixVisualization
    return [p.width, p.height, p.depth, p.topWidthFactor, p.cornerRadius, p.numberOfSlits, p.slitWidth, p.slitDepthFactor, p.slitBottomWidthFactor, p.slitTopWidthFactor, 'high'].join('|');
}
function lnKey(p, segments = 64) {
    return [p.width, p.height, p.depth, p.wallThickness, p.numberOfHoles, p.holeWidth, p.holeWidthFactor, segments].join('|');
}

// ---------------------------------------------------------------------------
// Build scene containing exactly one *lane-slice* for every heavy component.
// ---------------------------------------------------------------------------
const scene = new THREE.Scene();

const DEPTH_SLICE = VECTOR_DEPTH_SPACING; // ≈150 – exactly one lane wide

// Shared helper to clone params and override depth/holes.
function withDepth(original, depth = DEPTH_SLICE) {
    return { ...original, depth };
}

// === LayerNorm slice ===
{
    const lnParams = withDepth({
        ...LN_ALL_PARAMS,
        numberOfHoles: 1 // one vertical slit for a single lane
    });

    const vis = new LayerNormalizationVisualization(
        new THREE.Vector3(),
        lnParams.width,
        lnParams.height,
        lnParams.depth,
        lnParams.wallThickness,
        lnParams.numberOfHoles,
        lnParams.holeWidth,
        lnParams.holeWidthFactor
    );
    const key = lnKey(lnParams);
    vis.group.traverse(obj => {
        if (obj.isMesh && obj.geometry) obj.userData.cacheKey = `LN|${key}`;
    });
    scene.add(vis.group);
}

// === MHSA Q/K/V matrix slice ===
{
    const mhaParams = withDepth({ ...MHA_ALL_PARAMS, numberOfSlits: 1 });
    const vis = new WeightMatrixVisualization(null, new THREE.Vector3(0, 0, 2000),
        mhaParams.width,
        mhaParams.height,
        mhaParams.depth,
        mhaParams.topWidthFactor,
        mhaParams.cornerRadius,
        mhaParams.numberOfSlits,
        mhaParams.slitWidth,
        mhaParams.slitDepthFactor,
        mhaParams.slitBottomWidthFactor,
        mhaParams.slitTopWidthFactor);

    const key = wmKey(mhaParams);
    vis.group.traverse(obj => {
        if (obj.isMesh && obj.geometry) obj.userData.cacheKey = `WM|${key}`;
    });
    scene.add(vis.group);
}

// === MLP Up-projection matrix slice ===
{
    const upParams = withDepth({ ...MLP_UP_ALL, numberOfSlits: 1 });
    const vis = new WeightMatrixVisualization(null, new THREE.Vector3(0, 0, 4000),
        upParams.width,
        upParams.height,
        upParams.depth,
        upParams.topWidthFactor,
        upParams.cornerRadius,
        upParams.numberOfSlits,
        upParams.slitWidth,
        upParams.slitDepthFactor,
        upParams.slitBottomWidthFactor,
        upParams.slitTopWidthFactor);
    const key = wmKey(upParams);
    vis.group.traverse(obj => {
        if (obj.isMesh && obj.geometry) obj.userData.cacheKey = `WM|${key}`;
    });
    scene.add(vis.group);
}

// === MLP Down-projection matrix slice ===
{
    const downParams = withDepth({ ...MLP_DOWN_ALL, numberOfSlits: 1 });
    const vis = new WeightMatrixVisualization(null, new THREE.Vector3(0, 0, 6000),
        downParams.width,
        downParams.height,
        downParams.depth,
        downParams.topWidthFactor,
        downParams.cornerRadius,
        downParams.numberOfSlits,
        downParams.slitWidth,
        downParams.slitDepthFactor,
        downParams.slitBottomWidthFactor,
        downParams.slitTopWidthFactor);
    const key = wmKey(downParams);
    vis.group.traverse(obj => {
        if (obj.isMesh && obj.geometry) obj.userData.cacheKey = `WM|${key}`;
    });
    scene.add(vis.group);
}

// === MHSA Output-projection matrix slice ===
{
    const matrixHeight = MHA_ALL_PARAMS.height * MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.heightFactor;
    const outParams = withDepth({
        width: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.width,
        height: matrixHeight,
        depth: DEPTH_SLICE,
        topWidthFactor: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.topWidthFactor,
        cornerRadius: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.cornerRadius,
        numberOfSlits: 1,
        slitWidth: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.slitWidth,
        slitDepthFactor: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.slitDepthFactor,
        slitBottomWidthFactor: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.slitBottomWidthFactor,
        slitTopWidthFactor: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.slitTopWidthFactor
    });

    const vis = new WeightMatrixVisualization(null, new THREE.Vector3(0, 0, 8000),
        outParams.width,
        outParams.height,
        outParams.depth,
        outParams.topWidthFactor,
        outParams.cornerRadius,
        outParams.numberOfSlits,
        outParams.slitWidth,
        outParams.slitDepthFactor,
        outParams.slitBottomWidthFactor,
        outParams.slitTopWidthFactor);

    const key = wmKey(outParams);
    vis.group.traverse(obj => {
        if (obj.isMesh && obj.geometry) obj.userData.cacheKey = `WM|${key}`;
    });
    scene.add(vis.group);
}

// === New: Token/Vocab Embedding slice (bottom wide → top = d_model) ===
{
    const vocabParams = withDepth({ ...EMBEDDING_MATRIX_PARAMS_VOCAB, numberOfSlits: 1 });

    const vis = new WeightMatrixVisualization(null, new THREE.Vector3(0, 0, 10000),
        vocabParams.width,
        vocabParams.height,
        vocabParams.depth,
        vocabParams.topWidthFactor,
        vocabParams.cornerRadius,
        vocabParams.numberOfSlits,
        vocabParams.slitWidth,
        vocabParams.slitDepthFactor,
        vocabParams.slitBottomWidthFactor,
        vocabParams.slitTopWidthFactor);

    const key = wmKey(vocabParams);
    vis.group.traverse(obj => {
        if (obj.isMesh && obj.geometry) obj.userData.cacheKey = `WM|${key}`;
    });
    scene.add(vis.group);
}

// === New: Positional Embedding slice (bottom wide → top = d_model) ===
{
    const posParams = withDepth({ ...EMBEDDING_MATRIX_PARAMS_POSITION, numberOfSlits: 1 });

    const vis = new WeightMatrixVisualization(null, new THREE.Vector3(0, 0, 12000),
        posParams.width,
        posParams.height,
        posParams.depth,
        posParams.topWidthFactor,
        posParams.cornerRadius,
        posParams.numberOfSlits,
        posParams.slitWidth,
        posParams.slitDepthFactor,
        posParams.slitBottomWidthFactor,
        posParams.slitTopWidthFactor);

    const key = wmKey(posParams);
    vis.group.traverse(obj => {
        if (obj.isMesh && obj.geometry) obj.userData.cacheKey = `WM|${key}`;
    });
    scene.add(vis.group);
}

// ---------------------------------------------------------------------------
// Export scene to GLB
// ---------------------------------------------------------------------------
const exporter = new GLTFExporter();

exporter.parse(
    scene,
    (result) => {
        let buffer;
        if (result instanceof ArrayBuffer) buffer = Buffer.from(result);
        else if (ArrayBuffer.isView(result)) buffer = Buffer.from(result.buffer);
        else buffer = Buffer.from(JSON.stringify(result), 'utf8');
        fs.writeFileSync('precomputed_components_slice.glb', buffer);
        console.log('✔  precomputed_components_slice.glb generated');
    },
    (error) => {
        console.error('GLTF export failed:', error);
        process.exit(1);
    },
    { binary: true }
); 
