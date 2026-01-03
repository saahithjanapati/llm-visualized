import * as THREE from 'three';
import BaseLayer from '../src/engine/BaseLayer.js';
import { CoreEngine } from '../src/engine/CoreEngine.js';
import { VectorVisualizationInstancedPrism } from '../src/components/VectorVisualizationInstancedPrism.js';
import { VECTOR_LENGTH_PRISM } from '../src/utils/constants.js';
import { mapValueToColor } from '../src/utils/colors.js';
import { MHA_FINAL_Q_COLOR, MHA_FINAL_K_COLOR, MHA_FINAL_V_COLOR } from '../src/animations/LayerAnimationConstants.js';
import { PARAMETER_CHECKPOINTS } from '../src/data/parameterCheckpoints.js';

const fileInput = document.getElementById('fileInput');
const urlInput = document.getElementById('urlInput');
const loadUrlBtn = document.getElementById('loadUrl');
const statusEl = document.getElementById('status');
const searchInput = document.getElementById('searchInput');
const rotationSpeedInput = document.getElementById('rotationSpeed');
const rotationValue = document.getElementById('rotationValue');
const viewButtons = Array.from(document.querySelectorAll('[data-view]'));
const backgroundSelect = document.getElementById('backgroundSelect');

const treeRoot = document.getElementById('treeRoot');
const rawValuesEl = document.getElementById('rawValues');
const spectrumCanvas = document.getElementById('spectrumCanvas');
const spectrumMinEl = document.getElementById('spectrumMin');
const spectrumMaxEl = document.getElementById('spectrumMax');
const spectrumLabel = document.getElementById('spectrumLabel');
const schemeQCanvas = document.getElementById('schemeQ');
const schemeKCanvas = document.getElementById('schemeK');
const schemeVCanvas = document.getElementById('schemeV');

const hudPath = document.getElementById('hudPath');
const hudSamples = document.getElementById('hudSamples');
const hudMapped = document.getElementById('hudMapped');
const hudQuant = document.getElementById('hudQuant');
const hudStride = document.getElementById('hudStride');

const SPECTRUM_MIN = -2;
const SPECTRUM_MAX = 2;
const SPECTRUM_MARKER_MIN = '#f4b26b';
const SPECTRUM_MARKER_MAX = '#54c1b9';
const MONO_MIN_LIGHTNESS = 0.45;
const MONO_MAX_LIGHTNESS = 0.35;
const GRAYSCALE_MIN = 0.0;
const GRAYSCALE_MAX = 1.0;
const SCORE_SPHERE_RADIUS = 6;
const SCORE_SPHERE_SPACING = 12;
const SCORE_ROW_SPACING = 12;

const HEAD_SCHEMES = {
    q: { label: 'Q', color: new THREE.Color(MHA_FINAL_Q_COLOR) },
    k: { label: 'K', color: new THREE.Color(MHA_FINAL_K_COLOR) },
    v: { label: 'V', color: new THREE.Color(MHA_FINAL_V_COLOR) },
};

const BACKGROUND_PRESETS = {
    black: 0x000000,
    charcoal: 0x141414,
    slate: 0x1a2128,
    dusk: 0x191c2b,
    warm: 0x1f1a14,
};

const state = {
    data: null,
    selectedEl: null,
    selectedPath: null,
    viewMode: 'architecture',
    childLimit: 60,
    activeSpectrum: null,
    config: {
        quantisation: null,
        residualStride: null,
        attentionStride: null,
        mlpStride: null,
    },
    engine: null,
    layer: null,
};

function resetViewerCamera() {
    if (!state.engine || !state.engine.camera || !state.engine.controls) return;
    state.engine.camera.position.set(0, 35, 130);
    state.engine.controls.target.set(0, 0, 0);
    state.engine.notifyCameraUpdated();
    state.engine.controls.update();
}

function applySceneBackground(presetKey) {
    if (!state.engine || !state.engine.scene) return;
    const colorValue = BACKGROUND_PRESETS[presetKey] ?? BACKGROUND_PRESETS.black;
    state.engine.scene.background = new THREE.Color(colorValue);
}

class VectorInspectorLayer extends BaseLayer {
    constructor() {
        super(0);
        this.vector = null;
        this.scoreGroup = new THREE.Group();
        this.scoreMeshes = [];
        this.scoreGeometry = new THREE.SphereGeometry(SCORE_SPHERE_RADIUS, 16, 16);
        this.rotationSpeed = 0.35;
        this.instanceCount = VECTOR_LENGTH_PRISM;
        this.mode = 'vector';
    }

    init(scene) {
        super.init(scene);
        this._buildVector(VECTOR_LENGTH_PRISM);
        this.scoreGroup.visible = false;
        this.root.add(this.scoreGroup);
    }

    update(dt) {
        if (this.mode !== 'vector' || !this.vector) return;
        this.vector.group.rotation.y += dt * this.rotationSpeed;
    }

    setRotationSpeed(speed) {
        if (Number.isFinite(speed)) {
            this.rotationSpeed = speed;
        }
    }

    _buildVector(instanceCount) {
        if (this.vector) {
            this.root.remove(this.vector.group);
            this.vector.dispose();
        }
        this.instanceCount = Math.max(1, Math.floor(instanceCount));
        this.vector = new VectorVisualizationInstancedPrism(
            new Array(this.instanceCount).fill(0),
            new THREE.Vector3(0, 0, 0),
            30,
            this.instanceCount
        );
        this.vector.group.position.y = -this.vector.getUniformHeight() * 0.5;
        this.vector.group.userData.label = 'Vector sample';
        this.root.add(this.vector.group);
    }

    _clearScoreMeshes() {
        this.scoreMeshes.forEach((mesh) => {
            if (mesh.material) mesh.material.dispose();
            this.scoreGroup.remove(mesh);
        });
        this.scoreMeshes.length = 0;
    }

    _setMode(mode) {
        this.mode = mode;
        if (this.vector) this.vector.group.visible = mode === 'vector';
        this.scoreGroup.visible = mode === 'scores';
    }

    setVectorData(values, label, instanceCount) {
        if (!this.vector) return;
        const data = Array.isArray(values) ? values : [];
        const desiredCount = Math.max(1, Math.floor(instanceCount || data.length || VECTOR_LENGTH_PRISM));
        if (desiredCount !== this.instanceCount) {
            this._buildVector(desiredCount);
        }
        this._setMode('vector');
        this.vector.updateDataAndSnapVisuals(data);
        const numKeyColors = Math.min(30, data.length || 1);
        if (this._colorOptions) {
            this.vector.updateKeyColorsFromData(data, numKeyColors, this._colorOptions);
        } else {
            this.vector.updateKeyColorsFromData(data, numKeyColors);
        }
        if (label) {
            this.vector.group.userData.label = label;
        }
    }

    setColorOptions(colorOptions) {
        this._colorOptions = colorOptions || null;
    }

    setScoreData(rows, mode) {
        const scoreRows = Array.isArray(rows) ? rows : [];
        const total = scoreRows.reduce((acc, row) => acc + (Array.isArray(row) ? row.length : 0), 0);
        if (total <= 0) return;
        this._clearScoreMeshes();
        this._setMode('scores');
        this.scoreGroup.position.y = 0;
        this.scoreGroup.rotation.y = 0;
        const maxColumns = scoreRows.reduce((max, row) => Math.max(max, Array.isArray(row) ? row.length : 0), 0);
        const rowCount = scoreRows.length;
        const centerRow = (rowCount - 1) / 2;
        const centerCol = (maxColumns - 1) / 2;
        for (let r = 0; r < scoreRows.length; r++) {
            const row = scoreRows[r] || [];
            for (let c = 0; c < row.length; c++) {
                const x = (c - centerCol) * SCORE_SPHERE_SPACING;
                const z = (r - centerRow) * SCORE_ROW_SPACING;
                const value = row[c];
                const color = mode === 'post' ? mapValueToGrayscale(value) : mapValueToColor(value);
                const material = new THREE.MeshBasicMaterial({ color });
                const mesh = new THREE.Mesh(this.scoreGeometry, material);
                mesh.position.set(x, 0, z);
                this.scoreGroup.add(mesh);
                this.scoreMeshes.push(mesh);
            }
        }
    }
}

function setStatus(message) {
    statusEl.textContent = message;
}

function getType(value) {
    if (value === null) return 'null';
    if (Array.isArray(value)) return 'array';
    return typeof value;
}

function formatPath(path) {
    if (!path.length) return '(root)';
    return path.reduce((acc, seg) => {
        if (typeof seg === 'number') return `${acc}[${seg}]`;
        return acc ? `${acc}.${seg}` : seg;
    }, '');
}

function resolvePath(root, path) {
    let current = root;
    for (const segment of path) {
        if (current == null) return undefined;
        current = current[segment];
    }
    return current;
}

function isNumericArray(arr) {
    if (!Array.isArray(arr) || arr.length === 0) return false;
    const sample = arr.slice(0, 24);
    return sample.every((item) => typeof item === 'number');
}

function isQuantizedVector(obj) {
    return obj && typeof obj === 'object' && !Array.isArray(obj) && Array.isArray(obj.v);
}

function isQuantizedVectorArray(arr) {
    return Array.isArray(arr) && arr.length > 0 && isQuantizedVector(arr[0]);
}

function inferQuantisation(obj) {
    if (!obj || typeof obj !== 'object') return 'unknown';
    if (typeof obj.q === 'string') return obj.q;
    if (typeof obj.s === 'number') return 'int8_sym';
    return state.config.quantisation || 'float';
}

function decodeVector(obj) {
    if (!obj || !Array.isArray(obj.v)) return [];
    const scale = typeof obj.s === 'number' ? obj.s : 1;
    return obj.v.map((val) => {
        const num = Number(val);
        if (!Number.isFinite(num)) return 0;
        return num * scale;
    });
}

function cleanArray(values) {
    return values.map((val) => (Number.isFinite(val) ? val : 0));
}

function downsample(values, targetLength) {
    const ratio = values.length / targetLength;
    const result = new Array(targetLength).fill(0);
    for (let i = 0; i < targetLength; i++) {
        const start = i * ratio;
        const end = (i + 1) * ratio;
        let sum = 0;
        let weight = 0;
        let idx = Math.floor(start);
        while (idx < end && idx < values.length) {
            const left = Math.max(start, idx);
            const right = Math.min(end, idx + 1);
            const w = Math.max(0, right - left);
            sum += values[idx] * w;
            weight += w;
            idx += 1;
        }
        result[i] = weight ? sum / weight : 0;
    }
    return result;
}

function upsample(values, targetLength) {
    if (values.length === 1) return new Array(targetLength).fill(values[0]);
    const result = new Array(targetLength).fill(0);
    for (let i = 0; i < targetLength; i++) {
        const t = targetLength === 1 ? 0 : i / (targetLength - 1);
        const idx = t * (values.length - 1);
        const lo = Math.floor(idx);
        const hi = Math.min(values.length - 1, lo + 1);
        const frac = idx - lo;
        result[i] = values[lo] * (1 - frac) + values[hi] * frac;
    }
    return result;
}

function normalizeVector(values, targetLength) {
    const cleaned = cleanArray(values);
    const length = Math.max(1, Math.floor(targetLength || VECTOR_LENGTH_PRISM));
    if (cleaned.length === length) return cleaned;
    if (cleaned.length === 0) return new Array(length).fill(0);
    if (cleaned.length > length) {
        return downsample(cleaned, length);
    }
    return upsample(cleaned, length);
}

function getVectorStats(values) {
    if (!values || !values.length) return null;
    let min = Infinity;
    let max = -Infinity;
    let sum = 0;
    values.forEach((val) => {
        const num = Number(val);
        if (!Number.isFinite(num)) return;
        min = Math.min(min, num);
        max = Math.max(max, num);
        sum += num;
    });
    const mean = values.length ? sum / values.length : 0;
    return { min, max, mean };
}

function formatValues(values, perLine = 8) {
    if (!values || !values.length) return '(empty)';
    return values
        .map((val) => {
            const num = Number(val);
            if (!Number.isFinite(num)) return '0.0000';
            return num.toFixed(4);
        })
        .reduce((acc, val, idx) => {
            const sep = idx === 0 ? '' : idx % perLine === 0 ? '\n' : ', ';
            return acc + sep + val;
        }, '');
}

function clearSelection() {
    if (state.selectedEl) {
        state.selectedEl.classList.remove('selected');
    }
    state.selectedEl = null;
}

function selectElement(el) {
    clearSelection();
    state.selectedEl = el;
    el.classList.add('selected');
}

function selectPathInTree(path) {
    if (!path) return;
    const target = JSON.stringify(path);
    const nodes = treeRoot.querySelectorAll('[data-path]');
    for (const node of nodes) {
        if (node.dataset.path === target) {
            selectElement(node);
            return;
        }
    }
}

function createLabel(text, className) {
    const span = document.createElement('span');
    span.textContent = text;
    if (className) span.className = className;
    return span;
}

function createGroup(label, meta, open, path) {
    const details = document.createElement('details');
    details.className = 'node';
    if (open) details.open = true;

    const summary = document.createElement('summary');
    summary.className = 'node-summary';
    if (path) summary.dataset.path = JSON.stringify(path);
    summary.appendChild(createLabel(label, 'node-label'));
    if (meta) summary.appendChild(createLabel(meta, 'node-meta'));
    summary.addEventListener('click', () => {
        selectElement(summary);
        if (path) handleSelection(path);
    });
    details.appendChild(summary);

    const container = document.createElement('div');
    container.style.marginLeft = '14px';
    details.appendChild(container);

    return { details, container };
}

function describeValue(value) {
    const type = getType(value);
    if (isQuantizedVector(value)) {
        return `Vector · ${value.v.length} values`;
    }
    if (type === 'array') {
        const meta = isNumericArray(value) ? 'numbers' : `${value.length} items`;
        return `Array · ${meta}`;
    }
    if (type === 'object') {
        const keys = Object.keys(value || {});
        return `Object · ${keys.length} keys`;
    }
    if (type === 'string') {
        return `string · ${value.length} chars`;
    }
    return type;
}

function makeLeafNode(label, value, path) {
    const leaf = document.createElement('div');
    leaf.className = 'leaf';
    leaf.dataset.path = JSON.stringify(path);
    leaf.appendChild(createLabel(label, 'node-label'));
    leaf.appendChild(createLabel(describeValue(value), 'node-meta'));
    leaf.addEventListener('click', () => {
        selectElement(leaf);
        handleSelection(path);
    });
    return leaf;
}

function appendArrayChildren(container, arr, path) {
    const chunk = state.childLimit;
    let start = 0;
    let showMore = null;

    function addChunk() {
        const end = Math.min(arr.length, start + chunk);
        for (let i = start; i < end; i++) {
            container.appendChild(createNode(`[${i}]`, arr[i], path.concat(i)));
        }
        start = end;
        if (start >= arr.length && showMore) {
            showMore.remove();
        }
    }

    addChunk();

    if (arr.length > chunk) {
        showMore = document.createElement('button');
        showMore.textContent = `Show next ${chunk}`;
        showMore.className = 'leaf';
        showMore.addEventListener('click', (event) => {
            event.stopPropagation();
            addChunk();
        });
        container.appendChild(showMore);
    }
}

function appendObjectChildren(container, obj, path) {
    const keys = Object.keys(obj || {});
    const limit = state.childLimit;
    const displayKeys = keys.slice(0, limit);
    displayKeys.forEach((key) => {
        container.appendChild(createNode(key, obj[key], path.concat(key)));
    });
    if (keys.length > limit) {
        const more = document.createElement('div');
        more.className = 'leaf';
        more.textContent = `... ${keys.length - limit} more keys`;
        container.appendChild(more);
    }
}

function createNode(key, value, path) {
    const type = getType(value);
    const label = key;

    if (isQuantizedVector(value)) {
        return makeLeafNode(label, value, path);
    }

    if (type === 'array') {
        const isNumbers = isNumericArray(value);
        if (isNumbers && value.length > state.childLimit) {
            return makeLeafNode(label, value, path);
        }
        const details = document.createElement('details');
        details.className = 'node';
        if (path.length < 2) details.open = true;

        const summary = document.createElement('summary');
        summary.className = 'node-summary';
        summary.dataset.path = JSON.stringify(path);
        summary.appendChild(createLabel(label, 'node-label'));
        summary.appendChild(createLabel(`Array · ${value.length}`, 'node-meta'));
        summary.addEventListener('click', () => {
            selectElement(summary);
            handleSelection(path);
        });
        details.appendChild(summary);

        const container = document.createElement('div');
        container.style.marginLeft = '14px';
        appendArrayChildren(container, value, path);
        details.appendChild(container);
        return details;
    }

    if (type === 'object') {
        const details = document.createElement('details');
        details.className = 'node';
        if (path.length < 2) details.open = true;

        const summary = document.createElement('summary');
        summary.className = 'node-summary';
        summary.dataset.path = JSON.stringify(path);
        summary.appendChild(createLabel(label, 'node-label'));
        summary.appendChild(createLabel(`Object · ${Object.keys(value || {}).length}`, 'node-meta'));
        summary.addEventListener('click', () => {
            selectElement(summary);
            handleSelection(path);
        });
        details.appendChild(summary);

        const container = document.createElement('div');
        container.style.marginLeft = '14px';
        appendObjectChildren(container, value, path);
        details.appendChild(container);
        return details;
    }

    return makeLeafNode(label, value, path);
}

function renderRawTree() {
    treeRoot.innerHTML = '';
    if (!state.data) return;

    const rootNode = createNode('root', state.data, []);
    treeRoot.appendChild(rootNode);
}

function addPath(container, label, path) {
    const value = resolvePath(state.data, path);
    if (value === undefined) return;
    container.appendChild(createNode(label, value, path));
}

function renderArchitectureTree() {
    treeRoot.innerHTML = '';
    if (!state.data) return;

    const root = document.createElement('div');
    const metaGroup = createGroup('Meta', state.data.meta ? 'capture details' : null, true, ['meta']);
    if (state.data.meta) {
        addPath(metaGroup.container, 'Prompt', ['meta', 'prompt']);
        addPath(metaGroup.container, 'Completion', ['meta', 'completion']);
        addPath(metaGroup.container, 'Prompt tokens', ['meta', 'prompt_tokens']);
        addPath(metaGroup.container, 'Completion tokens', ['meta', 'completion_tokens']);
        addPath(metaGroup.container, 'Token strings', ['meta', 'token_strings']);
        addPath(metaGroup.container, 'Capture config', ['meta', 'config']);
    }
    root.appendChild(metaGroup.details);

    const activationsPath = ['activations'];
    const embeddings = resolvePath(state.data, activationsPath.concat('embeddings'));
    if (embeddings) {
        const embedGroup = createGroup('Embeddings', 'token + position', true, activationsPath.concat('embeddings'));
        addPath(embedGroup.container, 'Token embeddings', activationsPath.concat('embeddings', 'token'));
        addPath(embedGroup.container, 'Position embeddings', activationsPath.concat('embeddings', 'position'));
        addPath(embedGroup.container, 'Embedding sum', activationsPath.concat('embeddings', 'sum'));
        root.appendChild(embedGroup.details);
    }

    const layers = resolvePath(state.data, activationsPath.concat('layers'));
    if (Array.isArray(layers)) {
        const layersGroup = createGroup('Layers', `${layers.length} layers`, true, activationsPath.concat('layers'));
        layers.forEach((layerEntry, idx) => {
            const layerPath = activationsPath.concat('layers', idx);
            const layerGroup = createGroup(`Layer ${idx + 1}`, `index ${idx}`, idx === 0, layerPath);

            const residualGroup = createGroup('Residual stream', null, false);
            addPath(residualGroup.container, 'Incoming residual (pre-LN1)', layerPath.concat('incoming'));
            addPath(residualGroup.container, 'Post-attention residual', layerPath.concat('post_attn_residual'));
            addPath(residualGroup.container, 'Post-MLP residual', layerPath.concat('post_mlp_residual'));
            layerGroup.container.appendChild(residualGroup.details);

            const ln1Group = createGroup('LayerNorm 1 (pre-attention)', null, false, layerPath.concat('ln1'));
            addPath(ln1Group.container, 'Normed', layerPath.concat('ln1', 'norm'));
            addPath(ln1Group.container, 'Scaled', layerPath.concat('ln1', 'scale'));
            addPath(ln1Group.container, 'Shifted (LN1 output)', layerPath.concat('ln1', 'shift'));
            layerGroup.container.appendChild(ln1Group.details);

            const attnGroup = createGroup('Attention', null, false);
            addPath(attnGroup.container, 'Q heads', layerPath.concat('qkv', 'q'));
            addPath(attnGroup.container, 'K heads', layerPath.concat('qkv', 'k'));
            addPath(attnGroup.container, 'V heads', layerPath.concat('qkv', 'v'));
            addPath(attnGroup.container, 'Attention scores (pre-softmax)', layerPath.concat('attention_scores', 'pre'));
            addPath(attnGroup.container, 'Attention scores (post-softmax)', layerPath.concat('attention_scores', 'post'));
            addPath(attnGroup.container, 'Attention output projection', layerPath.concat('attn_output_proj'));
            layerGroup.container.appendChild(attnGroup.details);

            const ln2Group = createGroup('LayerNorm 2 (pre-MLP)', null, false, layerPath.concat('ln2'));
            addPath(ln2Group.container, 'Normed', layerPath.concat('ln2', 'norm'));
            addPath(ln2Group.container, 'Scaled', layerPath.concat('ln2', 'scale'));
            addPath(ln2Group.container, 'Shifted (LN2 output)', layerPath.concat('ln2', 'shift'));
            layerGroup.container.appendChild(ln2Group.details);

            const mlpGroup = createGroup('MLP', null, false);
            addPath(mlpGroup.container, 'Up projection', layerPath.concat('mlp_up'));
            addPath(mlpGroup.container, 'Activation', layerPath.concat('mlp_act'));
            addPath(mlpGroup.container, 'Down projection', layerPath.concat('mlp_down'));
            layerGroup.container.appendChild(mlpGroup.details);

            layersGroup.container.appendChild(layerGroup.details);
        });
        root.appendChild(layersGroup.details);
    }

    const finalResidual = resolvePath(state.data, activationsPath.concat('final_residual'));
    if (finalResidual) {
        const finalResidualGroup = createGroup('Final residual stream', null, false, activationsPath.concat('final_residual'));
        addPath(finalResidualGroup.container, 'Residual before final LayerNorm', activationsPath.concat('final_residual'));
        root.appendChild(finalResidualGroup.details);
    }

    const finalLn = resolvePath(state.data, activationsPath.concat('final_layernorm'));
    if (finalLn) {
        const finalGroup = createGroup('Final layernorm', null, false, activationsPath.concat('final_layernorm'));
        addPath(finalGroup.container, 'Normed', activationsPath.concat('final_layernorm', 'norm'));
        addPath(finalGroup.container, 'Scaled', activationsPath.concat('final_layernorm', 'scale'));
        addPath(finalGroup.container, 'Shifted', activationsPath.concat('final_layernorm', 'shift'));
        root.appendChild(finalGroup.details);
    }

    const logits = resolvePath(state.data, ['logits']);
    if (logits) {
        const logitsGroup = createGroup('Sampling logits', null, false, ['logits']);
        addPath(logitsGroup.container, 'Top logits', ['logits']);
        root.appendChild(logitsGroup.details);
    }

    const flops = resolvePath(state.data, ['flops']);
    if (flops) {
        const flopsGroup = createGroup('FLOPs', null, false, ['flops']);
        addPath(flopsGroup.container, 'FLOP checkpoints', ['flops']);
        root.appendChild(flopsGroup.details);
    }

    const params = resolvePath(state.data, ['parameters']);
    if (params) {
        const paramsGroup = createGroup('Parameters', null, false, ['parameters']);
        addPath(paramsGroup.container, 'Parameter checkpoints', ['parameters']);
        root.appendChild(paramsGroup.details);
    }

    treeRoot.appendChild(root);
}

function renderTree() {
    if (state.viewMode === 'raw') {
        renderRawTree();
    } else {
        renderArchitectureTree();
    }
    handleSearch();
    selectPathInTree(state.selectedPath);
}

function handleSearch() {
    const query = searchInput.value.trim().toLowerCase();
    const nodes = treeRoot.querySelectorAll('summary.node-summary, .leaf');
    if (!query) {
        nodes.forEach((node) => {
            node.style.display = '';
        });
        return;
    }
    nodes.forEach((node) => {
        const text = node.textContent.toLowerCase();
        node.style.display = text.includes(query) ? '' : 'none';
    });
}

function unwrapVector(value) {
    if (isQuantizedVector(value)) return { vector: value, note: null };
    if (isQuantizedVectorArray(value)) return { vector: value[0], note: 'Showing index [0] from vector array.' };
    if (Array.isArray(value) && value.length && Array.isArray(value[0]) && isQuantizedVectorArray(value[0])) {
        return { vector: value[0][0], note: 'Showing [0][0] from nested vector array.' };
    }
    return null;
}

function getHeadTypeFromPath(path) {
    if (!Array.isArray(path)) return null;
    const qkvIndex = path.indexOf('qkv');
    if (qkvIndex >= 0 && qkvIndex + 1 < path.length) {
        const head = path[qkvIndex + 1];
        if (head === 'q' || head === 'k' || head === 'v') return head;
    }
    return null;
}

function buildMonochromeOptions(color) {
    const hsl = { h: 0, s: 0, l: 0 };
    color.getHSL(hsl);
    const baseSat = Number.isFinite(hsl.s) ? hsl.s : 0.9;
    return {
        type: 'monochromatic',
        baseHue: hsl.h,
        saturation: Math.min(1, Math.max(0.85, baseSat * 1.2)),
        minLightness: MONO_MIN_LIGHTNESS,
        maxLightness: MONO_MAX_LIGHTNESS,
    };
}

function mapValueToMonoColor(value, scheme) {
    const normalized = (Math.max(SPECTRUM_MIN, Math.min(SPECTRUM_MAX, value)) - SPECTRUM_MIN) / (SPECTRUM_MAX - SPECTRUM_MIN);
    const lightness = MONO_MIN_LIGHTNESS + (MONO_MAX_LIGHTNESS - MONO_MIN_LIGHTNESS) * normalized;
    const hsl = { h: 0, s: 0, l: 0 };
    scheme.color.getHSL(hsl);
    const baseSat = Number.isFinite(hsl.s) ? hsl.s : 0.9;
    const saturation = Math.min(1, Math.max(0.85, baseSat * 1.2));
    return new THREE.Color().setHSL(hsl.h, saturation, lightness);
}

function mapValueToGrayscale(value) {
    const t = Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
    const level = GRAYSCALE_MIN + (GRAYSCALE_MAX - GRAYSCALE_MIN) * t;
    return new THREE.Color(level, level, level);
}

function getAttentionModeFromPath(path) {
    if (!Array.isArray(path)) return null;
    const attnIndex = path.indexOf('attention_scores');
    if (attnIndex >= 0 && attnIndex + 1 < path.length) {
        const mode = path[attnIndex + 1];
        if (mode === 'pre' || mode === 'post') return mode;
    }
    return null;
}

function extractScoreRows(value) {
    if (isQuantizedVector(value)) {
        return { rows: [decodeVector(value)], note: null, sample: value };
    }
    if (isQuantizedVectorArray(value)) {
        return { rows: value.map((vec) => decodeVector(vec)), note: 'Showing all score rows in this array.', sample: value[0] };
    }
    if (Array.isArray(value) && value.length && Array.isArray(value[0]) && isQuantizedVectorArray(value[0])) {
        const rows = [];
        value.forEach((entry) => {
            entry.forEach((vec) => rows.push(decodeVector(vec)));
        });
        const sample = value[0] && value[0][0] ? value[0][0] : null;
        return { rows, note: 'Flattened multiple score rows.', sample };
    }
    return null;
}

function updateHud(label, vector, mappedValues) {
    if (!vector) {
        hudPath.textContent = label || 'None';
        hudSamples.textContent = '--';
        hudMapped.textContent = '--';
        hudQuant.textContent = '--';
        hudStride.textContent = '--';
        return;
    }
    hudPath.textContent = label || 'Vector';
    hudSamples.textContent = String(vector.v.length || '--');
    hudMapped.textContent = String(mappedValues.length || '--');
    hudQuant.textContent = inferQuantisation(vector);
    const stride = state.config.residualStride || state.config.attentionStride || state.config.mlpStride || '--';
    hudStride.textContent = String(stride);
}

function updateHudForScores(label, totalCount, mode) {
    hudPath.textContent = label || 'Attention scores';
    hudSamples.textContent = totalCount ? String(totalCount) : '--';
    hudMapped.textContent = totalCount ? String(totalCount) : '--';
    hudQuant.textContent = mode === 'post' ? 'post-softmax' : 'pre-softmax';
    hudStride.textContent = '--';
}

function updateRawValues(vector, decoded, note) {
    if (!vector) {
        rawValuesEl.textContent = 'Select a vector to see decoded values.';
        spectrumMinEl.textContent = '--';
        spectrumMaxEl.textContent = '--';
        drawSpectrum(null, null);
        if (spectrumLabel) spectrumLabel.textContent = 'Rainbow';
        return;
    }
    const isRowSet = Array.isArray(decoded) && decoded.length && Array.isArray(decoded[0]);
    const flattened = isRowSet ? decoded.flat() : decoded;
    const lines = [];
    if (note) lines.push(note);
    const stats = getVectorStats(flattened);
    if (stats) {
        spectrumMinEl.textContent = stats.min.toFixed(4);
        spectrumMaxEl.textContent = stats.max.toFixed(4);
    } else {
        spectrumMinEl.textContent = '--';
        spectrumMaxEl.textContent = '--';
    }
    drawSpectrum(stats, state.activeSpectrum);

    if (isRowSet) {
        lines.push(`decoded rows (${decoded.length}):`);
        decoded.forEach((row, idx) => {
            lines.push(`row ${idx}: ${formatValues(row, 6)}`);
        });
    } else {
        lines.push(`decoded (${decoded.length}):`);
        lines.push(formatValues(decoded));
    }
    if (typeof vector.s === 'number') {
        lines.push('');
        lines.push(`quantized (scale ${vector.s}):`);
        lines.push(formatValues(vector.v));
    }
    rawValuesEl.textContent = lines.join('\n');
}

function handleSelection(path) {
    if (!state.data) return;
    const value = resolvePath(state.data, path);
    const label = formatPath(path);
    state.selectedPath = path;
    const attentionMode = getAttentionModeFromPath(path);
    if (attentionMode) {
        const extracted = extractScoreRows(value);
        if (!extracted || !extracted.rows.length) {
            state.activeSpectrum = null;
            if (spectrumLabel) spectrumLabel.textContent = 'Rainbow';
            updateHudForScores(label, 0, attentionMode);
            updateRawValues(null, [], null);
            setStatus(`Selected ${label} (no attention scores).`);
            return;
        }
        const totalCount = extracted.rows.reduce((acc, row) => acc + row.length, 0);
        const spectrum = attentionMode === 'post' ? { type: 'grayscale' } : null;
        state.activeSpectrum = spectrum;
        if (spectrumLabel) {
            spectrumLabel.textContent = attentionMode === 'post' ? 'Post-softmax (grayscale)' : 'Pre-softmax';
        }
        if (state.layer) {
            resetViewerCamera();
            state.layer.setColorOptions(null);
            state.layer.setScoreData(extracted.rows, attentionMode);
        }
        updateHudForScores(label, totalCount, attentionMode);
        updateRawValues(extracted.sample || { v: [] }, extracted.rows, extracted.note);
        setStatus(`Selected ${label}. ${extracted.note ? extracted.note : ''}`.trim());
        return;
    }
    const unwrapped = unwrapVector(value);
    if (!unwrapped) {
        state.activeSpectrum = null;
        if (spectrumLabel) spectrumLabel.textContent = 'Rainbow';
        updateHud(label, null, []);
        updateRawValues(null, [], null);
        setStatus(`Selected ${label} (not a vector).`);
        return;
    }

    const decoded = decodeVector(unwrapped.vector);
    const targetLength = decoded.length || VECTOR_LENGTH_PRISM;
    const mappedValues = normalizeVector(decoded, targetLength);
    if (state.layer) {
        resetViewerCamera();
        const headType = getHeadTypeFromPath(path);
        const scheme = headType ? HEAD_SCHEMES[headType] : null;
        state.activeSpectrum = scheme;
        if (spectrumLabel) spectrumLabel.textContent = scheme ? `${scheme.label} scheme` : 'Rainbow';
        state.layer.setColorOptions(scheme ? buildMonochromeOptions(scheme.color) : null);
        state.layer.setVectorData(mappedValues, label, targetLength);
    }
    updateHud(label, unwrapped.vector, mappedValues);
    updateRawValues(unwrapped.vector, decoded, unwrapped.note);
    setStatus(`Selected ${label}. ${unwrapped.note ? unwrapped.note : ''}`.trim());
}

function findFirstVectorPath(value, path = []) {
    if (isQuantizedVector(value)) return path;
    if (Array.isArray(value)) {
        for (let i = 0; i < value.length; i++) {
            const found = findFirstVectorPath(value[i], path.concat(i));
            if (found) return found;
        }
        return null;
    }
    if (value && typeof value === 'object') {
        for (const key of Object.keys(value)) {
            const found = findFirstVectorPath(value[key], path.concat(key));
            if (found) return found;
        }
    }
    return null;
}

function drawSpectrum(stats, scheme) {
    if (!spectrumCanvas) return;
    const ctx = spectrumCanvas.getContext('2d');
    if (!ctx) return;
    const width = spectrumCanvas.width;
    const height = spectrumCanvas.height;
    const isGray = scheme && scheme.type === 'grayscale';
    const isMono = scheme && scheme.color;
    for (let x = 0; x < width; x++) {
        const t = width === 1 ? 0 : x / (width - 1);
        const value = SPECTRUM_MIN + t * (SPECTRUM_MAX - SPECTRUM_MIN);
        const color = isGray
            ? mapValueToGrayscale((value - SPECTRUM_MIN) / (SPECTRUM_MAX - SPECTRUM_MIN))
            : isMono
                ? mapValueToMonoColor(value, scheme)
                : mapValueToColor(value);
        ctx.fillStyle = `rgb(${Math.round(color.r * 255)}, ${Math.round(color.g * 255)}, ${Math.round(color.b * 255)})`;
        ctx.fillRect(x, 0, 1, height);
    }

    if (stats) {
        const toX = (value) => {
            const t = (value - SPECTRUM_MIN) / (SPECTRUM_MAX - SPECTRUM_MIN);
            return Math.round(Math.max(0, Math.min(1, t)) * (width - 1));
        };
        const minX = toX(stats.min);
        const maxX = toX(stats.max);
        ctx.fillStyle = SPECTRUM_MARKER_MIN;
        ctx.fillRect(minX, 0, 2, height);
        ctx.fillStyle = SPECTRUM_MARKER_MAX;
        ctx.fillRect(maxX, 0, 2, height);
    }
}

function drawSchemeCanvas(canvas, scheme) {
    if (!canvas || !scheme) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const width = canvas.width;
    const height = canvas.height;
    for (let x = 0; x < width; x++) {
        const t = width === 1 ? 0 : x / (width - 1);
        const value = SPECTRUM_MIN + t * (SPECTRUM_MAX - SPECTRUM_MIN);
        const color = mapValueToMonoColor(value, scheme);
        ctx.fillStyle = `rgb(${Math.round(color.r * 255)}, ${Math.round(color.g * 255)}, ${Math.round(color.b * 255)})`;
        ctx.fillRect(x, 0, 1, height);
    }
}

function loadJsonData(data, sourceLabel) {
    if (data && typeof data === 'object' && Array.isArray(PARAMETER_CHECKPOINTS)) {
        data.parameters = PARAMETER_CHECKPOINTS;
    }
    state.data = data;
    const meta = data && data.meta && data.meta.config ? data.meta.config : {};
    state.config.quantisation = meta.quantisation || null;
    state.config.residualStride = meta.residual_stride || null;
    state.config.attentionStride = meta.attention_stride || null;
    state.config.mlpStride = meta.mlp_stride || null;

    renderTree();
    const firstPath = findFirstVectorPath(data);
    if (firstPath) {
        handleSelection(firstPath);
        selectPathInTree(firstPath);
    }
    setStatus(`Loaded ${sourceLabel}`);
}

async function loadFromUrl(url) {
    try {
        const resolvedUrl = new URL(url, window.location.href).toString();
        setStatus(`Loading ${resolvedUrl}...`);
        const res = await fetch(resolvedUrl);
        if (!res.ok) throw new Error(`Failed to load (${res.status})`);
        const data = await res.json();
        loadJsonData(data, resolvedUrl);
    } catch (err) {
        setStatus(`Failed to load: ${err.message}`);
    }
}

function loadFromFile(file) {
    const reader = new FileReader();
    reader.onload = () => {
        try {
            const data = JSON.parse(reader.result);
            loadJsonData(data, file.name);
        } catch (err) {
            setStatus(`Invalid JSON: ${err.message}`);
        }
    };
    reader.onerror = () => {
        setStatus('Failed to read file.');
    };
    reader.readAsText(file);
}

function initEngine() {
    const canvas = document.getElementById('viewerCanvas');
    const layer = new VectorInspectorLayer();
    const engine = new CoreEngine(canvas, [layer], {
        enableBloom: false,
        cameraPosition: new THREE.Vector3(0, 35, 130),
        cameraTarget: new THREE.Vector3(0, 0, 0),
        cameraFarMargin: 800,
    });
    engine.setRaycastingEnabled(false);
    engine.controls.minDistance = 40;
    engine.controls.maxDistance = 400;
    engine.controls.enablePan = true;
    engine.controls.update();
    state.engine = engine;
    state.layer = layer;
    if (backgroundSelect) {
        applySceneBackground(backgroundSelect.value);
    }
}

function attachHandlers() {
    fileInput.addEventListener('change', (event) => {
        const file = event.target.files && event.target.files[0];
        if (file) {
            loadFromFile(file);
        }
    });

    loadUrlBtn.addEventListener('click', () => {
        const url = urlInput.value.trim();
        if (url) loadFromUrl(url);
    });

    searchInput.addEventListener('input', handleSearch);

    rotationSpeedInput.addEventListener('input', () => {
        const speed = Number(rotationSpeedInput.value);
        rotationValue.textContent = speed.toFixed(2);
        if (state.layer) state.layer.setRotationSpeed(speed);
    });

    if (backgroundSelect) {
        backgroundSelect.addEventListener('change', () => {
            applySceneBackground(backgroundSelect.value);
        });
    }

    viewButtons.forEach((button) => {
        button.addEventListener('click', () => {
            const mode = button.dataset.view;
            if (!mode || mode === state.viewMode) return;
            state.viewMode = mode;
            viewButtons.forEach((btn) => btn.classList.toggle('active', btn.dataset.view === mode));
            renderTree();
        });
    });
}

function loadDefaultFromQuery() {
    const params = new URLSearchParams(window.location.search);
    const fileParam = params.get('file');
    if (fileParam) {
        urlInput.value = fileParam;
        loadFromUrl(fileParam);
        return;
    }
    if (window.location.hostname) {
        urlInput.value = 'capture.json';
    }
}

initEngine();
attachHandlers();
loadDefaultFromQuery();

rotationValue.textContent = Number(rotationSpeedInput.value).toFixed(2);

drawSpectrum(null);
drawSchemeCanvas(schemeQCanvas, HEAD_SCHEMES.q);
drawSchemeCanvas(schemeKCanvas, HEAD_SCHEMES.k);
drawSchemeCanvas(schemeVCanvas, HEAD_SCHEMES.v);

if (spectrumLabel) spectrumLabel.textContent = 'Rainbow';

window.addEventListener('beforeunload', () => {
    if (state.engine) state.engine.dispose();
});
