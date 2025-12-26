const fileInput = document.getElementById('fileInput');
const urlInput = document.getElementById('urlInput');
const loadUrlBtn = document.getElementById('loadUrl');
const statusEl = document.getElementById('status');
const treeRoot = document.getElementById('treeRoot');
const detailPath = document.getElementById('detailPath');
const detailBody = document.getElementById('detailBody');
const searchInput = document.getElementById('searchInput');
const previewLimitInput = document.getElementById('previewLimit');
const childLimitInput = document.getElementById('childLimit');
const dropOverlay = document.getElementById('dropOverlay');
const viewButtons = Array.from(document.querySelectorAll('[data-view]'));

const state = {
    data: null,
    source: null,
    selectedEl: null,
    viewMode: 'architecture',
    previewLimit: Number(previewLimitInput.value) || 24,
    childLimit: Number(childLimitInput.value) || 50,
};

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

function isNumericArray(arr) {
    if (!Array.isArray(arr) || arr.length === 0) return false;
    const sample = arr.slice(0, 24);
    return sample.every((item) => typeof item === 'number');
}

function isQuantizedVector(obj) {
    return obj
        && typeof obj === 'object'
        && !Array.isArray(obj)
        && Array.isArray(obj.v);
}

function isQuantizedVectorArray(arr) {
    return Array.isArray(arr) && arr.length > 0 && isQuantizedVector(arr[0]);
}

function inferQuantization(obj) {
    if (!obj || typeof obj !== 'object') return 'unknown';
    if (typeof obj.q === 'string') return obj.q;
    if (obj.s != null) return 'int8_sym';
    return 'float16';
}

function getNumericStats(arr) {
    const limit = Math.min(arr.length, 5000);
    let min = Infinity;
    let max = -Infinity;
    let sum = 0;
    for (let i = 0; i < limit; i++) {
        const value = arr[i];
        if (typeof value !== 'number' || Number.isNaN(value)) continue;
        min = Math.min(min, value);
        max = Math.max(max, value);
        sum += value;
    }
    const mean = limit ? sum / limit : 0;
    return { min, max, mean, sampled: limit };
}

function truncateArray(arr, limit) {
    const slice = arr.slice(0, limit);
    return {
        truncated: arr.length > limit,
        value: slice,
    };
}

function truncateObject(obj, limit) {
    const keys = Object.keys(obj);
    const slice = keys.slice(0, limit);
    const preview = {};
    slice.forEach((key) => {
        preview[key] = obj[key];
    });
    return {
        truncated: keys.length > limit,
        value: preview,
    };
}

function buildPreview(value) {
    const limit = state.previewLimit;
    if (Array.isArray(value)) {
        const { value: arr, truncated } = truncateArray(value, limit);
        return {
            truncated,
            preview: arr,
        };
    }
    if (value && typeof value === 'object') {
        const { value: obj, truncated } = truncateObject(value, limit);
        return {
            truncated,
            preview: obj,
        };
    }
    return { truncated: false, preview: value };
}

function stringifyPreview(value) {
    const { preview, truncated } = buildPreview(value);
    const json = JSON.stringify(preview, null, 2);
    return truncated ? `${json}\n... truncated` : json;
}

function resolvePath(root, path) {
    let current = root;
    for (const segment of path) {
        if (current == null) return undefined;
        current = current[segment];
    }
    return current;
}

function clearSelection() {
    if (state.selectedEl) {
        state.selectedEl.classList.remove('selected');
    }
    state.selectedEl = null;
}

function getSelectedPath() {
    if (!state.selectedEl || !state.selectedEl.dataset.path) return null;
    try {
        return JSON.parse(state.selectedEl.dataset.path);
    } catch (err) {
        return null;
    }
}

function selectElement(el) {
    clearSelection();
    state.selectedEl = el;
    el.classList.add('selected');
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
        if (path) {
            renderDetails(path);
        }
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
        return `Vector · ${getVectorLength(value)} values`;
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
        renderDetails(path);
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
        if (start >= arr.length) {
            if (showMore) showMore.remove();
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
    const keys = Object.keys(obj);
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
            renderDetails(path);
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
        summary.appendChild(createLabel(`Object · ${Object.keys(value).length}`, 'node-meta'));
        summary.addEventListener('click', () => {
            selectElement(summary);
            renderDetails(path);
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
}

function renderDetails(path) {
    if (!state.data) return;
    const value = resolvePath(state.data, path);
    detailPath.textContent = formatPath(path);
    detailBody.innerHTML = '';

    const type = getType(value);
    const headerCard = document.createElement('div');
    headerCard.className = 'detail-card';
    headerCard.innerHTML = `<h3>Overview</h3>`;

    const grid = document.createElement('div');
    grid.className = 'detail-grid';

    grid.appendChild(createLabel(`Type: ${type}`));
    if (type === 'array') {
        grid.appendChild(createLabel(`Length: ${value.length}`));
        if (isNumericArray(value)) {
            const stats = getNumericStats(value);
            grid.appendChild(createLabel(`Min: ${stats.min.toFixed(4)}`));
            grid.appendChild(createLabel(`Max: ${stats.max.toFixed(4)}`));
            grid.appendChild(createLabel(`Mean: ${stats.mean.toFixed(4)}`));
            grid.appendChild(createLabel(`Samples: ${stats.sampled}`));
        }
        if (isQuantizedVectorArray(value)) {
            const sample = value[0];
            grid.appendChild(createLabel(`Quantisation: ${inferQuantization(sample)}`));
            grid.appendChild(createLabel(`Vector length: ${getVectorLength(sample)}`));
        }
        if (Array.isArray(value) && value.length && Array.isArray(value[0]) && Array.isArray(value[0][0])) {
            grid.appendChild(createLabel('Nested array'));
        }
    }
    if (type === 'object') {
        const keys = Object.keys(value || {});
        grid.appendChild(createLabel(`Keys: ${keys.length}`));
        if (isQuantizedVector(value)) {
            grid.appendChild(createLabel(`Quantisation: ${inferQuantization(value)}`));
            grid.appendChild(createLabel(`Vector length: ${getVectorLength(value)}`));
            if (value.s != null) {
                grid.appendChild(createLabel(`Scale: ${value.s}`));
            }
        }
    }
    if (type === 'string') {
        grid.appendChild(createLabel(`Length: ${value.length}`));
    }

    headerCard.appendChild(grid);
    detailBody.appendChild(headerCard);

    if (isQuantizedVector(value)) {
        const vectorCard = document.createElement('div');
        vectorCard.className = 'detail-card';
        vectorCard.innerHTML = '<h3>Quantised vector</h3>';
        const limit = Math.min(state.previewLimit, getVectorLength(value));
        const preview = getVectorValues(value, limit);
        let decoded = null;
        if (preview && value.s != null) {
            decoded = preview.map((item) => Number(item) * Number(value.s));
        }

        const info = document.createElement('div');
        info.className = 'detail-grid';
        info.appendChild(createLabel(`q: ${inferQuantization(value)}`));
        info.appendChild(createLabel(`len: ${getVectorLength(value)}`));
        if (value.s != null) {
            info.appendChild(createLabel(`scale: ${value.s}`));
        }
        vectorCard.appendChild(info);

        const raw = document.createElement('pre');
        raw.className = 'mono';
        raw.textContent = JSON.stringify(preview, null, 2);
        vectorCard.appendChild(raw);
        if (decoded) {
            const decodedEl = document.createElement('pre');
            decodedEl.className = 'mono';
            decodedEl.textContent = JSON.stringify(decoded, null, 2);
            vectorCard.appendChild(decodedEl);
        }
        detailBody.appendChild(vectorCard);
    }

    const previewCard = document.createElement('div');
    previewCard.className = 'detail-card';
    previewCard.innerHTML = '<h3>Preview</h3>';
    const previewBlock = document.createElement('pre');
    previewBlock.className = 'mono';
    previewBlock.textContent = stringifyPreview(value);
    previewCard.appendChild(previewBlock);

    const actions = document.createElement('div');
    actions.className = 'actions';
    const copyBtn = document.createElement('button');
    copyBtn.textContent = 'Copy JSON preview';
    copyBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(previewBlock.textContent).catch(() => {});
    });
    actions.appendChild(copyBtn);
    previewCard.appendChild(actions);

    detailBody.appendChild(previewCard);
}

function loadParsedData(data, sourceLabel) {
    state.data = data;
    state.source = sourceLabel;
    setStatus(`Loaded ${sourceLabel}`);
    renderTree();
    renderDetails([]);
}

function loadText(text, sourceLabel) {
    const cleaned = text.replace(/^\uFEFF/, '');
    try {
        loadParsedData(JSON.parse(cleaned), sourceLabel);
    } catch (err) {
        console.error(err);
        const snippet = cleaned.slice(0, 200).replace(/\s+/g, ' ').trim();
        setStatus(`Failed to parse ${sourceLabel}. Starts with: ${snippet || '(empty)'} `);
    }
}

function loadFromFile(file) {
    const reader = new FileReader();
    reader.onload = () => {
        const text = reader.result;
        const label = `${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
        loadText(text, label);
    };
    reader.readAsText(file);
}

function loadFromUrl(url) {
    const resolvedUrl = new URL(url, window.location.href).toString();
    setStatus(`Loading ${resolvedUrl}...`);
    fetch(resolvedUrl)
        .then((response) => {
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            return response.text();
        })
        .then((text) => loadText(text, resolvedUrl))
        .catch((err) => {
            console.error(err);
            setStatus(`Failed to load ${resolvedUrl}`);
        });
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

function initDragDrop() {
    const show = () => dropOverlay.classList.add('active');
    const hide = () => dropOverlay.classList.remove('active');

    window.addEventListener('dragover', (event) => {
        event.preventDefault();
        show();
    });
    window.addEventListener('dragleave', (event) => {
        if (event.relatedTarget === null) hide();
    });
    window.addEventListener('drop', (event) => {
        event.preventDefault();
        hide();
        if (event.dataTransfer.files && event.dataTransfer.files[0]) {
            loadFromFile(event.dataTransfer.files[0]);
        }
    });
}

fileInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) loadFromFile(file);
});

loadUrlBtn.addEventListener('click', () => {
    const url = urlInput.value.trim();
    if (url) loadFromUrl(url);
});

searchInput.addEventListener('input', handleSearch);

previewLimitInput.addEventListener('change', () => {
    state.previewLimit = Number(previewLimitInput.value) || 24;
    const selectedPath = getSelectedPath() || [];
    if (state.data) renderDetails(selectedPath);
});

childLimitInput.addEventListener('change', () => {
    state.childLimit = Number(childLimitInput.value) || 50;
    if (state.data) renderTree();
});

viewButtons.forEach((button) => {
    button.addEventListener('click', () => {
        const mode = button.dataset.view;
        if (!mode || mode === state.viewMode) return;
        state.viewMode = mode;
        viewButtons.forEach((btn) => btn.classList.toggle('active', btn.dataset.view === mode));
        renderTree();
    });
});

initDragDrop();

const params = new URLSearchParams(window.location.search);
const fileParam = params.get('file');
if (fileParam) {
    loadFromUrl(fileParam);
}
function getVectorLength(vec) {
    if (!vec) return 0;
    return Array.isArray(vec.v) ? vec.v.length : 0;
}

function getVectorValues(vec, limit) {
    if (!vec) return null;
    return Array.isArray(vec.v) ? vec.v.slice(0, limit) : [];
}
