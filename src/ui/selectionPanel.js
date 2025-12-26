import * as THREE from 'three';
import { WeightMatrixVisualization } from '../components/WeightMatrixVisualization.js';
import { VectorVisualizationInstancedPrism } from '../components/VectorVisualizationInstancedPrism.js';
import {
    MHA_MATRIX_PARAMS,
    MLP_MATRIX_PARAMS_UP,
    MLP_MATRIX_PARAMS_DOWN,
    EMBEDDING_MATRIX_PARAMS_VOCAB,
    EMBEDDING_MATRIX_PARAMS_POSITION
} from '../utils/constants.js';
import {
    MHA_FINAL_Q_COLOR,
    MHA_FINAL_K_COLOR,
    MHA_FINAL_V_COLOR,
    MHA_OUTPUT_PROJECTION_MATRIX_COLOR,
    MHA_OUTPUT_PROJECTION_MATRIX_PARAMS
} from '../animations/LayerAnimationConstants.js';

const PREVIEW_LANES = 3;
const PREVIEW_MATRIX_DEPTH = 320;
const PREVIEW_LANE_SPACING = 80;
const PREVIEW_TARGET_SIZE = 140;
const PREVIEW_ROTATION_SPEED = 0.0035;

const D_MODEL = 768;
const VOCAB_SIZE = 50257;
const CONTEXT_LEN = 1024;

function formatNumber(value) {
    if (!Number.isFinite(value)) return 'TBD';
    return Math.round(value).toLocaleString('en-US');
}

function formatDims(rows, cols) {
    if (!Number.isFinite(rows) || !Number.isFinite(cols)) return 'TBD';
    return `${rows} x ${cols}`;
}

function resolveMetadata(label, kind = null) {
    const lower = (label || '').toLowerCase();
    if (lower.includes('query weight matrix')) {
        return { params: formatNumber(D_MODEL * D_MODEL), dims: formatDims(D_MODEL, D_MODEL) };
    }
    if (lower.includes('key weight matrix')) {
        return { params: formatNumber(D_MODEL * D_MODEL), dims: formatDims(D_MODEL, D_MODEL) };
    }
    if (lower.includes('value weight matrix')) {
        return { params: formatNumber(D_MODEL * D_MODEL), dims: formatDims(D_MODEL, D_MODEL) };
    }
    if (lower.includes('output projection matrix')) {
        return { params: formatNumber(D_MODEL * D_MODEL), dims: formatDims(D_MODEL, D_MODEL) };
    }
    if (lower.includes('mlp up weight matrix')) {
        return { params: formatNumber(D_MODEL * D_MODEL * 4), dims: formatDims(D_MODEL, D_MODEL * 4) };
    }
    if (lower.includes('mlp down weight matrix')) {
        return { params: formatNumber(D_MODEL * D_MODEL * 4), dims: formatDims(D_MODEL * 4, D_MODEL) };
    }
    if (lower.includes('vocab embedding')) {
        return { params: formatNumber(VOCAB_SIZE * D_MODEL), dims: formatDims(VOCAB_SIZE, D_MODEL) };
    }
    if (lower.includes('positional embedding')) {
        return { params: formatNumber(CONTEXT_LEN * D_MODEL), dims: formatDims(CONTEXT_LEN, D_MODEL) };
    }
    if (lower.includes('query vector') || lower.includes('key vector') || lower.includes('value vector')) {
        return { params: 'TBD', dims: 'TBD' };
    }
    if (lower.includes('attention') || (kind === 'mergedKV')) {
        return { params: 'TBD', dims: 'TBD' };
    }
    return { params: 'TBD', dims: 'TBD' };
}

function buildWeightMatrixPreview(params, colorHex) {
    const matrix = new WeightMatrixVisualization(
        null,
        new THREE.Vector3(0, 0, 0),
        params.width,
        params.height,
        PREVIEW_MATRIX_DEPTH,
        params.topWidthFactor,
        params.cornerRadius,
        PREVIEW_LANES,
        params.slitWidth,
        params.slitDepthFactor,
        params.slitBottomWidthFactor,
        params.slitTopWidthFactor
    );
    if (colorHex !== null && colorHex !== undefined) {
        matrix.setColor(new THREE.Color(colorHex));
    }
    matrix.setMaterialProperties({ opacity: 0.98, transparent: false, emissiveIntensity: 0.18 });
    return {
        object: matrix.group,
        dispose: () => {
            const meshes = [matrix.mesh, matrix.frontCapMesh, matrix.backCapMesh];
            meshes.forEach(mesh => {
                if (!mesh || !mesh.material) return;
                if (Array.isArray(mesh.material)) {
                    mesh.material.forEach(mat => mat && mat.dispose && mat.dispose());
                } else {
                    mesh.material.dispose();
                }
            });
        }
    };
}

function buildVectorPreview(colorHex) {
    const group = new THREE.Group();
    const vectors = [];
    const color = new THREE.Color(colorHex || 0xffffff);
    for (let i = 0; i < PREVIEW_LANES; i++) {
        const vec = new VectorVisualizationInstancedPrism(null, new THREE.Vector3(0, 0, 0), 1);
        vec.numSubsections = 1;
        vec.currentKeyColors = [color.clone(), color.clone()];
        vec.updateInstanceGeometryAndColors();
        vec.group.position.z = (i - (PREVIEW_LANES - 1) / 2) * PREVIEW_LANE_SPACING;
        group.add(vec.group);
        vectors.push(vec);
    }
    return {
        object: group,
        dispose: () => {
            vectors.forEach(vec => {
                if (vec.mesh?.geometry) vec.mesh.geometry.dispose();
                if (vec.mesh?.material) vec.mesh.material.dispose();
            });
        }
    };
}

function buildStackedBoxPreview(colorHex) {
    const group = new THREE.Group();
    const geometry = new THREE.BoxGeometry(140, 140, 8);
    const meshes = [];
    for (let i = 0; i < PREVIEW_LANES; i++) {
        const material = new THREE.MeshStandardMaterial({
            color: colorHex || 0x1f1f1f,
            metalness: 0.25,
            roughness: 0.65,
            emissive: new THREE.Color(0x060606),
            emissiveIntensity: 0.3
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.z = (i - (PREVIEW_LANES - 1) / 2) * 18;
        group.add(mesh);
        meshes.push(mesh);
    }
    return {
        object: group,
        dispose: () => {
            geometry.dispose();
            meshes.forEach(mesh => mesh.material && mesh.material.dispose());
        }
    };
}

function resolvePreviewObject(label, selectionInfo) {
    const lower = (label || '').toLowerCase();
    if (lower.includes('query weight matrix')) {
        return buildWeightMatrixPreview(MHA_MATRIX_PARAMS, MHA_FINAL_Q_COLOR);
    }
    if (lower.includes('key weight matrix')) {
        return buildWeightMatrixPreview(MHA_MATRIX_PARAMS, MHA_FINAL_K_COLOR);
    }
    if (lower.includes('value weight matrix')) {
        return buildWeightMatrixPreview(MHA_MATRIX_PARAMS, MHA_FINAL_V_COLOR);
    }
    if (lower.includes('output projection matrix')) {
        const height = MHA_MATRIX_PARAMS.height * MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.heightFactor;
        const params = {
            ...MHA_MATRIX_PARAMS,
            width: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.width,
            height,
            topWidthFactor: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.topWidthFactor,
            cornerRadius: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.cornerRadius,
            slitWidth: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.slitWidth,
            slitDepthFactor: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.slitDepthFactor,
            slitBottomWidthFactor: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.slitBottomWidthFactor,
            slitTopWidthFactor: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.slitTopWidthFactor
        };
        return buildWeightMatrixPreview(params, MHA_OUTPUT_PROJECTION_MATRIX_COLOR);
    }
    if (lower.includes('mlp up weight matrix')) {
        return buildWeightMatrixPreview(MLP_MATRIX_PARAMS_UP, 0xf59e0b);
    }
    if (lower.includes('mlp down weight matrix')) {
        return buildWeightMatrixPreview(MLP_MATRIX_PARAMS_DOWN, 0xf59e0b);
    }
    if (lower.includes('vocab embedding')) {
        return buildWeightMatrixPreview(EMBEDDING_MATRIX_PARAMS_VOCAB, MHA_FINAL_Q_COLOR);
    }
    if (lower.includes('positional embedding')) {
        return buildWeightMatrixPreview(EMBEDDING_MATRIX_PARAMS_POSITION, MHA_FINAL_K_COLOR);
    }

    if (lower.includes('query vector')) {
        return buildVectorPreview(MHA_FINAL_Q_COLOR);
    }
    if (lower.includes('key vector')) {
        return buildVectorPreview(MHA_FINAL_K_COLOR);
    }
    if (lower.includes('value vector')) {
        return buildVectorPreview(MHA_FINAL_V_COLOR);
    }
    if (selectionInfo?.kind === 'mergedKV') {
        if (selectionInfo.info?.category === 'V') {
            return buildVectorPreview(MHA_FINAL_V_COLOR);
        }
        return buildVectorPreview(MHA_FINAL_K_COLOR);
    }

    if (lower.includes('attention')) {
        return buildStackedBoxPreview(0x1b1b1b);
    }

    return buildStackedBoxPreview(0x202020);
}

function fitObjectToView(object, camera) {
    if (!object) return;
    const box = new THREE.Box3().setFromObject(object);
    if (box.isEmpty()) return;
    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    box.getSize(size);
    box.getCenter(center);
    object.position.sub(center);
    const maxDim = Math.max(size.x, size.y, size.z);
    const scale = maxDim > 0 ? PREVIEW_TARGET_SIZE / maxDim : 1;
    object.scale.setScalar(scale);

    const scaledBox = new THREE.Box3().setFromObject(object);
    const scaledSize = new THREE.Vector3();
    scaledBox.getSize(scaledSize);
    const scaledMax = Math.max(scaledSize.x, scaledSize.y, scaledSize.z);
    const fov = THREE.MathUtils.degToRad(camera.fov);
    const distance = (scaledMax / 2) / Math.tan(fov / 2);

    camera.near = Math.max(0.1, distance / 50);
    camera.far = distance * 20;
    camera.position.set(0, 0, distance * 1.6);
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();
}

class SelectionPanel {
    constructor() {
        this.panel = document.getElementById('detailPanel');
        this.title = document.getElementById('detailTitle');
        this.params = document.getElementById('detailParams');
        this.dims = document.getElementById('detailDims');
        this.closeBtn = document.getElementById('detailClose');
        this.canvas = document.getElementById('detailCanvas');

        if (!this.panel || !this.canvas || !this.title) {
            this.isReady = false;
            return;
        }

        this.isReady = true;
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000000);

        this.camera = new THREE.PerspectiveCamera(35, 1, 0.1, 1000);
        this.camera.position.set(0, 0, 220);

        this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, alpha: false });
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        this.renderer.setClearColor(0x000000, 1);

        const ambient = new THREE.AmbientLight(0xffffff, 0.7);
        const key = new THREE.DirectionalLight(0xffffff, 0.8);
        key.position.set(1, 1, 1);
        this.scene.add(ambient, key);

        this.currentPreview = null;
        this.currentDispose = null;
        this.isOpen = false;

        this._animate = this._animate.bind(this);
        this._onResize = this._onResize.bind(this);
        this._onKeydown = this._onKeydown.bind(this);
        this._startLoop();

        this.closeBtn?.addEventListener('click', () => this.close());
        window.addEventListener('resize', this._onResize);
        document.addEventListener('keydown', this._onKeydown);
        this._observeResize();
        this._onResize();
    }

    _observeResize() {
        if (!('ResizeObserver' in window) || !this.canvas?.parentElement) return;
        this._resizeObserver = new ResizeObserver(() => this._onResize());
        this._resizeObserver.observe(this.canvas.parentElement);
    }

    _onResize() {
        if (!this.isReady) return;
        const rect = this.canvas.getBoundingClientRect();
        const width = Math.max(1, Math.floor(rect.width));
        const height = Math.max(1, Math.floor(rect.height));
        this.renderer.setSize(width, height, false);
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
    }

    _onKeydown(event) {
        if (event.key === 'Escape' && this.isOpen) {
            this.close();
        }
    }

    _startLoop() {
        if (this._loopStarted) return;
        this._loopStarted = true;
        requestAnimationFrame(this._animate);
    }

    _animate() {
        requestAnimationFrame(this._animate);
        if (!this.isReady || !this.isOpen || !this.currentPreview) return;
        this.currentPreview.rotation.y += PREVIEW_ROTATION_SPEED;
        this.currentPreview.rotation.x += PREVIEW_ROTATION_SPEED * 0.45;
        this.renderer.render(this.scene, this.camera);
    }

    open() {
        if (!this.isReady) return;
        this.isOpen = true;
        this.panel.classList.add('is-open');
        this.panel.setAttribute('aria-hidden', 'false');
    }

    close() {
        if (!this.isReady) return;
        this.isOpen = false;
        this.panel.classList.remove('is-open');
        this.panel.setAttribute('aria-hidden', 'true');
    }

    showSelection(selection) {
        if (!this.isReady || !selection || !selection.label) return;

        const label = selection.label;
        const metadata = resolveMetadata(label, selection.kind);
        this.title.textContent = label;
        if (this.params) this.params.textContent = metadata.params;
        if (this.dims) this.dims.textContent = metadata.dims;

        if (this.currentPreview) {
            this.scene.remove(this.currentPreview);
            if (this.currentDispose) {
                try { this.currentDispose(); } catch (_) { /* no-op */ }
            }
            this.currentPreview = null;
            this.currentDispose = null;
        }

        const preview = resolvePreviewObject(label, selection);
        this.currentPreview = preview.object;
        this.currentDispose = preview.dispose;
        this.currentPreview.rotation.set(-0.35, 0.35, 0);
        fitObjectToView(this.currentPreview, this.camera);
        this.scene.add(this.currentPreview);

        this.open();
    }
}

export function initSelectionPanel() {
    const panel = new SelectionPanel();
    if (!panel.isReady) {
        return { handleSelection: () => {}, close: () => {} };
    }
    return {
        handleSelection: (selection) => panel.showSelection(selection),
        close: () => panel.close()
    };
}
