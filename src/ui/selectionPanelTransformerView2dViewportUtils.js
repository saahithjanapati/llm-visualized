import { resolveViewportFitTransform } from '../view2d/runtime/View2dViewportController.js';

export const TRANSFORMER_VIEW2D_OVERVIEW_MIN_SCALE_DEFAULT = 0.035;

const TRANSFORMER_VIEW2D_OVERVIEW_MIN_SCALE_SMALL_SCREEN_FLOOR = 0.015;
const TRANSFORMER_VIEW2D_OVERVIEW_MIN_SCALE_REFERENCE_WIDTH = 640;
const TRANSFORMER_VIEW2D_FIT_SCENE_MIN_SCALE_EPSILON = 1e-4;
const TRANSFORMER_VIEW2D_FIT_SCENE_SCALE_EPSILON_RATIO = 0.001;
const TRANSFORMER_VIEW2D_FIT_SCENE_PAN_EPSILON_PX = 1.5;

export function resolveTransformerView2dOverviewMinScale({
    isSmallScreen = false,
    viewportWidth = 0
} = {}) {
    if (!isSmallScreen) {
        return TRANSFORMER_VIEW2D_OVERVIEW_MIN_SCALE_DEFAULT;
    }

    const safeViewportWidth = Math.max(1, Math.floor(Number(viewportWidth) || 0));
    const widthRatio = Math.min(1, safeViewportWidth / TRANSFORMER_VIEW2D_OVERVIEW_MIN_SCALE_REFERENCE_WIDTH);

    return Math.max(
        TRANSFORMER_VIEW2D_OVERVIEW_MIN_SCALE_SMALL_SCREEN_FLOOR,
        TRANSFORMER_VIEW2D_OVERVIEW_MIN_SCALE_DEFAULT * widthRatio
    );
}

function resolveViewportDimension(value = 0) {
    return Number.isFinite(value) && value > 0 ? value : 0;
}

function resolveFitSceneViewport(controllerState = null, {
    viewportWidth = null,
    viewportHeight = null
} = {}) {
    const width = resolveViewportDimension(
        Number.isFinite(viewportWidth) ? viewportWidth : controllerState?.viewport?.width
    );
    const height = resolveViewportDimension(
        Number.isFinite(viewportHeight) ? viewportHeight : controllerState?.viewport?.height
    );

    if (!(width > 0) || !(height > 0)) {
        return null;
    }

    return { width, height };
}

function resolveTransformerView2dFitSceneTransform({
    controllerState = null,
    fitBounds = null,
    viewportWidth = null,
    viewportHeight = null,
    padding = 24,
    minScale = 0.05,
    maxScale = 4
} = {}) {
    const viewport = resolveFitSceneViewport(controllerState, {
        viewportWidth,
        viewportHeight
    });
    if (!viewport) return null;

    return resolveViewportFitTransform(fitBounds, viewport, {
        padding,
        minScale,
        maxScale
    });
}

export function isTransformerView2dViewportAtFitScene({
    controllerState = null,
    fitBounds = null,
    viewportWidth = null,
    viewportHeight = null,
    padding = 24,
    minScale = 0.05,
    maxScale = 4
} = {}) {
    const fitTransform = resolveTransformerView2dFitSceneTransform({
        controllerState,
        fitBounds,
        viewportWidth,
        viewportHeight,
        padding,
        minScale,
        maxScale
    });
    if (!fitTransform) return false;

    const scale = Number(controllerState?.scale);
    const panX = Number(controllerState?.panX);
    const panY = Number(controllerState?.panY);
    if (!(Number.isFinite(scale) && Number.isFinite(panX) && Number.isFinite(panY))) {
        return false;
    }

    const scaleEpsilon = Math.max(
        TRANSFORMER_VIEW2D_FIT_SCENE_MIN_SCALE_EPSILON,
        Math.abs(fitTransform.scale) * TRANSFORMER_VIEW2D_FIT_SCENE_SCALE_EPSILON_RATIO
    );

    return Math.abs(scale - fitTransform.scale) <= scaleEpsilon
        && Math.abs(panX - fitTransform.panX) <= TRANSFORMER_VIEW2D_FIT_SCENE_PAN_EPSILON_PX
        && Math.abs(panY - fitTransform.panY) <= TRANSFORMER_VIEW2D_FIT_SCENE_PAN_EPSILON_PX;
}

export function shouldShowTransformerView2dFitSceneAction(options = {}) {
    const fitTransform = resolveTransformerView2dFitSceneTransform(options);
    if (!fitTransform) return false;
    return !isTransformerView2dViewportAtFitScene(options);
}
