import {
    CONTEXT_LEN,
    D_MODEL
} from '../../ui/selectionPanelConstants.js';
import {
    createGroupNode,
    createMatrixNode,
    createTextNode,
    VIEW2D_LAYOUT_DIRECTIONS,
    VIEW2D_MATRIX_PRESENTATIONS,
    VIEW2D_MATRIX_SHAPES
} from '../schema/sceneTypes.js';
import { VIEW2D_STYLE_KEYS } from '../theme/visualTokens.js';

const CARD_LABEL_HORIZONTAL_INSET = 12;
const EMBEDDING_STREAM_LABEL_MIN_SCREEN_FONT_PX = 10;

function mergeSemantic(baseSemantic = {}, extra = {}) {
    return {
        ...(baseSemantic && typeof baseSemantic === 'object' ? baseSemantic : {}),
        ...(extra && typeof extra === 'object' ? extra : {})
    };
}

function buildLabel(tex = '', text = '') {
    return {
        tex: typeof tex === 'string' ? tex : '',
        text: typeof text === 'string' && text.length ? text : tex
    };
}

function createTextFitMetadata(maxWidth = null) {
    if (!Number.isFinite(maxWidth) || maxWidth <= 0) return null;
    return {
        textFit: {
            maxWidth: Math.floor(maxWidth)
        }
    };
}

function createPersistentTextMetadata({
    maxWidth = null,
    persistentMinScreenFontPx = null
} = {}) {
    const metadata = {
        ...(createTextFitMetadata(maxWidth) || {})
    };
    if (Number.isFinite(persistentMinScreenFontPx) && persistentMinScreenFontPx > 0) {
        metadata.persistentMinScreenFontPx = Number(persistentMinScreenFontPx);
    }
    return Object.keys(metadata).length ? metadata : null;
}

function createCardMetadata(width = null, height = null, {
    cornerRadius = null,
    shape = '',
    shapeConfig = null
} = {}) {
    const card = {};
    if (Number.isFinite(width) && width > 0) card.width = Math.floor(width);
    if (Number.isFinite(height) && height > 0) card.height = Math.floor(height);
    if (Number.isFinite(cornerRadius) && cornerRadius >= 0) card.cornerRadius = Math.floor(cornerRadius);
    if (typeof shape === 'string' && shape.length) card.shape = shape;
    if (shapeConfig && typeof shapeConfig === 'object') {
        card.shapeConfig = { ...shapeConfig };
    }
    return Object.keys(card).length ? { card } : null;
}

export function buildPositionEmbeddingModule({
    semantic = {},
    title = 'Position Embedding',
    cardWidth = 148,
    cardHeight = 108
} = {}) {
    return buildEmbeddingStreamModule({
        semantic,
        title,
        role: 'position-embedding-card',
        labelText: 'W position',
        labelTex: 'W_{position}',
        styleKey: VIEW2D_STYLE_KEYS.EMBEDDING_POSITION_STREAM,
        cardWidth,
        cardHeight,
        persistentMinScreenFontPx: EMBEDDING_STREAM_LABEL_MIN_SCREEN_FONT_PX
    });
}

export function buildVocabularyEmbeddingModule({
    semantic = {},
    title = 'Vocabulary Embedding',
    cardWidth = 196,
    cardHeight = 144
} = {}) {
    return buildEmbeddingStreamModule({
        semantic,
        title,
        role: 'vocabulary-embedding-card',
        labelText: 'W vocabulary',
        labelTex: 'W_{vocabulary}',
        styleKey: VIEW2D_STYLE_KEYS.EMBEDDING_TOKEN_STREAM,
        cardWidth,
        cardHeight,
        wideSide: 'left',
        persistentMinScreenFontPx: EMBEDDING_STREAM_LABEL_MIN_SCREEN_FONT_PX
    });
}

export function buildUnembeddingModule({
    semantic = {},
    title = 'Unembedding',
    cardWidth = 196,
    cardHeight = 144
} = {}) {
    return buildEmbeddingStreamModule({
        semantic,
        title,
        role: 'unembedding',
        labelText: 'W_U',
        labelTex: 'W_U',
        styleKey: VIEW2D_STYLE_KEYS.EMBEDDING_TOKEN_STREAM,
        cardWidth,
        cardHeight,
        wideSide: 'right'
    });
}

function createEmbeddingStreamShapeConfig(wideSide = 'left') {
    if (String(wideSide || '').trim().toLowerCase() === 'right') {
        return {
            leftInset: 4,
            rightInset: 0,
            leftHeightRatio: 0.36,
            rightHeightRatio: 1,
            cornerRadius: 14
        };
    }
    return {
        leftInset: 0,
        rightInset: 4,
        leftHeightRatio: 1,
        rightHeightRatio: 0.36,
        cornerRadius: 14
    };
}

function buildEmbeddingStreamModule({
    semantic = {},
    title = '',
    role = 'embedding-stream-card',
    labelText = '',
    labelTex = '',
    styleKey = VIEW2D_STYLE_KEYS.EMBEDDING_TOKEN_STREAM,
    cardWidth = 180,
    cardHeight = 128,
    wideSide = 'left',
    persistentMinScreenFontPx = null
} = {}) {
    const titleMaxWidth = Math.max(24, cardWidth - (CARD_LABEL_HORIZONTAL_INSET * 2));

    const cardNode = createMatrixNode({
        role,
        semantic: mergeSemantic(semantic, { role }),
        dimensions: {
            rows: CONTEXT_LEN,
            cols: D_MODEL
        },
        presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
        shape: VIEW2D_MATRIX_SHAPES.MATRIX,
        visual: {
            styleKey
        },
        metadata: createCardMetadata(cardWidth, cardHeight, {
            cornerRadius: 18,
            shape: 'curved-trapezoid',
            shapeConfig: createEmbeddingStreamShapeConfig(wideSide)
        })
    });

    const titleNode = createTextNode({
        role: 'module-title',
        semantic: mergeSemantic(semantic, { role: 'module-title' }),
        text: labelText,
        tex: labelTex,
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.LABEL
        },
        metadata: createPersistentTextMetadata({
            maxWidth: titleMaxWidth,
            persistentMinScreenFontPx
        })
    });

    return {
        node: createGroupNode({
            role: semantic.role || 'module',
            semantic,
            direction: VIEW2D_LAYOUT_DIRECTIONS.OVERLAY,
            gapKey: 'default',
            children: [
                cardNode,
                titleNode
            ]
        }),
        cardNode
    };
}
