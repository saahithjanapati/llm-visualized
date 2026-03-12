import {
    MHA_FINAL_K_COLOR,
    MHA_FINAL_Q_COLOR,
    MHA_FINAL_V_COLOR
} from '../../animations/LayerAnimationConstants.js';
import {
    FINAL_MLP_COLOR,
    PREVIEW_TRAIL_COLOR
} from '../../ui/selectionPanelConstants.js';

function hexToRgb(hexValue = 0xFFFFFF) {
    const safe = Number.isFinite(hexValue)
        ? Math.max(0, Math.min(0xFFFFFF, Math.floor(hexValue)))
        : 0xFFFFFF;
    return [
        (safe >> 16) & 0xFF,
        (safe >> 8) & 0xFF,
        safe & 0xFF
    ];
}

function hexToCss(hexValue = 0xFFFFFF, alpha = 1) {
    const [r, g, b] = hexToRgb(hexValue);
    if (alpha >= 1) {
        return `rgb(${r}, ${g}, ${b})`;
    }
    return `rgba(${r}, ${g}, ${b}, ${Math.max(0, Math.min(1, alpha)).toFixed(3)})`;
}

export const VIEW2D_STYLE_KEYS = Object.freeze({
    SCENE: 'scene',
    LABEL: 'label',
    CAPTION: 'caption',
    OPERATOR: 'operator',
    EMBEDDING_TOKEN: 'embedding.token',
    EMBEDDING_POSITION: 'embedding.position',
    LAYER_NORM: 'layer-norm',
    MATRIX_INPUT: 'matrix.input',
    MATRIX_WEIGHT: 'matrix.weight',
    MATRIX_BIAS: 'matrix.bias',
    OUTPUT_PROJECTION: 'output-projection',
    MHSA_Q: 'mhsa.q',
    MHSA_K: 'mhsa.k',
    MHSA_V: 'mhsa.v',
    MHSA_HEAD: 'mhsa.head',
    MHSA_SCORE: 'mhsa.score',
    MHSA_MASK: 'mhsa.mask',
    MHSA_POST: 'mhsa.post',
    MHSA_HEAD_OUTPUT: 'mhsa.head-output',
    MLP: 'mlp',
    LOGITS: 'logits',
    RESIDUAL: 'residual',
    CONNECTOR_NEUTRAL: 'connector.neutral',
    CONNECTOR_Q: 'connector.q',
    CONNECTOR_K: 'connector.k',
    CONNECTOR_V: 'connector.v',
    CONNECTOR_POST: 'connector.post'
});

const VIEW2D_VISUAL_TOKENS = Object.freeze({
    styleKeys: VIEW2D_STYLE_KEYS,
    palette: Object.freeze({
        q: hexToCss(MHA_FINAL_Q_COLOR),
        k: hexToCss(MHA_FINAL_K_COLOR),
        v: hexToCss(MHA_FINAL_V_COLOR),
        mlp: hexToCss(FINAL_MLP_COLOR),
        residual: hexToCss(PREVIEW_TRAIL_COLOR),
        neutral: 'rgb(243, 246, 251)',
        text: 'rgba(235, 239, 247, 0.960)',
        mutedText: 'rgba(206, 213, 224, 0.820)',
        border: 'rgba(193, 201, 216, 0.260)',
        panelBackground: 'rgba(7, 9, 14, 0.960)',
        sceneBackground: 'rgba(4, 6, 10, 0.980)'
    }),
    dimming: Object.freeze({
        inactiveOpacity: 0.18,
        inactiveFilter: 'saturate(0.06) brightness(0.62) grayscale(0.78)',
        activeOpacity: 1
    }),
    connector: Object.freeze({
        strokeWidth: 2,
        glowWidth: 6,
        defaultGap: 10
    }),
    matrix: Object.freeze({
        cornerRadius: 12,
        cardInset: 10,
        borderWidth: 1
    }),
    typography: Object.freeze({
        labelFontSize: 12,
        captionFontSize: 11,
        operatorFontSize: 20
    }),
    styles: Object.freeze({
        [VIEW2D_STYLE_KEYS.SCENE]: Object.freeze({
            fill: 'rgba(4, 6, 10, 0.980)'
        }),
        [VIEW2D_STYLE_KEYS.LABEL]: Object.freeze({
            color: 'rgba(235, 239, 247, 0.960)'
        }),
        [VIEW2D_STYLE_KEYS.CAPTION]: Object.freeze({
            color: 'rgba(206, 213, 224, 0.820)'
        }),
        [VIEW2D_STYLE_KEYS.OPERATOR]: Object.freeze({
            color: 'rgba(243, 246, 251, 0.940)'
        }),
        [VIEW2D_STYLE_KEYS.EMBEDDING_TOKEN]: Object.freeze({
            accent: 'rgba(110, 176, 255, 0.920)'
        }),
        [VIEW2D_STYLE_KEYS.EMBEDDING_POSITION]: Object.freeze({
            accent: 'rgba(146, 244, 208, 0.920)'
        }),
        [VIEW2D_STYLE_KEYS.LAYER_NORM]: Object.freeze({
            accent: 'rgba(236, 242, 251, 0.900)'
        }),
        [VIEW2D_STYLE_KEYS.MATRIX_INPUT]: Object.freeze({
            accent: 'rgba(222, 228, 238, 0.900)'
        }),
        [VIEW2D_STYLE_KEYS.MATRIX_WEIGHT]: Object.freeze({
            accent: 'rgba(243, 246, 251, 0.880)'
        }),
        [VIEW2D_STYLE_KEYS.MATRIX_BIAS]: Object.freeze({
            accent: 'rgba(243, 246, 251, 0.860)'
        }),
        [VIEW2D_STYLE_KEYS.OUTPUT_PROJECTION]: Object.freeze({
            accent: 'rgba(196, 214, 248, 0.920)'
        }),
        [VIEW2D_STYLE_KEYS.MHSA_Q]: Object.freeze({
            accent: hexToCss(MHA_FINAL_Q_COLOR)
        }),
        [VIEW2D_STYLE_KEYS.MHSA_K]: Object.freeze({
            accent: hexToCss(MHA_FINAL_K_COLOR)
        }),
        [VIEW2D_STYLE_KEYS.MHSA_V]: Object.freeze({
            accent: hexToCss(MHA_FINAL_V_COLOR)
        }),
        [VIEW2D_STYLE_KEYS.MHSA_HEAD]: Object.freeze({
            accent: 'rgba(173, 198, 255, 0.920)'
        }),
        [VIEW2D_STYLE_KEYS.MHSA_SCORE]: Object.freeze({
            accent: 'rgba(243, 246, 251, 0.940)'
        }),
        [VIEW2D_STYLE_KEYS.MHSA_MASK]: Object.freeze({
            accent: 'rgba(8, 10, 14, 0.980)'
        }),
        [VIEW2D_STYLE_KEYS.MHSA_POST]: Object.freeze({
            accent: 'rgba(243, 246, 251, 0.940)'
        }),
        [VIEW2D_STYLE_KEYS.MHSA_HEAD_OUTPUT]: Object.freeze({
            accent: 'rgba(243, 246, 251, 0.940)'
        }),
        [VIEW2D_STYLE_KEYS.MLP]: Object.freeze({
            accent: hexToCss(FINAL_MLP_COLOR)
        }),
        [VIEW2D_STYLE_KEYS.LOGITS]: Object.freeze({
            accent: 'rgba(242, 217, 107, 0.940)'
        }),
        [VIEW2D_STYLE_KEYS.RESIDUAL]: Object.freeze({
            accent: hexToCss(PREVIEW_TRAIL_COLOR)
        }),
        [VIEW2D_STYLE_KEYS.CONNECTOR_NEUTRAL]: Object.freeze({
            stroke: 'rgba(243, 246, 251, 0.940)'
        }),
        [VIEW2D_STYLE_KEYS.CONNECTOR_Q]: Object.freeze({
            stroke: hexToCss(MHA_FINAL_Q_COLOR)
        }),
        [VIEW2D_STYLE_KEYS.CONNECTOR_K]: Object.freeze({
            stroke: hexToCss(MHA_FINAL_K_COLOR)
        }),
        [VIEW2D_STYLE_KEYS.CONNECTOR_V]: Object.freeze({
            stroke: hexToCss(MHA_FINAL_V_COLOR)
        }),
        [VIEW2D_STYLE_KEYS.CONNECTOR_POST]: Object.freeze({
            stroke: 'rgba(243, 246, 251, 0.940)'
        })
    })
});

export function resolveView2dVisualTokens() {
    return VIEW2D_VISUAL_TOKENS;
}

export function resolveView2dStyle(styleKey) {
    return VIEW2D_VISUAL_TOKENS.styles[styleKey] || null;
}
