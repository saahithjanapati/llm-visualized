import {
    MHA_FINAL_K_COLOR,
    MHA_OUTPUT_PROJECTION_MATRIX_COLOR,
    MHA_FINAL_Q_COLOR,
    MHA_FINAL_V_COLOR
} from '../../animations/LayerAnimationConstants.js';
import {
    FINAL_MLP_COLOR,
    PREVIEW_TRAIL_COLOR
} from '../../ui/selectionPanelConstants.js';
import { GPT2_LAYER_VISUAL_TUNING } from '../../utils/visualTuningProfiles.js';

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

function mixHexValues(fromHex = 0xFFFFFF, toHex = 0xFFFFFF, amount = 0.5) {
    const t = Math.max(0, Math.min(1, Number.isFinite(amount) ? amount : 0.5));
    const [fromR, fromG, fromB] = hexToRgb(fromHex);
    const [toR, toG, toB] = hexToRgb(toHex);
    const mix = (from, to) => Math.round(from + ((to - from) * t));
    return (mix(fromR, toR) << 16) | (mix(fromG, toG) << 8) | mix(fromB, toB);
}

function buildAttentionProjectionStyle(baseHex = 0xFFFFFF, {
    hotspotMix = 0.46,
    shadowMix = 0.16,
    disableGlow = false
} = {}) {
    return Object.freeze({
        accent: hexToCss(mixHexValues(baseHex, 0xFFFFFF, 0.14), 1),
        fill: `linear-gradient(140deg, ${hexToCss(mixHexValues(baseHex, 0xFFFFFF, 0.34), 0.82)} 0%, ${hexToCss(mixHexValues(baseHex, 0xFFFFFF, 0.12), 0.94)} 38%, ${hexToCss(baseHex, 1)} 70%, ${hexToCss(mixHexValues(baseHex, 0x000000, shadowMix), 0.9)} 100%)`,
        stroke: hexToCss(mixHexValues(baseHex, 0xFFFFFF, 0.24), 0.98),
        cardGlowColor: disableGlow ? null : hexToCss(baseHex, 0.42),
        cardGlowOpacity: disableGlow ? 0 : 0.42,
        cardGlowBlur: disableGlow ? 0 : 30,
        cardHotspotColor: hexToCss(mixHexValues(baseHex, 0xFFFFFF, hotspotMix), 0.32),
        cardInnerGlowColor: hexToCss(baseHex, 0.18),
        cardSheenColor: 'rgba(255,255,255,0.16)',
        cardEdgeHighlight: 'rgba(255,255,255,0.38)'
    });
}

function buildStableMhsaHeadFill() {
    const q = hexToCss(MHA_FINAL_Q_COLOR, 0.98);
    const qBridge = hexToCss(mixHexValues(MHA_FINAL_Q_COLOR, MHA_FINAL_K_COLOR, 0.38), 0.98);
    const k = hexToCss(MHA_FINAL_K_COLOR, 0.98);
    const kBridge = hexToCss(mixHexValues(MHA_FINAL_K_COLOR, MHA_FINAL_V_COLOR, 0.34), 0.98);
    const v = hexToCss(MHA_FINAL_V_COLOR, 0.98);
    return `linear-gradient(90deg, ${q} 0%, ${q} 28%, ${qBridge} 38%, ${k} 50%, ${kBridge} 62%, ${v} 72%, ${v} 100%)`;
}

export const VIEW2D_STYLE_KEYS = Object.freeze({
    SCENE: 'scene',
    LABEL: 'label',
    LABEL_DARK: 'label.dark',
    RESIDUAL_LABEL: 'residual.label',
    CAPTION: 'caption',
    OPERATOR: 'operator',
    EMBEDDING_TOKEN: 'embedding.token',
    EMBEDDING_POSITION: 'embedding.position',
    EMBEDDING_TOKEN_STREAM: 'embedding.token-stream',
    EMBEDDING_POSITION_STREAM: 'embedding.position-stream',
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
    MHSA_HEAD_DETAIL_FRAME: 'mhsa.head-detail-frame',
    CONCATENATE: 'concatenate',
    MLP: 'mlp',
    LOGITS: 'logits',
    RESIDUAL: 'residual',
    RESIDUAL_ADD: 'residual.add',
    RESIDUAL_ADD_SYMBOL: 'residual.add-symbol',
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
        neutral: 'rgba(241, 244, 248, 0.96)',
        text: 'rgba(235, 239, 247, 0.960)',
        mutedText: 'rgba(206, 213, 224, 0.820)',
        border: 'rgba(241, 244, 248, 0.54)',
        panelBackground: 'rgba(255, 255, 255, 0.055)',
        sceneBackground: 'rgba(0, 0, 0, 0)'
    }),
    dimming: Object.freeze({
        inactiveOpacity: 0.18,
        inactiveFilter: 'saturate(0.06) brightness(0.62) grayscale(0.78)',
        activeOpacity: 1
    }),
    connector: Object.freeze({
        strokeWidth: 1.45,
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
        [VIEW2D_STYLE_KEYS.LABEL_DARK]: Object.freeze({
            color: 'rgba(16, 20, 28, 0.940)'
        }),
        [VIEW2D_STYLE_KEYS.RESIDUAL_LABEL]: Object.freeze({
            color: 'rgba(0, 0, 0, 0.960)'
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
        [VIEW2D_STYLE_KEYS.EMBEDDING_TOKEN_STREAM]: Object.freeze({
            accent: 'rgba(54, 132, 255, 0.980)',
            fill: 'linear-gradient(132deg, rgba(134, 187, 255, 0.94) 0%, rgba(54, 132, 255, 0.98) 44%, rgba(18, 78, 208, 0.99) 100%)',
            stroke: 'rgba(236, 246, 255, 0.96)',
            cardGlowColor: 'rgba(36, 108, 240, 0.42)',
            cardGlowOpacity: 0.34,
            cardGlowBlur: 26,
            cardHotspotColor: 'rgba(255,255,255,0.18)',
            cardInnerGlowColor: 'rgba(72, 146, 255, 0.12)',
            cardSheenColor: 'rgba(255,255,255,0.14)',
            cardEdgeHighlight: 'rgba(255,255,255,0.30)'
        }),
        [VIEW2D_STYLE_KEYS.EMBEDDING_POSITION_STREAM]: Object.freeze({
            accent: 'rgba(70, 198, 134, 0.980)',
            fill: 'linear-gradient(132deg, rgba(168, 244, 198, 0.94) 0%, rgba(70, 198, 134, 0.97) 44%, rgba(24, 132, 82, 0.99) 100%)',
            stroke: 'rgba(234, 255, 243, 0.96)',
            cardGlowColor: 'rgba(48, 170, 112, 0.34)',
            cardGlowOpacity: 0.28,
            cardGlowBlur: 22,
            cardHotspotColor: 'rgba(255,255,255,0.16)',
            cardInnerGlowColor: 'rgba(88, 216, 150, 0.10)',
            cardSheenColor: 'rgba(255,255,255,0.12)',
            cardEdgeHighlight: 'rgba(255,255,255,0.30)'
        }),
        [VIEW2D_STYLE_KEYS.LAYER_NORM]: Object.freeze({
            accent: hexToCss(GPT2_LAYER_VISUAL_TUNING.layerNorm.finalColor, 1),
            fill: `linear-gradient(140deg, ${hexToCss(0xFFFFFF, 0.62)} 0%, ${hexToCss(0xFFFFFF, 0.86)} 52%, ${hexToCss(0xDDE6F4, 0.64)} 100%)`,
            stroke: hexToCss(GPT2_LAYER_VISUAL_TUNING.layerNorm.finalColor, 1),
            cardGlowColor: hexToCss(GPT2_LAYER_VISUAL_TUNING.layerNorm.finalColor, 0.34),
            cardGlowOpacity: 0.34,
            cardGlowBlur: 24,
            cardHotspotColor: 'rgba(255,255,255,0.26)',
            cardInnerGlowColor: 'rgba(255,255,255,0.12)',
            cardSheenColor: 'rgba(255,255,255,0.24)',
            cardEdgeHighlight: 'rgba(255,255,255,0.54)'
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
            accent: hexToCss(MHA_OUTPUT_PROJECTION_MATRIX_COLOR, 1),
            fill: `linear-gradient(138deg, ${hexToCss(mixHexValues(MHA_OUTPUT_PROJECTION_MATRIX_COLOR, 0xFFFFFF, 0.34), 0.78)} 0%, ${hexToCss(MHA_OUTPUT_PROJECTION_MATRIX_COLOR, 1)} 56%, ${hexToCss(mixHexValues(MHA_OUTPUT_PROJECTION_MATRIX_COLOR, 0x000000, 0.08), 0.82)} 100%)`,
            stroke: hexToCss(mixHexValues(MHA_OUTPUT_PROJECTION_MATRIX_COLOR, 0xFFFFFF, 0.10), 1),
            cardGlowColor: hexToCss(MHA_OUTPUT_PROJECTION_MATRIX_COLOR, 0.46),
            cardGlowOpacity: 0.42,
            cardGlowBlur: 34,
            cardHotspotColor: hexToCss(mixHexValues(MHA_OUTPUT_PROJECTION_MATRIX_COLOR, 0xFFFFFF, 0.38), 0.30),
            cardInnerGlowColor: hexToCss(MHA_OUTPUT_PROJECTION_MATRIX_COLOR, 0.18),
            cardSheenColor: 'rgba(255,255,255,0.18)',
            cardEdgeHighlight: 'rgba(255,255,255,0.34)'
        }),
        [VIEW2D_STYLE_KEYS.MHSA_Q]: buildAttentionProjectionStyle(MHA_FINAL_Q_COLOR, {
            disableGlow: true
        }),
        [VIEW2D_STYLE_KEYS.MHSA_K]: buildAttentionProjectionStyle(MHA_FINAL_K_COLOR, {
            disableGlow: true,
            hotspotMix: 0.42,
            shadowMix: 0.14
        }),
        [VIEW2D_STYLE_KEYS.MHSA_V]: buildAttentionProjectionStyle(MHA_FINAL_V_COLOR, {
            disableGlow: true,
            hotspotMix: 0.34,
            shadowMix: 0.12
        }),
        [VIEW2D_STYLE_KEYS.MHSA_HEAD]: Object.freeze({
            accent: hexToCss(MHA_FINAL_Q_COLOR, 0.98),
            fill: buildStableMhsaHeadFill(),
            stroke: 'rgba(148, 164, 186, 0.82)',
            disableCardSurfaceEffects: true,
            cardGlowColor: hexToCss(mixHexValues(MHA_FINAL_Q_COLOR, 0x000000, 0.10), 0.30),
            cardGlowOpacity: 0.22,
            cardGlowBlur: 18,
            cardHotspotColor: 'rgba(255,255,255,0.02)',
            cardInnerGlowColor: 'rgba(0,0,0,0.18)',
            cardSheenColor: 'rgba(255,255,255,0.03)',
            cardEdgeHighlight: 'rgba(255,255,255,0.16)'
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
        [VIEW2D_STYLE_KEYS.MHSA_HEAD_OUTPUT]: buildAttentionProjectionStyle(MHA_FINAL_V_COLOR, {
            hotspotMix: 0.28,
            shadowMix: 0.08
        }),
        [VIEW2D_STYLE_KEYS.MHSA_HEAD_DETAIL_FRAME]: Object.freeze({
            accent: 'rgba(255, 255, 255, 0.98)',
            fill: 'rgba(0, 0, 0, 0.985)',
            stroke: 'rgba(255, 255, 255, 0.10)',
            disableCardSurfaceEffects: true
        }),
        [VIEW2D_STYLE_KEYS.CONCATENATE]: Object.freeze({
            accent: 'rgba(255, 255, 255, 0.98)',
            fill: 'linear-gradient(145deg, rgba(6, 8, 12, 1) 0%, rgba(6, 8, 12, 1) 100%)',
            stroke: 'rgba(255, 255, 255, 0.92)',
            disableCardSurfaceEffects: true
        }),
        [VIEW2D_STYLE_KEYS.MLP]: Object.freeze({
            accent: hexToCss(FINAL_MLP_COLOR, 0.98),
            fill: `linear-gradient(138deg, ${hexToCss(mixHexValues(FINAL_MLP_COLOR, 0xFFFFFF, 0.28), 0.82)} 0%, ${hexToCss(FINAL_MLP_COLOR, 0.98)} 56%, ${hexToCss(mixHexValues(FINAL_MLP_COLOR, 0x000000, 0.06), 0.84)} 100%)`,
            stroke: hexToCss(mixHexValues(FINAL_MLP_COLOR, 0xFFFFFF, 0.10), 1),
            cardGlowColor: hexToCss(FINAL_MLP_COLOR, 0.44),
            cardGlowOpacity: 0.42,
            cardGlowBlur: 34,
            cardHotspotColor: hexToCss(mixHexValues(FINAL_MLP_COLOR, 0xFFFFFF, 0.34), 0.30),
            cardInnerGlowColor: hexToCss(FINAL_MLP_COLOR, 0.18),
            cardSheenColor: 'rgba(255,255,255,0.16)',
            cardEdgeHighlight: 'rgba(255,255,255,0.32)'
        }),
        [VIEW2D_STYLE_KEYS.LOGITS]: Object.freeze({
            accent: 'rgba(242, 217, 107, 0.940)'
        }),
        [VIEW2D_STYLE_KEYS.RESIDUAL]: Object.freeze({
            accent: hexToCss(PREVIEW_TRAIL_COLOR, 0.98),
            fill: `linear-gradient(138deg, ${hexToCss(mixHexValues(PREVIEW_TRAIL_COLOR, 0xFFFFFF, 0.24), 0.52)} 0%, ${hexToCss(PREVIEW_TRAIL_COLOR, 0.78)} 54%, ${hexToCss(mixHexValues(PREVIEW_TRAIL_COLOR, 0x000000, 0.08), 0.58)} 100%)`,
            stroke: 'rgba(236, 244, 255, 0.98)',
            cardGlowColor: hexToCss(PREVIEW_TRAIL_COLOR, 0.28),
            cardGlowOpacity: 0.24,
            cardGlowBlur: 22,
            cardHotspotColor: 'rgba(255,255,255,0.16)',
            cardInnerGlowColor: hexToCss(PREVIEW_TRAIL_COLOR, 0.12),
            cardSheenColor: 'rgba(255,255,255,0.12)',
            cardEdgeHighlight: 'rgba(255,255,255,0.28)'
        }),
        [VIEW2D_STYLE_KEYS.RESIDUAL_ADD]: Object.freeze({
            accent: 'rgba(160, 166, 176, 0.98)',
            fill: 'linear-gradient(145deg, rgba(12, 14, 18, 1) 0%, rgba(12, 14, 18, 1) 100%)',
            stroke: 'rgba(176, 182, 191, 0.92)',
            cardGlowColor: 'rgba(120, 126, 138, 0.10)',
            cardGlowOpacity: 0.08,
            cardGlowBlur: 10,
            cardHotspotColor: 'rgba(0,0,0,0.0)',
            cardInnerGlowColor: 'rgba(0,0,0,0.0)',
            cardSheenColor: 'rgba(255,255,255,0.0)',
            cardEdgeHighlight: 'rgba(196, 201, 209, 0.18)'
        }),
        [VIEW2D_STYLE_KEYS.RESIDUAL_ADD_SYMBOL]: Object.freeze({
            color: 'rgba(255, 255, 255, 0.98)'
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
            stroke: 'rgba(225, 238, 255, 0.82)'
        })
    })
});

export function resolveView2dVisualTokens() {
    return VIEW2D_VISUAL_TOKENS;
}

export function resolveView2dStyle(styleKey) {
    return VIEW2D_VISUAL_TOKENS.styles[styleKey] || null;
}
