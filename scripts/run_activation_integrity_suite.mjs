import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

function fail(message, code = 1) {
    console.error(message);
    process.exit(code);
}

function runCommand(command, args, options = {}) {
    const result = spawnSync(command, args, {
        stdio: 'inherit',
        ...options
    });
    if (result.error) {
        fail(result.error.message);
    }
    if (result.status !== 0) {
        process.exit(result.status ?? 1);
    }
}

function resolveTokenWindow() {
    const fallback = {
        promptTokens: [6090, 8217, 892, 30],
        completionTokens: [1849, 5297]
    };
    const captureFixturePath = path.resolve('public/capture.json');
    if (!fs.existsSync(captureFixturePath)) return fallback;

    try {
        const fixture = JSON.parse(fs.readFileSync(captureFixturePath, 'utf8'));
        const promptTokens = Array.isArray(fixture?.meta?.prompt_tokens)
            ? fixture.meta.prompt_tokens.slice(0, 4)
            : [];
        const completionTokens = Array.isArray(fixture?.meta?.completion_tokens)
            ? fixture.meta.completion_tokens.slice(0, 2)
            : [];
        if (promptTokens.length >= 2 && completionTokens.length >= 1) {
            return { promptTokens, completionTokens };
        }
    } catch (_) {
        // Fall back to the baked-in token window below.
    }

    return fallback;
}

const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'activation-integrity-'));
const capturePath = path.join(tempDir, 'capture.json');
const keepCapture = process.argv.includes('--keep-capture') || process.env.ACTIVATION_CAPTURE_KEEP === '1';
const device = process.env.ACTIVATION_INTEGRITY_DEVICE || 'cpu';
const { promptTokens, completionTokens } = resolveTokenWindow();

console.log(`Generating activation capture in ${capturePath}`);
runCommand('python3', [
    '-c',
    [
        'import importlib.util, runpy, sys',
        'real_find_spec = importlib.util.find_spec',
        'def patched(name, *args, **kwargs):',
        "    if name in {'sklearn', 'scipy'}:",
        '        return None',
        '    return real_find_spec(name, *args, **kwargs)',
        'importlib.util.find_spec = patched',
        'script_path = sys.argv[1]',
        'sys.argv = sys.argv[1:]',
        "runpy.run_path(script_path, run_name='__main__')"
    ].join('\n'),
    'scripts/extract_gpt2_data.py',
    '--prompt-tokens', JSON.stringify(promptTokens),
    '--completion-tokens', JSON.stringify(completionTokens),
    '--output', capturePath,
    '--device', device,
    '--residual-stride', '64',
    '--attention-stride', '64',
    '--mlp-stride', '64',
    '--quantisation', 'float16',
    '--round-decimals', '4',
    '--attention-score-round-decimals', '6',
    '--store-embedding-sum',
    '--store-residual-sums',
    '--logit-top-k', '8'
], {
    env: {
        ...process.env,
        PYTHONUNBUFFERED: '1'
    }
});

console.log('Running activation integrity suite');
runCommand(process.execPath, [
    path.resolve('node_modules/vitest/vitest.mjs'),
    'run',
    'tests/activationPipelineIntegrity.test.js'
], {
    env: {
        ...process.env,
        ACTIVATION_CAPTURE_PATH: capturePath
    }
});

if (keepCapture) {
    console.log(`Keeping generated capture at ${capturePath}`);
} else {
    fs.rmSync(tempDir, { recursive: true, force: true });
}
