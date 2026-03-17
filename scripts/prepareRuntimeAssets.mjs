import { readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(scriptDir, '..');
const sourceDir = resolve(projectRoot, 'public');
const captureFiles = ['capture.json'];

async function validateCaptureAsset(fileName) {
    const sourcePath = resolve(sourceDir, fileName);
    const rawJson = await readFile(sourcePath, 'utf8');
    JSON.parse(rawJson);
}

await Promise.all(captureFiles.map(validateCaptureAsset));
