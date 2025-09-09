const fs = require('fs');
const path = require('path');

// Function to extract vite config inputs, ensuring it's robust
function getViteInputs() {
    try {
        const viteConfigPath = path.join(__dirname, 'vite.config.js');
        const viteConfigContent = fs.readFileSync(viteConfigPath, 'utf8');
        
        // Use a more robust regex to find the input object
        const inputRegex = /input:\s*{[^}]*}/s;
        const match = viteConfigContent.match(inputRegex);

        if (!match) {
            console.warn("Could not find 'input:' block in vite.config.js. Proceeding with all files in tests directory.");
            return null; // Return null if input block not found
        }

        const inputBlock = match[0];
        
        // Extract file paths from the input block
        // This regex looks for resolve(__dirname, 'tests/filename.html')
        const fileRegex = /resolve\(__dirname, 'tests\/([^']+\.html)'\)/g;
        let fileMatch;
        const viteFiles = [];
        while ((fileMatch = fileRegex.exec(inputBlock)) !== null) {
            viteFiles.push(fileMatch[1]);
        }
        console.log('Successfully extracted files from vite.config.js:', viteFiles);
        return viteFiles;
    } catch (err) {
        console.warn('Warning: Could not read or parse vite.config.js. Proceeding with all files in tests directory.', err);
        return null; // Return null in case of other errors
    }
}

const testsDir = path.join(__dirname, 'tests');
const templatePath = path.join(__dirname, 'index.template.html');
// Write the generated index into the tests directory so /tests loads the test selector
const outputPath = path.join(__dirname, 'tests', 'index.html');
const placeholder = '<!-- TEST_LINKS_PLACEHOLDER -->';

console.log(`Scanning directory: ${testsDir}`);

// Function to generate a display name from a filename
function generateDisplayName(filename) {
    // Remove .html extension
    let name = filename.replace(/\.html$/, '');
    // Replace underscores/hyphens with spaces
    name = name.replace(/[_-]/g, ' ');
    // Capitalize first letter of each word
    name = name.split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
    return `Test: ${name}`;
}

try {
    // Read the template file
    let templateContent = fs.readFileSync(templatePath, 'utf8');
    console.log('Template file read successfully.');

    // Get the list of allowed test files from vite.config.js
    const allowedTestFiles = getViteInputs();

    // Read the tests directory
    const files = fs.readdirSync(testsDir);
    console.log(`Found files in ${testsDir}: ${files.join(', ')}`);

    const testLinks = files
        .filter(file => {
            if (!file.endsWith('.html')) return false; // Only include HTML files
            if (file === 'index.html') return false; // Exclude the generated index itself
            if (allowedTestFiles) { // If vite config was parsed successfully
                return allowedTestFiles.includes(file);
            }
            return true; // If vite config parsing failed, include all html files (fallback)
        })
        .map(file => {
            const displayName = generateDisplayName(file);
            const filePath = file; // Links are relative within /tests
            // Ensure forward slashes for web paths, even on Windows
            const webPath = filePath.replace(/\\/g, '/');
            return `        <li><a href="${webPath}">${displayName}</a></li>`;
        })
        .join('\n');

    console.log(`Generated links:\n${testLinks}`);

    // Replace the placeholder in the template
    const finalHtml = templateContent.replace(placeholder, testLinks);

    // Write the final index.html
    fs.writeFileSync(outputPath, finalHtml, 'utf8');
    console.log(`Successfully generated ${outputPath}`);

} catch (err) {
    console.error('Error generating index.html:', err);
    process.exit(1); // Exit with error code
} 