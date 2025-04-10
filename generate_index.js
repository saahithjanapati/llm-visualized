const fs = require('fs');
const path = require('path');

const testsDir = path.join(__dirname, 'tests');
const templatePath = path.join(__dirname, 'index.template.html');
const outputPath = path.join(__dirname, 'index.html');
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

    // Read the tests directory
    const files = fs.readdirSync(testsDir);
    console.log(`Found files: ${files.join(', ')}`);

    const testLinks = files
        .filter(file => file.endsWith('.html')) // Only include HTML files
        .map(file => {
            const displayName = generateDisplayName(file);
            const filePath = path.join('tests', file);
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