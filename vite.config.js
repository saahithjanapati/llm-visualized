import { defineConfig } from 'vite';
import { resolve } from 'path';
import fs from 'fs';
import basicSsl from '@vitejs/plugin-basic-ssl'; // Import the SSL plugin

// Custom plugin to generate index.html
function generateIndexPlugin() {
  return {
    name: 'generate-index',
    // Use buildStart hook to ensure it runs before the build
    buildStart() {
      // Resolve paths relative to the project root
      const testsDir = resolve('public/tests');
      const templatePath = resolve('index.template.html');
      const outputPath = resolve('index.html'); // Generate index.html in the root for dev server
      const placeholder = '<!-- TEST_LINKS_PLACEHOLDER -->';

      console.log(`[generateIndexPlugin] Scanning directory: ${testsDir}`);

      function generateDisplayName(filename) {
        let name = filename.replace(/\.html$/, '');
        name = name.replace(/[_-]/g, ' ');
        name = name.split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
        return `Test: ${name}`;
      }

      try {
        // Ensure the template file exists
        if (!fs.existsSync(templatePath)) {
          console.error(`[generateIndexPlugin] Error: Template file not found at ${templatePath}`);
          return; // Stop if template is missing
        }
        const templateContent = fs.readFileSync(templatePath, 'utf8');
        console.log('[generateIndexPlugin] Template file read successfully.');

        // Ensure the tests directory exists before reading
        if (!fs.existsSync(testsDir)) {
           console.warn(`[generateIndexPlugin] Warning: Tests directory not found at ${testsDir}. Creating index.html without test links.`);
           // Create the file with placeholder intact or empty list
           const finalHtml = templateContent.replace(placeholder, '<!-- No tests found -->');
           fs.writeFileSync(outputPath, finalHtml, 'utf8');
           console.log(`[generateIndexPlugin] Generated ${outputPath} (no tests found).`);
           return;
        }

        const files = fs.readdirSync(testsDir);
        console.log(`[generateIndexPlugin] Found files: ${files.join(', ')}`);

        const testLinks = files
          .filter(file => file.endsWith('.html'))
          .map(file => {
            const displayName = generateDisplayName(file);
            // Generate relative paths for the links
            const webPath = `./tests/${file}`; // Relative path for serving from root
            return `        <li><a href="${webPath}">${displayName}</a></li>`;
          })
          .join('\n');

        console.log(`[generateIndexPlugin] Generated links:\n${testLinks}`);

        const finalHtml = templateContent.replace(placeholder, testLinks || '<!-- No HTML tests found -->');
        fs.writeFileSync(outputPath, finalHtml, 'utf8');
        console.log(`[generateIndexPlugin] Successfully generated ${outputPath}`);

      } catch (err) {
        console.error('[generateIndexPlugin] Error generating index.html:', err);
        // Don't throw error during dev, just log it
        // process.exit(1); // Avoid exiting during dev
      }
    }
  };
}


export default defineConfig({
  // Make sure vite serves from the root directory
  root: '.',
  // Specify the public directory explicitly
  publicDir: 'public',
  plugins: [
    generateIndexPlugin(),
    basicSsl() // Add the SSL plugin
 ],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        // Define entry points. Vite automatically finds index.html in the root.
        main: resolve(__dirname, 'index.html'),
        // If you still need app.html as a separate entry point:
        app: resolve(__dirname, 'app.html')
      }
    }
  },
  server: {
    // Enable HTTPS
    https: true,
    // Optional: Open the browser automatically
    open: '/'
  }
}); 