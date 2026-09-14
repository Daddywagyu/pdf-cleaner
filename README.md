# PDF Exam Cleaner

Browser-only PDF cleaning tool for old exam papers. PDF files are processed locally in the browser and are not uploaded to a backend.

## Run locally

```bash
npm install
npm run dev
```

## Build

```bash
npm run build
```

## GitHub Pages

Pushes to `main` build and deploy automatically through GitHub Actions. Enable **Settings → Pages → Source: GitHub Actions** if GitHub Pages is not already enabled.

The application uses Vite, PDF.js and pdf-lib. Export currently creates a new flattened PDF from the cleaned page images, preserving the original page dimensions but not the original text layer.
