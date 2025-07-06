// docs/javascripts/katex-render.js

function renderAllMath() {
  if (!window.renderMathInElement) return;

  renderMathInElement(document.body, {
    // tell auto-render which delimiters to look for
    delimiters: [
      { left: '$$', right: '$$', display: true },
      { left: '\\[', right: '\\]', display: true },
      { left: '$',  right: '$',  display: false },
      { left: '\\(', right: '\\)', display: false }
    ],
    throwOnError: false  // so bad TeX won’t break your page
  });
}

// Initial page load
document.addEventListener('DOMContentLoaded', renderAllMath);

// For older Material (<v8)
if (window.document$ && document$.subscribe) {
  document$.subscribe(renderAllMath);
}

// For newer Material (v8+)
window.addEventListener('navchange', renderAllMath);
