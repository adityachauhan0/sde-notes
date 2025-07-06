// docs/javascripts/mathjax.js

// ➊ MathJax config: process only our arithmatex spans
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    // ignore everything _except_ nodes with class="arithmatex"
    // negative‐lookahead: skip tags unless they have arithmatex
    ignoreHtmlClass: '(?!arithmatex)',
    processHtmlClass: 'arithmatex'
  }
};

// ➋ PJAX hook: retypeset on every instant nav
(function() {
  function rerender() {
    if (window.MathJax && MathJax.typesetPromise) {
      console.log('[mathjax] re-typesetting…');
      MathJax.typesetPromise();
    }
  }

  // first load
  document.addEventListener('DOMContentLoaded', rerender);

  // older Material
  if (window.document$ && document$.subscribe) {
    document$.subscribe(rerender);
  }

  // newer Material
  window.addEventListener('navchange', rerender);
})();
