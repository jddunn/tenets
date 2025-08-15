// docs/js/page-flags.js
// Tag the document <body> with flags based on URL to enable scoped CSS behaviors.
(function(){
  function applyFlags(){
    try {
      var path = window.location.pathname || '';
      var body = document.body;
      if (!body) return;
      // API pages: match "/api/" at start of path (directory URLs enabled)
      if (/^\/?api\//.test(path.replace(/^\//, ''))) body.classList.add('is-api');
      else body.classList.remove('is-api');
    } catch(e) { /* noop */ }
  }

  // Initial run
  applyFlags();

  // Re-apply on history navigation (Material navigation.instant uses pushState)
  window.addEventListener('popstate', applyFlags, { passive: true });

  // Re-apply after clicks that may trigger in-page navigation
  document.addEventListener('click', function(){ setTimeout(applyFlags, 50); }, true);
})();
