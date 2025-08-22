// docs/js/page-flags.js
// Tag the document <body> with flags based on URL to enable scoped CSS behaviors.
(function(){
  var lastIsApi = null;
  function applyFlags(){
    try {
      var path = window.location.pathname || '';
      var body = document.body;
      if (!body) return;
      // API pages: match "/api/" at start of path (directory URLs enabled)
      var isApi = /^\/?api\//.test(path.replace(/^\//, ''));
      if (isApi) body.classList.add('is-api'); else body.classList.remove('is-api');
      if (lastIsApi !== isApi) {
        lastIsApi = isApi;
        var evt;
        try {
          evt = new CustomEvent('tenets:page-flags', { detail: { isApi: isApi } });
        } catch(_) {
          evt = document.createEvent('CustomEvent');
          evt.initCustomEvent('tenets:page-flags', true, true, { isApi: isApi });
        }
        document.dispatchEvent(evt);
      }
    } catch(e) { /* noop */ }
  }

  // Initial run
  applyFlags();

  // Re-apply on history navigation (Material navigation.instant uses pushState)
  window.addEventListener('popstate', applyFlags, { passive: true });

  // Re-apply after clicks that may trigger in-page navigation
  document.addEventListener('click', function(){ setTimeout(applyFlags, 50); }, true);
})();
