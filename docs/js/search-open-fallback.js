// Ensures a __search checkbox exists for Material; helps our open logic.
(function(){
  try {
    // If Material didn't render the toggle yet, add a compatible hidden one.
    if (!document.getElementById('__search')) {
      var cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.id = '__search';
      cb.setAttribute('data-md-toggle', 'search');
      cb.className = 'md-toggle';
      cb.autocomplete = 'off';
      cb.style.position = 'absolute';
      cb.style.left = '-9999px';
      document.body.appendChild(cb);
    }
  } catch (e) {}
})();
