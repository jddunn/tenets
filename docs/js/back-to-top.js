(function() {
  'use strict';

  var THRESHOLD = 300; // px
  var BUTTON_ID = 'back-to-top';

  function throttle(fn, limit) {
    var inThrottle, lastFunc, lastTime;
    return function() {
      var context = this, args = arguments;
      if (!inThrottle) {
        fn.apply(context, args);
        lastTime = Date.now();
        inThrottle = true;
      } else {
        clearTimeout(lastFunc);
        lastFunc = setTimeout(function() {
          if ((Date.now() - lastTime) >= limit) {
            fn.apply(context, args);
            lastTime = Date.now();
          }
        }, Math.max(limit - (Date.now() - lastTime), 0));
      }
    };
  }

  var svgIcon = '\n    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">\n      <path d="M12 19V5"/>\n      <path d="M5 12l7-7 7 7"/>\n    </svg>\n  ';

  function ensureSingleButton() {
    var nodes = document.querySelectorAll('#' + BUTTON_ID);
    if (nodes.length > 1) {
      // Keep the first, remove the rest
      for (var i = 1; i < nodes.length; i++) nodes[i].parentNode && nodes[i].parentNode.removeChild(nodes[i]);
    }
    return document.getElementById(BUTTON_ID);
  }

  function createOrAdoptButton() {
    var btn = ensureSingleButton();
    if (!btn) {
      btn = document.createElement('button');
      btn.id = BUTTON_ID;
      btn.type = 'button';
      btn.className = 'back-to-top';
      btn.setAttribute('aria-label', 'Back to top');
      btn.innerHTML = svgIcon;
      document.body.appendChild(btn);
    } else {
      // Adopt existing markup and normalize
      btn.type = 'button';
      btn.classList.add('back-to-top');
      btn.setAttribute('aria-label', 'Back to top');
      if (!btn.querySelector('svg')) btn.innerHTML = svgIcon;
    }
    return btn;
  }

  function init() {
    var button = createOrAdoptButton();

    function toggleVisibility() {
      if (window.pageYOffset > THRESHOLD) {
        button.classList.add('visible');
      } else {
        button.classList.remove('visible');
      }
    }

    var onScroll = throttle(toggleVisibility, 100);
    window.addEventListener('scroll', onScroll);
    toggleVisibility();

    function scrollToTop(e) {
      if (e) e.preventDefault();
      try {
        window.scrollTo({ top: 0, behavior: 'smooth' });
      } catch (_) {
        // Fallback for older browsers
        document.documentElement.scrollTop = 0;
        document.body.scrollTop = 0;
      }
    }

    button.addEventListener('click', scrollToTop);
    button.addEventListener('keydown', function(e) {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        scrollToTop(e);
      }
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
