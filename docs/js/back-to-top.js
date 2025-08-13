(function() {
  'use strict';

  var THRESHOLD = 300; // px (show after meaningful scroll)
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

  // Text-based icon for maximum contrast/visibility (with SVG fallback appended)
  var textIcon = '<span class="back-to-top-icon" aria-hidden="true">▲</span>';
  var svgFallback = '\n    <svg class="back-to-top-svg-fallback" width="24" height="24" viewBox="0 0 24 24" aria-hidden="true" focusable="false" role="img">\n      <path class="filled" d="M6 14L12 8L18 14H6Z"/>\n      <rect class="filled" x="11" y="14" width="2" height="6" rx="1"/>\n    </svg>\n  ';

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
  btn.innerHTML = textIcon + svgFallback;
      document.body.appendChild(btn);
    } else {
      // Adopt existing markup and normalize
      btn.type = 'button';
      btn.classList.add('back-to-top');
      btn.setAttribute('aria-label', 'Back to top');
  if (!btn.querySelector('.back-to-top-icon')) btn.insertAdjacentHTML('afterbegin', textIcon);
  if (!btn.querySelector('.back-to-top-svg-fallback')) btn.insertAdjacentHTML('beforeend', svgFallback);
    }
    return btn;
  }

  function init() {
    var button = createOrAdoptButton();

  // Essential styles inline to ensure fixed positioning; centered at bottom
  var computed = window.getComputedStyle(button);
  var cssApplied = computed && computed.position === 'fixed';
  button.style.setProperty('position', 'fixed', 'important');
  button.style.setProperty('left', '50%', 'important');
  button.style.removeProperty('right');
  button.style.removeProperty('top');
  button.style.setProperty('bottom', '16px', 'important');
  button.style.setProperty('transform', 'translate(-50%, 0)', 'important');
  button.style.setProperty('display', 'inline-flex', 'important');
  // Baseline dimensions/appearance (let CSS handle colors/theme)
  button.style.width = '48px';
  button.style.height = '48px';
  button.style.alignItems = 'center';
  button.style.justifyContent = 'center';
  button.style.borderRadius = '9999px';
  button.style.setProperty('z-index', '13001', 'important');

    // Track potential nested scroll containers used by the theme
    var containerSelectors = ['main', '.md-content', '.md-main', '.md-main__inner', '.md-content__inner'];
    var attached = new WeakSet();
    var scrollContainers = [];

    function findScrollContainers() {
      var set = new Set();
      containerSelectors.forEach(function(sel){
        var els = document.querySelectorAll(sel);
        els.forEach(function(el){
          if (el && el !== document.body && el !== document.documentElement) set.add(el);
        });
      });
      scrollContainers = Array.from(set);
      // Attach scroll listeners to any new containers
      scrollContainers.forEach(function(el){
        if (!attached.has(el)) {
          el.addEventListener('scroll', onScroll, { passive: true });
          attached.add(el);
        }
      });
    }

    function currentScrollTop() {
      // Prefer scrollingElement for cross-browser accuracy, fallback to window
      var se = document.scrollingElement || document.documentElement;
      var st = se ? se.scrollTop : 0;
      if (st === 0 && typeof window.pageYOffset === 'number') st = window.pageYOffset;
      // Also consider nested containers; take the max scrollTop among them
      if (scrollContainers && scrollContainers.length) {
        for (var i = 0; i < scrollContainers.length; i++) {
          var el = scrollContainers[i];
          if (el && typeof el.scrollTop === 'number') {
            if (el.scrollTop > st) st = el.scrollTop;
          }
        }
      }
      return st;
    }

    function toggleVisibility() {
      var st = currentScrollTop();
      var show = st > THRESHOLD;
      if (show) button.classList.add('visible');
      else button.classList.remove('visible');
    }

    var onScroll = throttle(toggleVisibility, 100);
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll);
    window.addEventListener('hashchange', onScroll);
    // Initial container discovery and listener wiring
    findScrollContainers();
    // Re-discover containers when Material swaps content dynamically
    var observer = new MutationObserver(function(mutations){
      var needsUpdate = false;
      for (var i = 0; i < mutations.length; i++) {
        var m = mutations[i];
        if (m.type === 'childList') { needsUpdate = true; break; }
      }
      if (needsUpdate) {
        findScrollContainers();
        onScroll();
      }
    });
    observer.observe(document.body, { childList: true, subtree: true });
  // Initial state
  button.classList.remove('visible');
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
