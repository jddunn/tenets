(function(){
  'use strict';

  // Toast utilities
  function ensureToastContainer(){
    var id = 'tenets-toast-container';
    var el = document.getElementById(id);
    if(!el){
      el = document.createElement('div');
      el.id = id;
      document.body.appendChild(el);
    }
    return el;
  }

  function showToast(message, variant){
    try{
      var container = ensureToastContainer();
      // Remove any existing toasts to prevent duplicates
      var existing = container.querySelectorAll('.tenets-toast');
      existing.forEach(function(t){ t.remove(); });

      var toast = document.createElement('div');
      toast.className = 'tenets-toast' + (variant ? (' tenets-toast--'+variant) : '');
      toast.setAttribute('role','status');
      toast.setAttribute('aria-live','polite');
      toast.textContent = message || 'Copied to clipboard';
      container.appendChild(toast);
      // Force reflow for animation
      void toast.offsetWidth;
      toast.classList.add('visible');
      setTimeout(function(){
        toast.classList.remove('visible');
        setTimeout(function(){ toast.remove(); }, 220);
      }, 1600);
    }catch(e){ /* noop */ }
  }

  // Clipboard helpers
  function writeClipboard(text){
    if(!text) return Promise.reject(new Error('No text to copy'));
    if(navigator.clipboard && window.isSecureContext){
      return navigator.clipboard.writeText(text);
    }
    // Fallback for insecure context or older browsers
    return new Promise(function(resolve, reject){
      try{
        var ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.top = '-1000px';
        ta.style.left = '-1000px';
        ta.setAttribute('readonly','');
        document.body.appendChild(ta);
        ta.select();
        ta.setSelectionRange(0, ta.value.length);
        var ok = document.execCommand('copy');
        document.body.removeChild(ta);
        ok ? resolve() : reject(new Error('execCommand copy failed'));
      }catch(err){ reject(err); }
    });
  }

  function extractNearbyText(btn){
    // Priority: data attribute(s), then nearest code/pre in common wrappers
    var txt = btn.getAttribute('data-clipboard-text') || btn.getAttribute('data-text');
    if(txt){ return txt; }
    var scope = btn.closest('.install-command, .highlight, pre, code, .md-typeset');
    if(scope){
      var el = scope.querySelector('code, pre code, pre');
      if(el && el.textContent) return el.textContent.trim();
    }
    // Try immediate previous code sibling
    var prev = btn.previousElementSibling;
    if(prev && (/^(code|pre)$/i).test(prev.tagName)) return prev.textContent.trim();
    return null;
  }

  function onCustomCopyClick(e){
    var btn = e.target.closest('.copy-btn, .copy-btn-mini, .copy-btn-inline');
    if(!btn) return;
    e.preventDefault();
    var text = extractNearbyText(btn);
    if(!text){ showToast('Nothing to copy', 'error'); return; }

    // Add visual feedback for mini button
    if(btn.classList.contains('copy-btn-mini')) {
      btn.classList.add('copied');
      setTimeout(function(){
        btn.classList.remove('copied');
      }, 2000);
    }

    writeClipboard(text)
      .then(function(){ showToast('Copied to clipboard'); })
      .catch(function(){ showToast('Copy failed', 'error'); });
  }

  function onMaterialCopyClick(e){
    var btn = e.target.closest('button.md-clipboard');
    if(!btn) return;

    // Hide Material's own toast notification if it exists
    var materialToast = document.querySelector('.md-clipboard__message');
    if(materialToast) {
      materialToast.style.display = 'none';
    }

    // Show our custom toast
    setTimeout(function(){
      showToast('Copied to clipboard');
      // Keep Material's toast hidden
      var mt = document.querySelector('.md-clipboard__message');
      if(mt) mt.style.display = 'none';
    }, 40);
  }

  function wire(){
    if(!document.__tenetsCopyWired){
      document.addEventListener('click', onCustomCopyClick);
      document.addEventListener('click', onMaterialCopyClick);
      document.__tenetsCopyWired = true;
    }
  }

  function init(){ wire(); }

  if(document.readyState === 'loading'){
    document.addEventListener('DOMContentLoaded', init);
  } else { init(); }
  // No MutationObserver: event delegation handles dynamic content too
})();
