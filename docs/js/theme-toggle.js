// Minimal native theme toggle relying on Material's built-in palette inputs
(function(){
  const LS_KEY = 'md-color-scheme';
  const htmlEl = document.documentElement;

  function inputs(){ return Array.from(document.querySelectorAll('input[id^="__palette_"].md-toggle')); }

  function apply(s){
    if(!s) return;
    htmlEl.setAttribute('data-md-color-scheme', s);
    if(document.body && document.body.hasAttribute('data-md-color-scheme')) document.body.removeAttribute('data-md-color-scheme');
    try{localStorage.setItem(LS_KEY,s);}catch(e){}
  }

  function syncFromAttribute(){
    const ps = inputs();
    const scheme = htmlEl.getAttribute('data-md-color-scheme') || 'default';
    if(ps.length>=2){
      ps[0].checked = scheme === 'default';
      ps[1].checked = scheme === 'slate';
    }
    const opposite = scheme==='default' ? 'Switch to dark mode':'Switch to light mode';
    document.querySelectorAll('label.md-header__button[for^="__palette_"]').forEach(l=>{
      l.title = opposite; l.setAttribute('aria-label', opposite); l.setAttribute('data-tooltip', opposite);
    });
  }

  function onRadioChange(){
    const ps = inputs();
    if(ps.length>=2){
      const scheme = ps[0].checked ? 'default' : (ps[1].checked ? 'slate' : 'default');
      apply(scheme);
      requestAnimationFrame(syncFromAttribute);
    }
  }

  function wire(){
    inputs().forEach(i=>{ if(!i.__wired){ i.__wired=true; i.addEventListener('change', onRadioChange); }});
  }

  // Fallback: direct label click toggles scheme if radios fail
  function wireLabelFallback(){
    document.querySelectorAll('label.md-header__button[for^="__palette_"]').forEach(l => {
      if(l.__fallbackWired) return; l.__fallbackWired = true;
      l.addEventListener('click', () => {
        // Delay to allow native radio toggle; then verify
        setTimeout(()=>{
          const before = htmlEl.getAttribute('data-md-color-scheme');
          const ps = inputs();
          if(ps.length>=2){
            const expected = (ps[0].checked ? 'default' : (ps[1].checked ? 'slate' : null));
            if(expected && expected !== before){ apply(expected); syncFromAttribute(); return; }
          }
          // If radios didn't fire, manually flip
          const flipped = before === 'default' ? 'slate' : 'default';
          apply(flipped); syncFromAttribute();
        }, 60);
      });
    });
  }

  // Observe DOM for late-added radios/labels (navigation.instant, etc.)
  const mo = new MutationObserver(()=>{ wire(); wireLabelFallback(); syncFromAttribute(); });
  mo.observe(document.documentElement, { childList:true, subtree:true });

  function init(){
    const saved = (()=>{ try{return localStorage.getItem(LS_KEY);}catch(e){ return null; }})();
    if(saved) apply(saved); else if(!htmlEl.hasAttribute('data-md-color-scheme')) apply('default');
  wire();
  wireLabelFallback();
  syncFromAttribute();
  // Retry a few times quickly for early load race conditions
  let tries=0; const iv=setInterval(()=>{ tries++; wire(); wireLabelFallback(); syncFromAttribute(); if(tries>30 || inputs().length>=2) clearInterval(iv); },120);
  }

  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
