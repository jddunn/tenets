(function(){
  function setupReveals(root){
    const outputs = root.querySelectorAll('.see-output');
    outputs.forEach(out => {
      // Optional caption support
      try {
        const frame = out.querySelector('.shot-frame');
        const img = out.querySelector('img');
        const capText = out.dataset.caption || (img && img.dataset.caption);
        if (frame && capText) {
          const cap = document.createElement('div');
          cap.className = 'see-caption';
          cap.textContent = capText;
          frame.appendChild(cap);
        }
        // If a poster/static src and a data-gif is present, ensure we start with poster
        if (img && img.dataset && img.dataset.gif) {
          if (img.dataset.poster) {
            img.src = img.dataset.poster;
          }
        }
      } catch(e){}

      let revealed = false;
      function reveal(){
        if (revealed) return;
        out.classList.add('revealed');
        // Swap to GIF on reveal if provided
        const img = out.querySelector('img');
        if (img && img.dataset && img.dataset.gif) {
          img.src = img.dataset.gif;
        }
        // Hide hint after reveal
        const hint = out.querySelector('.see-hint');
        if (hint) hint.style.display = 'none';
        revealed = true;
      }
      out.addEventListener('click', reveal);

      // If image fails to load, show a styled placeholder so terminals still render
  // Intentionally do not replace missing images; allow 404s to show so markup remains visible
      // Lazy reveal when scrolled into view beyond 40% (fallback to timeout if IO missing)
      if ('IntersectionObserver' in window) {
        const io = new IntersectionObserver((entries)=>{
          entries.forEach(en=>{ if(en.isIntersecting && en.intersectionRatio > 0.4) reveal(); });
        },{threshold:[0,0.25,0.4,0.6,0.8,1]});
        io.observe(out);
      } else {
        setTimeout(reveal, 300);
      }
    });
  }

  if (document.readyState === 'loading'){
    document.addEventListener('DOMContentLoaded', ()=> setupReveals(document));
  } else {
    setupReveals(document);
  }
})();
