// Move search bar into the drawer on mobile (<1100px) while keeping desktop placement intact
(function(){
  const MOBILE_BREAKPOINT = 1100;
  let moved = false;
  let placeholderContainer; // wrapper inside drawer

  function ensurePlaceholder(){
    if(!placeholderContainer){
      const nav = document.querySelector('.md-nav--primary');
      if(nav){
        // Insert after the nav title/logo area (first heading or logo link in drawer)
        placeholderContainer = document.createElement('div');
        placeholderContainer.className = 'mobile-drawer-search-wrapper';
        placeholderContainer.style.margin = '0.75rem 0 0.5rem';
        placeholderContainer.style.padding = '0.5rem 0.75rem 0.75rem';
        placeholderContainer.style.borderBottom = '1px solid var(--md-default-fg-color--lightest, rgba(0,0,0,0.08))';
        nav.parentNode.insertBefore(placeholderContainer, nav.nextSibling);
      }
    }
    return placeholderContainer;
  }

  function moveIn(){
    if(moved) return;
    const search = document.querySelector('.md-search');
    const drawer = document.querySelector('.md-sidebar--primary');
    if(search && drawer){
      const target = ensurePlaceholder();
      if(target){
        target.appendChild(search);
        search.classList.add('in-mobile-drawer');
        moved = true;
      }
    }
  }

  function moveOut(){
    if(!moved) return;
    const search = document.querySelector('.md-search.in-mobile-drawer');
    const header = document.querySelector('.md-header__inner');
    if(search && header){
      // Place it back before theme toggle (order already handled by CSS)
      header.appendChild(search);
      search.classList.remove('in-mobile-drawer');
      moved = false;
    }
  }

  function evaluate(){
    if(window.innerWidth < MOBILE_BREAKPOINT){
      moveIn();
    } else {
      moveOut();
    }
  }

  // Watch for dynamic search mount (Material injects late sometimes)
  const observer = new MutationObserver(()=>{ evaluate(); });
  observer.observe(document.documentElement, {childList:true, subtree:true});

  window.addEventListener('resize', evaluate);
  document.addEventListener('DOMContentLoaded', evaluate);
})();
