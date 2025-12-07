// Mobile menu enhancement for Material for MkDocs
(function(){
  function init(){
    // Find the checkbox and hamburger button
    const checkbox = document.getElementById('__drawer') || document.querySelector('.md-toggle--drawer');
    const hamburger = document.querySelector('.md-header__button[for="__drawer"]');
    const sidebar = document.querySelector('.md-sidebar--primary');
    const overlay = document.querySelector('.md-overlay');
    
    if(!checkbox || !hamburger) {
      console.warn('Mobile menu: Required elements not found, retrying...');
      return;
    }
    
    console.log('Mobile menu: Initialized successfully');
    
    // Ensure hamburger click toggles the checkbox
    hamburger.addEventListener('click', function(e) {
      e.preventDefault();
      checkbox.checked = !checkbox.checked;
      
      // Toggle body scroll lock
      if(checkbox.checked) {
        document.body.style.overflow = 'hidden';
        document.body.classList.add('drawer-open');
      } else {
        document.body.style.overflow = '';
        document.body.classList.remove('drawer-open');
      }
    });
    
    // Close drawer on overlay click - set up proper event handling
    function closeDrawer() {
      checkbox.checked = false;
      document.body.style.overflow = '';
      document.body.classList.remove('drawer-open');
    }
    
    // Handle overlay clicks
    if(overlay) {
      overlay.addEventListener('click', function(e) {
        e.preventDefault();
        closeDrawer();
      });
    }
    
    // Handle clicks on the document when drawer is open
    document.addEventListener('click', function(e) {
      // Only process if drawer is open
      if(!checkbox.checked) return;
      
      // Check if click is outside drawer and hamburger
      if(sidebar && !sidebar.contains(e.target) && !hamburger.contains(e.target)) {
        closeDrawer();
      }
    }, true); // Use capture phase to catch events before they bubble
    
    // Close drawer on navigation link click
    if(sidebar) {
      sidebar.addEventListener('click', function(e) {
        if(e.target.tagName === 'A' && !e.target.href.includes('#')) {
          setTimeout(() => {
            checkbox.checked = false;
            document.body.style.overflow = '';
            document.body.classList.remove('drawer-open');
          }, 100);
        }
      });
    }
    
    // Handle back button (browser history)
    window.addEventListener('popstate', function() {
      if(checkbox.checked) {
        checkbox.checked = false;
        document.body.style.overflow = '';
        document.body.classList.remove('drawer-open');
      }
    });
    
    // Make drawer search placeholder clickable - triggers the main search
    if(sidebar) {
      const navTitle = sidebar.querySelector('.md-nav__title:first-child');
      if(navTitle) {
        navTitle.addEventListener('click', function(e) {
          // Check if click is on the ::after pseudo element area (bottom part)
          const rect = navTitle.getBoundingClientRect();
          const clickY = e.clientY - rect.top;
          
          // If click is in the search area (bottom 48px)
          if(clickY > rect.height - 48) {
            e.preventDefault();
            closeDrawer();
            
            // Trigger the search overlay
            const searchToggle = document.querySelector('label[for="__search"]');
            if(searchToggle) {
              searchToggle.click();
            }
          }
        });
      }
    }
  }

  // Initialize when DOM is ready
  if(document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
  
  // Retry initialization after a delay for dynamically loaded content
  setTimeout(init, 1000);
  
  // Also reinitialize on navigation (for SPA-like behavior)
  document.addEventListener('DOMContentLoaded', init);
  window.addEventListener('load', init);
})();