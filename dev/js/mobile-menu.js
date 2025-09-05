// Mobile menu enhancement for Material for MkDocs
(function(){
  'use strict';

  function init(){
    // Wait a bit for Material to initialize
    setTimeout(function() {
      setupMobileMenu();
    }, 100);
  }

  function setupMobileMenu() {
    // Find the drawer toggle elements
    const drawerToggle = document.getElementById('__drawer');
    const hamburger = document.querySelector('.md-header__button[for="__drawer"]');
    const sidebar = document.querySelector('.md-sidebar--primary');
    const overlay = document.querySelector('.md-overlay');

    if(!drawerToggle) {
      console.warn('Mobile menu: Drawer toggle not found');
      return;
    }

    console.log('Mobile menu: Setting up...');

    // Fix hamburger click to properly toggle drawer
    if(hamburger) {
      // Remove any existing click handlers
      const newHamburger = hamburger.cloneNode(true);
      hamburger.parentNode.replaceChild(newHamburger, hamburger);

      newHamburger.addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        drawerToggle.checked = !drawerToggle.checked;

        // Update body state
        if(drawerToggle.checked) {
          document.body.classList.add('drawer-open');
          document.body.style.overflow = 'hidden';
        } else {
          document.body.classList.remove('drawer-open');
          document.body.style.overflow = '';
        }
      });
    }

    // Fix navigation links in the drawer
    if(sidebar) {
      const navLinks = sidebar.querySelectorAll('.md-nav__link');

      navLinks.forEach(function(link) {
        // Ensure links are clickable
        link.style.pointerEvents = 'auto';
        link.style.cursor = 'pointer';

        // Remove any existing handlers
        const newLink = link.cloneNode(true);
        link.parentNode.replaceChild(newLink, link);

        // Add click handler for navigation
        newLink.addEventListener('click', function(e) {
          // Don't prevent default for actual navigation
          const href = this.getAttribute('href');

          // If it's an external link (blog/contact), let it navigate
          if(href && (href.includes('http') || href.includes('//'))) {
            // External link - allow normal navigation
            return true;
          }

          // For internal links, close the drawer after a delay
          if(href && href !== '#') {
            setTimeout(function() {
              drawerToggle.checked = false;
              document.body.classList.remove('drawer-open');
              document.body.style.overflow = '';
            }, 100);
          }
        });
      });
    }

    // Handle overlay clicks to close drawer
    if(overlay) {
      overlay.addEventListener('click', function(e) {
        e.preventDefault();
        drawerToggle.checked = false;
        document.body.classList.remove('drawer-open');
        document.body.style.overflow = '';
      });
    }

    // Close drawer on ESC key
    document.addEventListener('keydown', function(e) {
      if(e.key === 'Escape' && drawerToggle.checked) {
        drawerToggle.checked = false;
        document.body.classList.remove('drawer-open');
        document.body.style.overflow = '';
      }
    });

    // Ensure drawer state is synced on page load
    if(drawerToggle.checked) {
      document.body.classList.add('drawer-open');
      document.body.style.overflow = 'hidden';
    }

    console.log('Mobile menu: Setup complete');
  }

  // Re-initialize on navigation for SPA behavior
  document.addEventListener('DOMContentLoaded', init);

  // Also handle instant navigation
  document.addEventListener('contentChanged', init);

  // Initial setup
  if(document.readyState !== 'loading') {
    init();
  }
})();
