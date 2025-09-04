// Image Modal System - Victorian Themed
// =====================================

(function() {
  'use strict';

  let currentImageIndex = 0;
  let allImages = [];
  let modalOverlay = null;
  let modalImg = null;
  let modalCaption = null;
  let isLoading = false;

  // Initialize modal system when DOM is ready
  function initImageModal() {
    console.log('Initializing image modal system...');

    // Create modal HTML structure
    createModalStructure();

    // Wait a bit for all images to load
    setTimeout(function() {
      // Find all clickable images
      collectClickableImages();

      // Add click handlers to images
      attachImageHandlers();

      // Setup keyboard navigation
      setupKeyboardNav();

      console.log('Image modal initialized with ' + allImages.length + ' images');
    }, 500);
  }

  // Create the modal HTML elements
  function createModalStructure() {
    // Check if modal already exists
    if (document.getElementById('image-modal-overlay')) {
      console.log('Modal already exists');
      return;
    }

    const modalHTML = `
      <div id="image-modal-overlay" class="image-modal-overlay">
        <div class="image-modal-container">
          <div class="image-modal-wrapper">
            <div class="modal-corner-left"></div>
            <div class="modal-corner-right"></div>
            <div class="image-modal-loading"></div>
            <img id="image-modal-img" class="image-modal-img" alt="">
            <div class="image-modal-caption"></div>
            <div class="image-modal-zoom">Click anywhere to close</div>
          </div>
          <button class="image-modal-close" aria-label="Close modal"></button>
          <button class="image-modal-nav prev" aria-label="Previous image"></button>
          <button class="image-modal-nav next" aria-label="Next image"></button>
        </div>
      </div>
    `;

    document.body.insertAdjacentHTML('beforeend', modalHTML);

    // Get references to modal elements
    modalOverlay = document.getElementById('image-modal-overlay');
    modalImg = document.getElementById('image-modal-img');
    modalCaption = modalOverlay.querySelector('.image-modal-caption');

    // Add event listeners
    setupModalEvents();
  }

  // Collect all images that should be clickable
  function collectClickableImages() {
    allImages = [];

    // Select images from various containers - be more specific
    const selectors = [
      '.screenshot-item img',
      '.example-showcase img',
      '.feature-card img',
      '.code-example img',
      '.clickable-image',
      '.terminal-window img',
      // Also look for images in the see-it-in-action section
      '.see-pane img',
      '.screenshots-container img',
      // Any content image that's not a logo or icon
      '.md-content__inner img:not(.md-logo):not(.hero-logo):not(.torch-logo-static):not([width="20"]):not([width="24"]):not([width="16"])'
    ];

    selectors.forEach(selector => {
      try {
        const images = document.querySelectorAll(selector);
        images.forEach(img => {
          // Skip if already added
          if (allImages.indexOf(img) !== -1) return;

          // Skip icons and very small images
          // Check both natural size (if loaded) and attribute size
          const width = img.naturalWidth || parseInt(img.getAttribute('width')) || 0;
          const height = img.naturalHeight || parseInt(img.getAttribute('height')) || 0;

          // If image not loaded yet, add it anyway (we'll check on click)
          if (!img.complete || width > 50 || height > 50) {
            allImages.push(img);
          }
        });
      } catch(e) {
        console.error('Error collecting images for selector:', selector, e);
      }
    });

    console.log('Found ' + allImages.length + ' clickable images');
  }

  // Attach click handlers to images
  function attachImageHandlers() {
    allImages.forEach((img, index) => {
      // Add visual indicator
      img.style.cursor = 'zoom-in';
      img.classList.add('modal-clickable');
      img.setAttribute('data-modal-index', index);

      // Remove any existing listeners
      img.removeEventListener('click', handleImageClick);

      // Add new listener
      img.addEventListener('click', handleImageClick);
    });
  }

  // Handle image click
  function handleImageClick(e) {
    e.preventDefault();
    e.stopPropagation();

    const index = parseInt(this.getAttribute('data-modal-index'));
    console.log('Image clicked, index:', index);
    openModal(index);
  }

  // Setup modal event listeners
  function setupModalEvents() {
    if (!modalOverlay) return;

    // Close button
    const closeBtn = modalOverlay.querySelector('.image-modal-close');
    if (closeBtn) {
      closeBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        closeModal();
      });
    }

    // Overlay click to close
    modalOverlay.addEventListener('click', function(e) {
      if (e.target === modalOverlay || e.target.classList.contains('image-modal-container')) {
        closeModal();
      }
    });

    // Navigation arrows
    const prevBtn = modalOverlay.querySelector('.image-modal-nav.prev');
    const nextBtn = modalOverlay.querySelector('.image-modal-nav.next');

    if (prevBtn) {
      prevBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        navigateModal(-1);
      });
    }

    if (nextBtn) {
      nextBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        navigateModal(1);
      });
    }

    // Image load events
    if (modalImg) {
      modalImg.addEventListener('load', function() {
        hideLoading();
        updateCaption();
      });

      modalImg.addEventListener('error', function() {
        hideLoading();
        modalCaption.textContent = 'Failed to load image';
      });
    }
  }

  // Setup keyboard navigation
  function setupKeyboardNav() {
    document.addEventListener('keydown', function(e) {
      if (!modalOverlay || !modalOverlay.classList.contains('active')) return;

      switch(e.key) {
        case 'Escape':
          closeModal();
          break;
        case 'ArrowLeft':
          navigateModal(-1);
          break;
        case 'ArrowRight':
          navigateModal(1);
          break;
      }
    });
  }

  // Open modal with specific image
  function openModal(index) {
    if (index < 0 || index >= allImages.length) {
      console.error('Invalid image index:', index);
      return;
    }

    currentImageIndex = index;
    const img = allImages[index];

    console.log('Opening modal for image:', img.src);

    // Show modal
    modalOverlay.classList.add('active');
    document.body.style.overflow = 'hidden';

    // Show loading
    showLoading();

    // Load image - use the full src
    const imgSrc = img.getAttribute('data-large-src') || img.src;
    modalImg.src = imgSrc;
    modalImg.alt = img.alt || '';

    // Update navigation visibility
    updateNavigation();

    // Track analytics if available
    if (typeof gtag !== 'undefined') {
      try {
        gtag('event', 'view_image', {
          'event_category': 'engagement',
          'event_label': imgSrc
        });
      } catch(e) {}
    }
  }

  // Close modal
  function closeModal() {
    if (!modalOverlay) return;

    console.log('Closing modal');
    modalOverlay.classList.remove('active');
    document.body.style.overflow = '';

    if (modalImg) {
      modalImg.src = '';
    }
    currentImageIndex = 0;
  }

  // Navigate to previous/next image
  function navigateModal(direction) {
    const newIndex = currentImageIndex + direction;

    if (newIndex >= 0 && newIndex < allImages.length) {
      openModal(newIndex);
    }
  }

  // Update navigation button visibility
  function updateNavigation() {
    if (!modalOverlay) return;

    const prevBtn = modalOverlay.querySelector('.image-modal-nav.prev');
    const nextBtn = modalOverlay.querySelector('.image-modal-nav.next');

    if (prevBtn) {
      prevBtn.style.display = currentImageIndex > 0 ? 'flex' : 'none';
    }
    if (nextBtn) {
      nextBtn.style.display = currentImageIndex < allImages.length - 1 ? 'flex' : 'none';
    }
  }

  // Update caption from image alt or title
  function updateCaption() {
    if (!modalCaption || currentImageIndex >= allImages.length) return;

    const img = allImages[currentImageIndex];
    const caption = img.alt || img.getAttribute('title') || '';

    if (caption) {
      modalCaption.textContent = caption;
      modalCaption.style.display = 'block';
    } else {
      modalCaption.style.display = 'none';
    }
  }

  // Show loading spinner
  function showLoading() {
    if (!modalOverlay) return;
    const loader = modalOverlay.querySelector('.image-modal-loading');
    if (loader) {
      loader.style.display = 'block';
    }
    isLoading = true;
  }

  // Hide loading spinner
  function hideLoading() {
    if (!modalOverlay) return;
    const loader = modalOverlay.querySelector('.image-modal-loading');
    if (loader) {
      loader.style.display = 'none';
    }
    isLoading = false;
  }

  // Touch gesture support for mobile
  let touchStartX = 0;
  let touchEndX = 0;

  function setupTouchGestures() {
    if (!modalOverlay) return;

    modalOverlay.addEventListener('touchstart', function(e) {
      touchStartX = e.changedTouches[0].screenX;
    }, { passive: true });

    modalOverlay.addEventListener('touchend', function(e) {
      touchEndX = e.changedTouches[0].screenX;
      handleSwipe();
    }, { passive: true });
  }

  function handleSwipe() {
    const swipeThreshold = 50;
    const diff = touchStartX - touchEndX;

    if (Math.abs(diff) > swipeThreshold) {
      if (diff > 0) {
        // Swiped left - next image
        navigateModal(1);
      } else {
        // Swiped right - previous image
        navigateModal(-1);
      }
    }
  }

  // Reinitialize when content changes (for dynamic content)
  function reinitialize() {
    console.log('Reinitializing image modal...');
    collectClickableImages();
    attachImageHandlers();
  }

  // Wait for page to be fully ready
  function waitForReady() {
    // Check if MkDocs has finished rendering
    if (document.querySelector('.md-content__inner')) {
      initImageModal();

      // Also reinit after a delay to catch any dynamically loaded content
      setTimeout(reinitialize, 1500);

      // Setup touch gestures
      setTimeout(setupTouchGestures, 100);
    } else {
      // Try again in a moment
      setTimeout(waitForReady, 100);
    }
  }

  // Multiple initialization strategies to ensure it works

  // Strategy 1: DOMContentLoaded
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', waitForReady);
  } else {
    // Strategy 2: If DOM already loaded
    waitForReady();
  }

  // Strategy 3: Also listen for window load (all resources loaded)
  window.addEventListener('load', function() {
    setTimeout(reinitialize, 500);
  });

  // Export for external use and debugging
  window.imageModal = {
    open: openModal,
    close: closeModal,
    reinit: reinitialize,
    getImages: function() { return allImages; },
    debug: function() {
      console.log('Modal overlay:', modalOverlay);
      console.log('Images found:', allImages.length);
      console.log('Images:', allImages);
    }
  };

})();
