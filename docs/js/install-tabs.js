// Install Tabs functionality for the hero section
// Handles tab switching for Core/MCP/Full installation options

(function() {
    'use strict';

    function initInstallTabs() {
        const buttons = document.querySelectorAll('.install-tab-btn');
        const panes = document.querySelectorAll('.install-tab-pane');

        if (!buttons.length || !panes.length) return;

        buttons.forEach(btn => {
            btn.addEventListener('click', () => {
                const tab = btn.dataset.tab;

                // Update buttons
                buttons.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');

                // Update panes
                panes.forEach(p => {
                    if (p.dataset.tab === tab) {
                        p.classList.add('active');
                    } else {
                        p.classList.remove('active');
                    }
                });
            });
        });
    }

    // Initialize on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initInstallTabs);
    } else {
        initInstallTabs();
    }

    // Re-initialize on navigation (for MkDocs instant loading)
    document.addEventListener('DOMContentLoaded', initInstallTabs);
    if (typeof document$ !== 'undefined') {
        document$.subscribe(initInstallTabs);
    }
})();

