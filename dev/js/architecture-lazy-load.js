// Architecture Documentation Performance Optimizations
(function(){
    'use strict';

    class ArchitectureLazyLoader {
        constructor() {
            this.mermaidLoaded = false;
            this.diagramsObserver = null;
            this.init();
        }

        init() {
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', () => this.setup());
            } else {
                this.setup();
            }
        }

        setup() {
            this.setupCollapsibleSections();
            this.setupLazyMermaidLoading();
            this.setupProgressiveDisclosure();
        }

        // Simple collapsible sections - CSS-only approach
        setupCollapsibleSections() {
            // Look for details/summary elements which are natively supported
            const detailsElements = document.querySelectorAll('details');
            detailsElements.forEach(details => {
                // Add smooth animation classes
                details.classList.add('collapsible-details');

                // Optional: Add event listeners for custom behavior
                details.addEventListener('toggle', () => {
                    if (details.open) {
                        details.classList.add('expanded');
                    } else {
                        details.classList.remove('expanded');
                    }
                });
            });
        }


        // Lazy load Mermaid diagrams using Intersection Observer
        setupLazyMermaidLoading() {
            const diagrams = document.querySelectorAll('pre code.language-mermaid');
            if (diagrams.length === 0) return;

            // Create placeholders
            diagrams.forEach((diagram, index) => {
                const placeholder = document.createElement('div');
                placeholder.className = 'mermaid-placeholder';
                placeholder.innerHTML = `
                    <div class="mermaid-loading">
                        <div class="loading-spinner"></div>
                        <p>Loading diagram ${index + 1}...</p>
                        <button class="load-diagram-btn" data-diagram-id="${index}">Load Now</button>
                    </div>
                `;

                // Store original content
                placeholder.dataset.originalContent = diagram.textContent;
                diagram.parentElement.replaceChild(placeholder, diagram);
            });

            // Set up intersection observer
            if ('IntersectionObserver' in window) {
                this.diagramsObserver = new IntersectionObserver((entries) => {
                    entries.forEach(entry => {
                        if (entry.isIntersecting) {
                            this.loadMermaidDiagram(entry.target);
                        }
                    });
                }, {
                    rootMargin: '100px 0px', // Load when 100px away from viewport
                    threshold: 0.1
                });

                // Observe all placeholders
                document.querySelectorAll('.mermaid-placeholder').forEach(placeholder => {
                    this.diagramsObserver.observe(placeholder);
                });
            }

            // Manual load buttons
            document.addEventListener('click', (e) => {
                if (e.target.matches('.load-diagram-btn')) {
                    const placeholder = e.target.closest('.mermaid-placeholder');
                    this.loadMermaidDiagram(placeholder);
                }
            });
        }

        async loadMermaidDiagram(placeholder) {
            if (placeholder.classList.contains('loaded')) return;
            placeholder.classList.add('loaded');

            try {
                // Load Mermaid library if not already loaded
                if (!this.mermaidLoaded) {
                    await this.loadMermaidLibrary();
                }

                // Create diagram element
                const diagramDiv = document.createElement('div');
                diagramDiv.className = 'mermaid';
                diagramDiv.textContent = placeholder.dataset.originalContent;

                // Replace placeholder
                placeholder.parentElement.replaceChild(diagramDiv, placeholder);

                // Initialize Mermaid for this specific diagram
                if (window.mermaid) {
                    window.mermaid.init(undefined, diagramDiv);
                }

                // Stop observing this element
                if (this.diagramsObserver) {
                    this.diagramsObserver.unobserve(placeholder);
                }

            } catch (error) {
                console.error('Failed to load Mermaid diagram:', error);
                placeholder.innerHTML = `
                    <div class="mermaid-error">
                        <p>⚠️ Failed to load diagram</p>
                        <button class="retry-load-btn">Retry</button>
                    </div>
                `;
            }
        }

        loadMermaidLibrary() {
            return new Promise((resolve, reject) => {
                if (this.mermaidLoaded) {
                    resolve();
                    return;
                }

                const script = document.createElement('script');
                script.src = 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js';
                script.onload = () => {
                    window.mermaid.initialize({
                        startOnLoad: false,
                        theme: document.documentElement.dataset.mdColorScheme === 'slate' ? 'dark' : 'default'
                    });
                    this.mermaidLoaded = true;
                    resolve();
                };
                script.onerror = () => {
                    reject(new Error('Failed to load Mermaid library'));
                };
                document.head.appendChild(script);
            });
        }

        // Progressive disclosure for very large sections
        setupProgressiveDisclosure() {
            const largeSections = document.querySelectorAll('.large-section');
            largeSections.forEach(section => {
                const content = section.querySelector('.section-content');
                if (!content) return;

                const showMoreBtn = document.createElement('button');
                showMoreBtn.className = 'show-more-btn';
                showMoreBtn.textContent = 'Show More';
                showMoreBtn.addEventListener('click', () => {
                    section.classList.add('expanded');
                    showMoreBtn.style.display = 'none';
                });

                section.appendChild(showMoreBtn);
            });
        }
    }

    // Initialize when page loads
    new ArchitectureLazyLoader();

})();
