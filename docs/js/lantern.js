// Lantern Glow and Animation Effects
// Creates the iridescent flame effect for logos

(function() {
    'use strict';
    
    // ============================================
    // Lantern Controller Class
    // ============================================
    
    class LanternController {
        constructor(element, options = {}) {
            this.element = element;
            this.options = {
                glowIntensity: options.glowIntensity || 1,
                flickerSpeed: options.flickerSpeed || 3000,
                particleCount: options.particleCount || 8,
                particleSize: options.particleSize || 4,
                particleOpacity: options.particleOpacity || 0.5,
                interactive: options.interactive !== false,
                iridescent: options.iridescent !== false,
                ...options
            };
            
            this.mousePosition = { x: 0, y: 0 };
            this.isHovered = false;
            this.particles = [];
            this.particlesContainer = null;
            this.active = true;
            this.allowRespawn = true;
            this.flickerTimer = null;
            this.inViewport = true;
            this.observer = null;
            // Respect prefers-reduced-motion
            try {
                if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
                    this.options.flickerSpeed = Math.max(5000, this.options.flickerSpeed);
                    this.options.particleCount = 0; // disable particles for reduced motion
                }
            } catch (_) {}
            
            this.init();
        }
        
        init() {
            this.createStructure();
            this.setupAnimations();
            this.setupInteractions();
            // Only dots + flicker; no iridescent radiance
            this.startParticles();
            // Optimize: pause when off-screen or tab hidden
            this.setupVisibilityControls();
        }
        
        // ============================================
        // Structure Creation
        // ============================================
        
        createStructure() {
            // Clear existing content
            this.element.innerHTML = '';
            this.element.classList.add('lantern-container');
            
            // Create SVG container
            const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            svg.setAttribute('viewBox', '0 0 200 200');
            svg.setAttribute('class', 'lantern-svg');
            svg.style.cssText = `
                width: 100%;
                height: 100%;
                position: absolute;
                top: 0;
                left: 0;
                z-index: 1;
            `;
            
            // Create gradient definitions
            const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
            
            // Multiple gradients for iridescent effect
            const colors = [
                { id: 'orange', colors: ['#ff6b35', '#f7931e'] },
                { id: 'blue', colors: ['#00ffff', '#0099ff'] },
                { id: 'purple', colors: ['#8b00ff', '#ff00ff'] },
                { id: 'green', colors: ['#00ff00', '#00ff99'] },
                { id: 'pink', colors: ['#ff0066', '#ff99cc'] }
            ];
            
            colors.forEach(gradient => {
                const grad = this.createGradient(gradient.id, [
                    { offset: '0%', color: gradient.colors[0], opacity: 0.8 },
                    { offset: '100%', color: gradient.colors[1], opacity: 0.3 }
                ]);
                defs.appendChild(grad);
            });
            
            svg.appendChild(defs);
            
            // Subtle inner light only (no concentric circles or rays)
            const innerLight = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            innerLight.setAttribute('cx', '100');
            innerLight.setAttribute('cy', '100');
            innerLight.setAttribute('r', '18');
            innerLight.setAttribute('fill', '#f59e0b');
            innerLight.setAttribute('opacity', '0.25');
            innerLight.setAttribute('class', 'lantern-inner-light');
            svg.appendChild(innerLight);
            
            this.element.appendChild(svg);
            
            // Store references
            this.svg = svg;
            this.innerLight = innerLight;
        }
        
        createGradient(id, stops) {
            const gradient = document.createElementNS('http://www.w3.org/2000/svg', 'radialGradient');
            gradient.setAttribute('id', `lantern-${id}`);
            
            stops.forEach(stop => {
                const stopElement = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
                stopElement.setAttribute('offset', stop.offset);
                stopElement.setAttribute('style', 
                    `stop-color:${stop.color};stop-opacity:${stop.opacity}`
                );
                gradient.appendChild(stopElement);
            });
            
            return gradient;
        }
        
        createRay(angle, colorId) {
            const ray = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            ray.setAttribute('transform', `rotate(${angle})`);
            ray.setAttribute('class', 'lantern-ray');
            
            const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            path.setAttribute('d', 'M0,-40 L3,-70 L0,-100 L-3,-70 Z');
            path.setAttribute('fill', `url(#lantern-${colorId})`);
            path.setAttribute('opacity', '0.6');
            ray.appendChild(path);
            
            return ray;
        }
        
        // ============================================
        // Iridescent Effect
        // ============================================
        
    // Removed iridescent radiance effect
        
        // ============================================
        // Animations
        // ============================================
        
        setupAnimations() {
            // Flicker animation only
            this.startFlicker();
        }
        
        startFlicker() {
            const flicker = () => {
                if (!this.active) { return; }
                const intensity = 0.55 + Math.random() * 0.45; // wider range
                const duration = 80 + Math.random() * 180; // snappier flicker
                
                // Inner light subtle flicker
                this.innerLight.style.transition = `opacity ${duration}ms ease`;
                this.innerLight.style.opacity = 0.18 + Math.random() * 0.22;
                // Slight radius pulse to accent flicker (keep small)
                const rBase = 18;
                const rPulse = rBase + (Math.random() * 2 - 1) * 0.8; // +/- 0.8
                this.innerLight.setAttribute('r', String(Math.max(14, Math.min(20, rPulse))));
                
                this.flickerTimer = setTimeout(flicker, duration + Math.random() * this.options.flickerSpeed);
            };
            
            flicker();
        }
        
    // Removed ray rotation and pulsing radiance
        
        // ============================================
        // Particles
        // ============================================
        
    startParticles() {
            const particlesContainer = document.createElement('div');
            particlesContainer.className = 'lantern-particles';
            particlesContainer.style.cssText = `
                position: absolute;
                bottom: 0;
                left: 0;
                width: 100%;
                height: 38%; /* confine to bottom area */
                pointer-events: none;
                z-index: 1;
            `;
            
            this.element.appendChild(particlesContainer);
        this.particlesContainer = particlesContainer;
            
            // Create subtle amber dots along the bottom
        for (let i = 0; i < this.options.particleCount; i++) {
                setTimeout(() => {
            if (this.active) this.createParticle(particlesContainer);
                }, i * 200);
            }
        }
        
        createParticle(container) {
            const particle = document.createElement('div');
            particle.className = 'lantern-particle';
            
            // Subtle amber hue
            const hue = 42 + Math.random() * 8; // around amber
            
            // Start near bottom edge, random horizontal position
            const x = 5 + Math.random() * 90;  // percent
            const y = 90 + Math.random() * 8;  // near bottom
            
            particle.style.cssText = `
                position: absolute;
                width: ${Math.max(2, this.options.particleSize - 1)}px;
                height: ${Math.max(2, this.options.particleSize - 1)}px;
                background: hsla(${hue}, 85%, 55%, ${Math.min(0.5, this.options.particleOpacity)});
                border-radius: 50%;
                left: ${x}%;
                top: ${y}%;
                pointer-events: none;
                opacity: 0;
                filter: blur(0.4px);
                box-shadow: 0 0 3px hsla(${hue}, 85%, 55%, ${Math.min(0.35, this.options.particleOpacity)});
            `;
            
            container.appendChild(particle);
            this.animateParticle(particle, container);
        }
        
        animateParticle(particle, container) {
            const duration = 3500 + Math.random() * 2400; // longer, calmer drift
            const startY = parseFloat(particle.style.top);
            const endY = startY + (Math.random() * 4 - 2); // slight vertical wobble
            const startX = parseFloat(particle.style.left);
            const endX = startX + (Math.random() - 0.5) * 24; // horizontal drift
            
            const startTime = Date.now();
            
            const animate = () => {
                if (!this.active) { return; }
                const now = Date.now();
                const progress = (now - startTime) / duration;
                
                if (progress >= 1) {
                    if (particle.parentNode) {
                        container.removeChild(particle);
                    }
                    if (this.active && this.allowRespawn) {
                        this.createParticle(container);
                    }
                    return;
                }
                
                const easeProgress = 1 - Math.pow(1 - progress, 2);
                
                particle.style.left = startX + (endX - startX) * easeProgress + '%';
                particle.style.top = startY + (endY - startY) * easeProgress + '%';
                
                // Fade in and out
                if (progress < 0.2) {
                    particle.style.opacity = progress * 5;
                } else if (progress > 0.8) {
                    particle.style.opacity = (1 - progress) * 5;
                } else {
                    particle.style.opacity = 1;
                }
                
                // Slight scale pulse, no rotation
                const scale = 1 + Math.sin(progress * Math.PI) * 0.12;
                particle.style.transform = `scale(${scale})`;
                
                requestAnimationFrame(animate);
            };
            
            animate();
        }
        
        // ============================================
        // Interactions
        // ============================================
        
        setupInteractions() {
            if (!this.options.interactive) return;
            
            this.element.addEventListener('mouseenter', () => {
                this.isHovered = true;
                this.onHoverStart();
            });
            
            this.element.addEventListener('mouseleave', () => {
                this.isHovered = false;
                this.onHoverEnd();
            });
            
            this.element.addEventListener('click', () => {
                this.onLanternClick();
            });
        }
        
        onHoverStart() {
            // Slightly brighten inner light on hover
            this.innerLight.style.transition = 'opacity 0.3s ease, r 0.3s ease';
            this.innerLight.style.opacity = 0.28;
            this.innerLight.setAttribute('r', '19');
        }
        
        onHoverEnd() {
            this.innerLight.style.opacity = 0.2;
            this.innerLight.setAttribute('r', '18');
        }
        
        onLanternClick() {
            // No burst
        }
        
        // Removed burst visual
        
        // ============================================
        // Visibility/Performance Controls
        // ============================================
        setupVisibilityControls() {
            // Pause when tab hidden
            document.addEventListener('visibilitychange', () => {
                if (document.hidden) {
                    this.pauseAnimations();
                } else if (this.inViewport) {
                    this.resumeAnimations();
                }
            });
            // Pause when element off-screen
            if ('IntersectionObserver' in window) {
                const io = new IntersectionObserver((entries) => {
                    entries.forEach(entry => {
                        this.inViewport = entry.isIntersecting && entry.intersectionRatio > 0;
                        if (this.inViewport && !document.hidden) {
                            this.resumeAnimations();
                        } else {
                            this.pauseAnimations();
                        }
                    });
                }, { root: null, threshold: 0.05 });
                io.observe(this.element);
                this.observer = io;
            }
        }
        
        pauseAnimations() {
            if (!this.active) return;
            this.active = false;
            this.allowRespawn = false;
            if (this.flickerTimer) {
                clearTimeout(this.flickerTimer);
                this.flickerTimer = null;
            }
        }
        
        resumeAnimations() {
            if (this.active) return;
            this.active = true;
            this.allowRespawn = true;
            if (!this.flickerTimer) {
                this.startFlicker();
            }
            // Restart particles if container exists but few/no children
            if (this.particlesContainer) {
                // Clear stale particles
                this.particlesContainer.innerHTML = '';
                for (let i = 0; i < this.options.particleCount; i++) {
                    this.createParticle(this.particlesContainer);
                }
            }
        }
    }
    
    // ============================================
    // Auto-initialization
    // ============================================
    
    document.addEventListener('DOMContentLoaded', () => {
    const lanternElements = document.querySelectorAll('.lantern-container, [data-lantern]');
        
        lanternElements.forEach(element => {
            const options = {
                glowIntensity: parseFloat(element.dataset.glowIntensity) || 1,
                flickerSpeed: parseInt(element.dataset.flickerSpeed) || 3000,
        particleCount: parseInt(element.dataset.particleCount) || 8,
                interactive: element.dataset.interactive !== 'false',
                iridescent: element.dataset.iridescent !== 'false'
            };
            
            new LanternController(element, options);
        });
        
        window.LanternController = LanternController;
    });
    
    // Add required CSS
    const style = document.createElement('style');
    style.textContent = `
        .lantern-container {
            position: relative;
            width: 100%;
            height: 100%;
            cursor: pointer;
            user-select: none;
        }
    `;
    document.head.appendChild(style);
    
})();