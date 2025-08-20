
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
                particleCount: options.particleCount || 10,
                particleSize: options.particleSize || 4,
                particleOpacity: options.particleOpacity || 0.5,
                interactive: options.interactive !== false,
                iridescent: options.iridescent !== false,
                ...options
            };
            
            this.mousePosition = { x: 0, y: 0 };
            this.isHovered = false;
            this.particles = [];
            
            this.init();
        }
        
        init() {
            const isApi = document.body && document.body.classList.contains('is-api');
            if (isApi) {
                return; // skip all DOM work on API pages
            }
            this.createStructure();
            if (!isApi) {
                this.setupAnimations();
                this.setupInteractions();
                if (this.options.iridescent) {
                    this.startIridescentEffect();
                }
                this.startParticles();
            }
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
            
            // Create multiple glow circles for layered effect
            this.glowLayers = [];
            for (let i = 0; i < 5; i++) {
                const glowCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                glowCircle.setAttribute('cx', '100');
                glowCircle.setAttribute('cy', '100');
                glowCircle.setAttribute('r', 60 + i * 10);
                glowCircle.setAttribute('fill', `url(#${colors[i].id})`);
                glowCircle.setAttribute('class', `lantern-glow-${i}`);
                glowCircle.style.opacity = 0.3 - i * 0.05;
                glowCircle.style.mixBlendMode = 'screen';
                svg.appendChild(glowCircle);
                this.glowLayers.push(glowCircle);
            }
            
            // Create rays with gradient colors
            const raysGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            raysGroup.setAttribute('class', 'lantern-rays');
            raysGroup.setAttribute('transform', 'translate(100,100)');
            
            for (let i = 0; i < 12; i++) {
                const ray = this.createRay(i * 30, colors[i % colors.length].id);
                raysGroup.appendChild(ray);
            }
            
            svg.appendChild(raysGroup);
            
            // Inner light circle
            const innerLight = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            innerLight.setAttribute('cx', '100');
            innerLight.setAttribute('cy', '100');
            innerLight.setAttribute('r', '25');
            innerLight.setAttribute('fill', '#ffffff');
            innerLight.setAttribute('opacity', '0.9');
            innerLight.setAttribute('class', 'lantern-inner-light');
            svg.appendChild(innerLight);
            
            this.element.appendChild(svg);
            
            // Store references
            this.svg = svg;
            this.raysGroup = raysGroup;
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
        
        startIridescentEffect() {
            let hue = 0;
            const animate = () => {
                hue = (hue + 0.3) % 360; // slower cycle to be less distracting
                
                // Animate each glow layer with different hues
                this.glowLayers.forEach((layer, index) => {
                    const layerHue = (hue + index * 72) % 360;
                    layer.style.fill = `hsl(${layerHue}, 100%, 50%)`;
                });
                
                requestAnimationFrame(animate);
            };
            
            animate();
        }
        
        // ============================================
        // Animations
        // ============================================
        
        setupAnimations() {
            // Flicker animation
            this.startFlicker();
            
            // Ray rotation
            this.startRayRotation();
            
            // Pulse animation
            this.startPulse();
        }
        
        startFlicker() {
            const flicker = () => {
                const intensity = 0.55 + Math.random() * 0.45; // wider range
                const duration = 80 + Math.random() * 180; // snappier flicker
                
                this.glowLayers.forEach((layer, index) => {
                    layer.style.transition = `opacity ${duration}ms ease`;
                    const base = 0.28 - index * 0.05;
                    layer.style.opacity = Math.max(0, base) * intensity * this.options.glowIntensity;
                });
                
                // Inner light: stronger presence
                this.innerLight.style.transition = `opacity ${duration}ms ease`;
                this.innerLight.style.opacity = 0.75 + Math.random() * 0.35;
                // Slight radius pulse to accent flicker
                const rBase = 25;
                const rPulse = rBase + (Math.random() * 2 - 1) * 1.2; // +/- 1.2
                this.innerLight.setAttribute('r', String(Math.max(22, Math.min(28, rPulse))));
                
                setTimeout(flicker, duration + Math.random() * this.options.flickerSpeed);
            };
            
            flicker();
        }
        
        startRayRotation() {
            let rotation = 0;
            const rotate = () => {
                rotation += 0.5;
                this.raysGroup.setAttribute('transform', 
                    `translate(100,100) rotate(${rotation})`
                );
                requestAnimationFrame(rotate);
            };
            
            rotate();
        }
        
        startPulse() {
            const pulse = () => {
                const scale = 1 + Math.sin(Date.now() / 1000) * 0.05;
                this.glowLayers.forEach((layer, index) => {
                    const layerScale = scale + index * 0.01;
                    layer.setAttribute('transform', 
                        `translate(100,100) scale(${layerScale}) translate(-100,-100)`
                    );
                });
                requestAnimationFrame(pulse);
            };
            
            pulse();
        }
        
        // ============================================
        // Particles
        // ============================================
        
        startParticles() {
            const particlesContainer = document.createElement('div');
            particlesContainer.className = 'lantern-particles';
            particlesContainer.style.cssText = `
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                z-index: 3;
            `;
            
            this.element.appendChild(particlesContainer);
            
            // Create colorful particles
            for (let i = 0; i < this.options.particleCount; i++) {
                setTimeout(() => {
                    this.createParticle(particlesContainer);
                }, i * 200);
            }
        }
        
        createParticle(container) {
            const particle = document.createElement('div');
            particle.className = 'lantern-particle';
            
            // Random color from rainbow spectrum
            const hue = Math.random() * 360;
            
            // Random starting position
            const angle = Math.random() * Math.PI * 2;
            const distance = 30 + Math.random() * 20;
            const x = 50 + Math.cos(angle) * distance;
            const y = 50 + Math.sin(angle) * distance;
            
            particle.style.cssText = `
                position: absolute;
                width: ${this.options.particleSize}px;
                height: ${this.options.particleSize}px;
                background: radial-gradient(circle, 
                    hsla(${hue}, 90%, 55%, 0.9) 0%, 
                    transparent 70%);
                border-radius: 50%;
                left: ${x}%;
                top: ${y}%;
                pointer-events: none;
                opacity: 0;
                filter: blur(0.5px);
                box-shadow: 0 0 6px hsla(${hue}, 90%, 55%, ${this.options.particleOpacity});
            `;
            
            container.appendChild(particle);
            this.animateParticle(particle, container);
        }
        
        animateParticle(particle, container) {
            const duration = 2600 + Math.random() * 1600; // slightly shorter life
            const startY = parseFloat(particle.style.top);
            const endY = startY - 24 - Math.random() * 16; // less travel
            const startX = parseFloat(particle.style.left);
            const endX = startX + (Math.random() - 0.5) * 14; // less drift
            
            const startTime = Date.now();
            
            const animate = () => {
                const now = Date.now();
                const progress = (now - startTime) / duration;
                
                if (progress >= 1) {
                    container.removeChild(particle);
                    this.createParticle(container);
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
                
                // Scale and rotate
                const scale = 1 + Math.sin(progress * Math.PI) * 0.3; // smaller scale pulse
                const rotation = progress * 180; // slower rotation
                particle.style.transform = `scale(${scale}) rotate(${rotation}deg)`;
                
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
            this.glowLayers.forEach((layer, index) => {
                layer.style.transition = 'all 0.3s ease';
                const newRadius = 60 + index * 10 + 10;
                layer.setAttribute('r', newRadius);
            });
            
            this.raysGroup.style.animation = 'spin 2s linear infinite';
        }
        
        onHoverEnd() {
            this.glowLayers.forEach((layer, index) => {
                const originalRadius = 60 + index * 10;
                layer.setAttribute('r', originalRadius);
            });
            
            this.raysGroup.style.animation = 'none';
        }
        
        onLanternClick() {
            this.createBurst();
        }
        
        createBurst() {
            const burst = document.createElement('div');
            burst.className = 'lantern-burst';
            
            const hue = Math.random() * 360;
            
            burst.style.cssText = `
                position: absolute;
                top: 50%;
                left: 50%;
                width: 200%;
                height: 200%;
                transform: translate(-50%, -50%) scale(0);
                background: radial-gradient(circle, 
                    hsla(${hue}, 100%, 50%, 0.4) 0%, 
                    transparent 50%);
                border-radius: 50%;
                pointer-events: none;
                z-index: 0;
            `;
            
            this.element.appendChild(burst);
            
            requestAnimationFrame(() => {
                burst.style.transition = 'transform 0.6s ease-out, opacity 0.6s ease-out';
                burst.style.transform = 'translate(-50%, -50%) scale(1)';
                burst.style.opacity = '0';
            });
            
            setTimeout(() => {
                this.element.removeChild(burst);
            }, 600);
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
                particleCount: parseInt(element.dataset.particleCount) || 20,
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
        @keyframes spin {
            from { transform: translate(100,100) rotate(0deg); }
            to { transform: translate(100,100) rotate(360deg); }
        }
        
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