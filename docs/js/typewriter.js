// Typewriter and Text Animation Effects for Victorian Newspaper Theme

// Usage examples

// <!-- Basic typewriter -->
// <span class="typewriter" data-speed="150">Welcome to Tenets</span>

// <!-- Rotating headlines -->
// <div class="rotating-headline" data-headlines="Build better|Code smarter|Ship faster"></div>

// <!-- Fade in text -->
// <p class="fade-in-text">This text will fade in word by word</p>

// <!-- Counter -->
// <span class="counter" data-count-to="10000" data-suffix=" files analyzed"></span>

document.addEventListener('DOMContentLoaded', function() {
    
    // ==============================================
    // TYPEWRITER EFFECT
    // ==============================================
    
    class TypewriterEffect {
        constructor(element, options = {}) {
            this.element = element;
            this.text = options.text || element.textContent;
            this.speed = options.speed || 100;
            this.delay = options.delay || 500;
            this.cursor = options.cursor !== false;
            this.loop = options.loop || false;
            this.loopDelay = options.loopDelay || 2000;
            
            this.element.textContent = '';
            this.element.style.minHeight = this.element.offsetHeight + 'px';
            
            if (this.cursor) {
                this.addCursor();
            }
            
            setTimeout(() => this.type(), this.delay);
        }
        
        addCursor() {
            const cursor = document.createElement('span');
            cursor.className = 'typewriter-cursor';
            cursor.textContent = '|';
            cursor.style.cssText = `
                display: inline-block;
                margin-left: 2px;
                animation: blink 1s infinite;
                color: var(--md-accent-fg-color, #f59e0b);
                font-weight: 300;
            `;
            this.element.appendChild(cursor);
            this.cursorElement = cursor;
        }
        
        async type() {
            for (let i = 0; i < this.text.length; i++) {
                await this.wait(this.speed + Math.random() * 50);
                
                if (this.cursorElement) {
                    this.cursorElement.remove();
                }
                
                this.element.textContent += this.text[i];
                
                if (this.cursorElement) {
                    this.element.appendChild(this.cursorElement);
                }
            }
            
            if (this.loop) {
                await this.wait(this.loopDelay);
                this.element.textContent = '';
                if (this.cursorElement) {
                    this.element.appendChild(this.cursorElement);
                }
                this.type();
            } else if (this.cursorElement) {
                // Keep cursor blinking at the end
                this.cursorElement.style.animation = 'blink 1s infinite';
            }
        }
        
        wait(ms) {
            return new Promise(resolve => setTimeout(resolve, ms));
        }
    }
    
    // Initialize typewriter effects
    const typewriterElements = document.querySelectorAll('.typewriter');
    typewriterElements.forEach(element => {
        new TypewriterEffect(element, {
            speed: parseInt(element.dataset.speed) || 100,
            delay: parseInt(element.dataset.delay) || 500,
            cursor: element.dataset.cursor !== 'false',
            loop: element.dataset.loop === 'true'
        });
    });
    
    // ==============================================
    // HEADLINE ROTATION EFFECT
    // ==============================================
    
    class HeadlineRotator {
        constructor(element) {
            this.element = element;
            this.headlines = element.dataset.headlines ? 
                element.dataset.headlines.split('|') : 
                ['Context that feeds your prompts', 'Illuminate your codebase', 'Build better with AI'];
            this.currentIndex = 0;
            this.interval = parseInt(element.dataset.interval) || 3000;
            
            this.init();
        }
        
        init() {
            this.element.style.position = 'relative';
            this.element.style.overflow = 'hidden';
            this.element.style.height = this.element.offsetHeight + 'px';
            
            this.createHeadlineElements();
            this.startRotation();
        }
        
        createHeadlineElements() {
            this.headlineElements = this.headlines.map((text, index) => {
                const span = document.createElement('span');
                span.textContent = text;
                span.style.cssText = `
                    position: absolute;
                    width: 100%;
                    text-align: center;
                    transition: all 0.8s cubic-bezier(0.4, 0, 0.2, 1);
                    transform: ${index === 0 ? 'translateY(0)' : 'translateY(100%)'};
                    opacity: ${index === 0 ? '1' : '0'};
                `;
                this.element.appendChild(span);
                return span;
            });
        }
        
        startRotation() {
            setInterval(() => {
                const current = this.headlineElements[this.currentIndex];
                const nextIndex = (this.currentIndex + 1) % this.headlines.length;
                const next = this.headlineElements[nextIndex];
                
                // Slide current up and fade out
                current.style.transform = 'translateY(-100%)';
                current.style.opacity = '0';
                
                // Slide next in from bottom
                next.style.transform = 'translateY(0)';
                next.style.opacity = '1';
                
                this.currentIndex = nextIndex;
            }, this.interval);
        }
    }
    
    // Initialize headline rotators
    const rotatingHeadlines = document.querySelectorAll('.rotating-headline');
    rotatingHeadlines.forEach(element => {
        new HeadlineRotator(element);
    });
    
    // ==============================================
    // FADE-IN TEXT ANIMATION
    // ==============================================
    
    class FadeInText {
        constructor(element) {
            this.element = element;
            this.words = element.textContent.split(' ');
            this.delay = parseInt(element.dataset.fadeDelay) || 100;
            
            this.init();
        }
        
        init() {
            this.element.innerHTML = '';
            
            this.words.forEach((word, index) => {
                const span = document.createElement('span');
                span.textContent = word + ' ';
                span.style.cssText = `
                    display: inline-block;
                    opacity: 0;
                    transform: translateY(20px);
                    animation: fadeInUp 0.6s ease forwards;
                    animation-delay: ${index * this.delay}ms;
                `;
                this.element.appendChild(span);
            });
        }
    }
    
    // Initialize fade-in text elements
    const fadeInElements = document.querySelectorAll('.fade-in-text');
    fadeInElements.forEach(element => {
        // Use Intersection Observer to trigger animation when visible
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    new FadeInText(entry.target);
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.1 });
        
        observer.observe(element);
    });
    
    // ==============================================
    // NEWSPAPER DATE UPDATER
    // ==============================================
    
    function updateNewspaperDate() {
        const dateElement = document.getElementById('current-date');
        if (dateElement) {
            const options = { 
                weekday: 'long',
                year: 'numeric', 
                month: 'long', 
                day: 'numeric' 
            };
            const today = new Date();
            dateElement.textContent = today.toLocaleDateString('en-US', options).toUpperCase();
        }
        
        // Update version number from package.json or git tag
        const versionElement = document.getElementById('version-number');
        if (versionElement) {
            // This would typically fetch from your API or build process
            versionElement.textContent = '050'; // v0.5.0 as edition number
        }
    }
    
    updateNewspaperDate();
    
    // ==============================================
    // SCRAMBLE TEXT EFFECT
    // ==============================================
    
    class ScrambleText {
        constructor(element) {
            this.element = element;
            this.originalText = element.textContent;
            this.chars = '!<>-_\\/[]{}—=+*^?#________';
            this.duration = parseInt(element.dataset.scrambleDuration) || 2000;
            
            this.init();
        }
        
        init() {
            this.element.addEventListener('mouseenter', () => this.scramble());
        }
        
        scramble() {
            const iterations = this.duration / 50;
            let iteration = 0;
            
            const interval = setInterval(() => {
                this.element.textContent = this.originalText
                    .split('')
                    .map((char, index) => {
                        if (index < iteration) {
                            return this.originalText[index];
                        }
                        return this.chars[Math.floor(Math.random() * this.chars.length)];
                    })
                    .join('');
                
                if (iteration >= this.originalText.length) {
                    clearInterval(interval);
                }
                
                iteration += 1 / 3;
            }, 50);
        }
    }
    
    // Initialize scramble text elements
    const scrambleElements = document.querySelectorAll('.scramble-text');
    scrambleElements.forEach(element => {
        new ScrambleText(element);
    });
    
    // ==============================================
    // COUNTER ANIMATION
    // ==============================================
    
    class CounterAnimation {
        constructor(element) {
            this.element = element;
            this.target = parseInt(element.dataset.countTo) || 100;
            this.duration = parseInt(element.dataset.countDuration) || 2000;
            this.prefix = element.dataset.countPrefix || '';
            this.suffix = element.dataset.countSuffix || '';
            
            this.init();
        }
        
        init() {
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        this.animate();
                        observer.unobserve(entry.target);
                    }
                });
            }, { threshold: 0.5 });
            
            observer.observe(this.element);
        }
        
        animate() {
            const start = 0;
            const increment = this.target / (this.duration / 16);
            let current = start;
            
            const timer = setInterval(() => {
                current += increment;
                if (current >= this.target) {
                    current = this.target;
                    clearInterval(timer);
                }
                
                const display = this.formatNumber(Math.floor(current));
                this.element.textContent = this.prefix + display + this.suffix;
            }, 16);
        }
        
        formatNumber(num) {
            if (num >= 1000000) {
                return (num / 1000000).toFixed(1) + 'M';
            } else if (num >= 1000) {
                return (num / 1000).toFixed(1) + 'k';
            }
            return num.toString();
        }
    }
    
    // Initialize counter animations
    const counterElements = document.querySelectorAll('.counter');
    counterElements.forEach(element => {
        new CounterAnimation(element);
    });
    
    // ==============================================
    // TEXT REVEAL ON SCROLL
    // ==============================================
    
    class ScrollReveal {
        constructor() {
            this.elements = document.querySelectorAll('.reveal-on-scroll');
            this.init();
        }
        
        init() {
            const observer = new IntersectionObserver((entries) => {
                entries.forEach((entry, index) => {
                    if (entry.isIntersecting) {
                        setTimeout(() => {
                            entry.target.classList.add('revealed');
                        }, index * 100);
                    }
                });
            }, {
                threshold: 0.1,
                rootMargin: '0px 0px -50px 0px'
            });
            
            this.elements.forEach(element => {
                element.style.cssText = `
                    opacity: 0;
                    transform: translateY(30px);
                    transition: all 0.8s cubic-bezier(0.4, 0, 0.2, 1);
                `;
                observer.observe(element);
            });
        }
    }
    
    new ScrollReveal();
    
    // ==============================================
    // TYPING SOUND EFFECT (Optional)
    // ==============================================
    
    class TypewriterSound {
        constructor() {
            this.audioContext = null;
            this.enabled = localStorage.getItem('typewriterSound') !== 'false';
        }
        
        init() {
            if (!this.enabled || !window.AudioContext) return;
            
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
        
        playKeystroke() {
            if (!this.audioContext || !this.enabled) return;
            
            const oscillator = this.audioContext.createOscillator();
            const gainNode = this.audioContext.createGain();
            
            oscillator.connect(gainNode);
            gainNode.connect(this.audioContext.destination);
            
            oscillator.frequency.value = 800 + Math.random() * 400;
            oscillator.type = 'square';
            
            gainNode.gain.setValueAtTime(0.05, this.audioContext.currentTime);
            gainNode.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + 0.01);
            
            oscillator.start(this.audioContext.currentTime);
            oscillator.stop(this.audioContext.currentTime + 0.01);
        }
    }
    
    const typewriterSound = new TypewriterSound();
    
    // Add sound to typewriter effect if enabled
    if (typewriterElements.length > 0) {
        document.addEventListener('click', () => {
            typewriterSound.init();
        }, { once: true });
    }
    
});

// ==============================================
// CSS ANIMATIONS (to be added to stylesheet)
// ==============================================

const style = document.createElement('style');
style.textContent = `
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .revealed {
        opacity: 1 !important;
        transform: translateY(0) !important;
    }
    
    .typewriter-cursor {
        animation: blink 1s infinite;
    }
`;
document.head.appendChild(style);