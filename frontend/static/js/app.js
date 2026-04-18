class EmotionDetector {
    constructor() {
        this.video = document.getElementById('webcam');
        this.canvas = document.getElementById('overlay');
        this.ctx = this.canvas.getContext('2d');
        this.stream = null;
        this.isRunning = false;
        this.animationId = null;
        this.lastTime = 0;
        this.fps = 0;
        
        this.emotionEmojis = {
            'angry': '😠',
            'disgust': '🤢',
            'fear': '😨',
            'happy': '😄',
            'neutral': '😐',
            'sad': '😢',
            'surprise': '😲'
        };

        this.emotionColors = {
            'angry': '#ef4444',
            'disgust': '#10b981',
            'fear': '#8b5cf6',
            'happy': '#f59e0b',
            'neutral': '#6b7280',
            'sad': '#3b82f6',
            'surprise': '#ec4899'
        };

        this.init();
    }

    init() {
        document.getElementById('startBtn').addEventListener('click', () => this.start());
        document.getElementById('stopBtn').addEventListener('click', () => this.stop());
        
        // Redimensionner le canvas quand la vidéo change de taille
        window.addEventListener('resize', () => this.resizeCanvas());
        
        this.updateStatus('Prêt', false);
    }

    async start() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                } 
            });
            
            this.video.srcObject = this.stream;
            
            this.video.onloadedmetadata = () => {
                this.isRunning = true;
                this.resizeCanvas();
                document.getElementById('placeholder').classList.add('hidden');
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
                this.updateStatus('Caméra active', true);
                this.detectLoop();
            };
            
        } catch (err) {
            console.error('Erreur accès caméra:', err);
            alert('Impossible d\'accéder à la caméra. Vérifiez les permissions.');
        }
    }

    stop() {
        this.isRunning = false;
        
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        this.video.srcObject = null;
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        document.getElementById('placeholder').classList.remove('hidden');
        document.getElementById('startBtn').disabled = false;
        document.getElementById('stopBtn').disabled = true;
        document.getElementById('resultsContent').classList.add('hidden');
        document.getElementById('noFaces').style.display = 'block';
        
        this.updateStatus('Arrêté', false);
    }

    resizeCanvas() {
        const rect = this.video.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
    }

    updateStatus(text, active) {
        document.getElementById('statusText').textContent = text;
        const dot = document.getElementById('statusDot');
        if (active) {
            dot.classList.add('active');
        } else {
            dot.classList.remove('active');
        }
    }

    async detectLoop() {
        if (!this.isRunning) return;

        const now = performance.now();
        const delta = now - this.lastTime;
        
        // Limiter à ~10 FPS pour l'API
        if (delta > 100) {
            this.fps = Math.round(1000 / delta);
            document.getElementById('fpsCounter').textContent = `${this.fps} FPS`;
            this.lastTime = now;
            
            await this.detectFrame();
        }
        
        this.animationId = requestAnimationFrame(() => this.detectLoop());
    }

    async detectFrame() {
        // Capturer l'image actuelle
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = this.video.videoWidth;
        tempCanvas.height = this.video.videoHeight;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(this.video, 0, 0);
        
        // Convertir en base64
        const imageData = tempCanvas.toDataURL('image/jpeg', 0.8);
        
        // Montrer le loading
        document.getElementById('loadingOverlay').classList.add('active');
        
        try {
            const startTime = performance.now();
            
            //modification de l'URL pour fonctionner en local et en prod
            const API_URL = window.location.origin;  // Fonctionne en local ET en prod
            const response = await fetch(`${API_URL}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageData })
            });
            
            const data = await response.json();
            const inferenceTime = Math.round(performance.now() - startTime);
            
            if (data.success && data.faces_detected > 0) {
                this.drawFaces(data.faces);
                this.updateResults(data.faces[0], inferenceTime, data.faces_detected);
            } else {
                this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                document.getElementById('resultsContent').classList.add('hidden');
                document.getElementById('noFaces').style.display = 'block';
            }
            
        } catch (err) {
            console.error('Erreur prédiction:', err);
        } finally {
            document.getElementById('loadingOverlay').classList.remove('active');
        }
    }

    drawFaces(faces) {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Calculer le ratio pour mapper les coordonnées
        const scaleX = this.canvas.width / this.video.videoWidth;
        const scaleY = this.canvas.height / this.video.videoHeight;
        
        faces.forEach(face => {
            const [x, y, w, h] = face.bbox;
            const color = this.emotionColors[face.primary_emotion] || '#6366f1';
            
            // Dessiner le rectangle (miroir horizontal)
            const mirroredX = this.canvas.width - (x * scaleX) - (w * scaleX);
            
            this.ctx.strokeStyle = color;
            this.ctx.lineWidth = 3;
            this.ctx.strokeRect(mirroredX, y * scaleY, w * scaleX, h * scaleY);
            
            // Label
            this.ctx.fillStyle = color;
            this.ctx.fillRect(mirroredX, y * scaleY - 30, w * scaleX, 30);
            
            this.ctx.fillStyle = 'white';
            this.ctx.font = 'bold 16px Inter, sans-serif';
            this.ctx.fillText(
                `${face.primary_emotion} ${face.primary_confidence.toFixed(1)}%`,
                mirroredX + 10,
                y * scaleY - 10
            );
        });
    }

    updateResults(face, inferenceTime, facesCount) {
        document.getElementById('noFaces').style.display = 'none';
        document.getElementById('resultsContent').classList.remove('hidden');
        
        // Mettre à jour l'émotion principale
        const emoji = this.emotionEmojis[face.primary_emotion] || '😐';
        document.getElementById('emotionIcon').textContent = emoji;
        document.getElementById('primaryEmotion').textContent = face.primary_emotion;
        document.getElementById('primaryEmotion').style.color = this.emotionColors[face.primary_emotion];
        
        // Barre de confiance
        const confidence = face.primary_confidence;
        document.getElementById('confidenceFill').style.width = `${confidence}%`;
        document.getElementById('confidenceText').textContent = `${confidence.toFixed(1)}%`;
        
        // Liste des probabilités
        const list = document.getElementById('probabilitiesList');
        list.innerHTML = '';
        
        const sorted = Object.entries(face.all_probabilities)
            .sort((a, b) => b[1] - a[1]);
        
        sorted.forEach(([emotion, prob]) => {
            const isActive = emotion === face.primary_emotion;
            const item = document.createElement('div');
            item.className = 'prob-item';
            item.innerHTML = `
                <span class="prob-label">${emotion}</span>
                <div class="prob-bar-container">
                    <div class="prob-bar ${isActive ? 'active' : ''}" style="width: ${prob}%"></div>
                </div>
                <span class="prob-value">${prob.toFixed(1)}%</span>
            `;
            list.appendChild(item);
        });
        
        // Stats
        document.getElementById('facesCount').textContent = facesCount;
        document.getElementById('inferenceTime').textContent = inferenceTime;
    }
}

// Initialiser l'application
document.addEventListener('DOMContentLoaded', () => {
    new EmotionDetector();
});