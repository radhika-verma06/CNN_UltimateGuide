import streamlit as st
import streamlit.components.v1 as components

def get_hero_component():
    """Renders the cinematic particle hero section."""
    html_code = """
    <div id="hero-container" style="height: 400px; position: relative; overflow: hidden; background: #04040a; border-radius: 20px; margin-bottom: 40px;">
        <canvas id="heroCanvas" style="position: absolute; inset: 0;"></canvas>
        <div style="position: relative; z-index: 2; text-align: center; padding-top: 100px;">
            <div style="font-family: 'Syne Mono', monospace; color: #00d4ff; font-size: 12px; letter-spacing: 4px; text-transform: uppercase; margin-bottom: 20px;">
                Deep Learning · Interpretability · Visualizer
            </div>
            <h1 style="font-family: 'Syne', sans-serif; font-size: 4.5rem; line-height: 0.9; margin-bottom: 20px; font-weight: 800; color: white;">
                CNN<br>
                <span style="background: linear-gradient(135deg, #00d4ff 0%, #a855f7 50%, #ff6b35 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Interpretability Lab</span>
            </h1>
            <p style="font-family: 'Outfit', sans-serif; color: #c4c4e0; font-size: 1.1rem; max-width: 500px; margin: 0 auto;">
                A cinematic deep-dive into the inner workings of convolutional neural networks.
            </p>
        </div>
    </div>
    <script>
        const canvas = document.getElementById('heroCanvas');
        const ctx = canvas.getContext('2d');
        let w, h, particles = [];
        function resize() { 
            w = canvas.width = canvas.parentElement.offsetWidth; 
            h = canvas.height = 400; 
        }
        resize();
        window.onresize = resize;
        for(let i=0; i<60; i++) particles.push({
            x: Math.random()*w, y: Math.random()*h,
            vx: (Math.random()-0.5)*0.5, vy: (Math.random()-0.5)*0.5,
            r: Math.random()*2, a: Math.random()
        });
        function animate() {
            ctx.clearRect(0,0,w,h);
            ctx.strokeStyle = 'rgba(0,212,255,0.05)';
            const gs = 60;
            for(let x=0; x<w; x+=gs) { ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,h); ctx.stroke(); }
            for(let y=0; y<h; y+=gs) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(w,y); ctx.stroke(); }
            particles.forEach((p, i) => {
                p.x += p.vx; p.y += p.vy;
                if(p.x<0) p.x=w; if(p.x>w) p.x=0;
                if(p.y<0) p.y=h; if(p.y>h) p.y=0;
                ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI*2);
                ctx.fillStyle = `rgba(0,212,255,${p.a*0.3})`; ctx.fill();
                particles.slice(i+1).forEach(q => {
                    const d = Math.hypot(p.x-q.x, p.y-q.y);
                    if(d<100) {
                        ctx.beginPath(); ctx.moveTo(p.x,p.y); ctx.lineTo(q.x,q.y);
                        ctx.strokeStyle = `rgba(0,212,255,${(1-d/100)*0.1})`; ctx.lineWidth=0.5; ctx.stroke();
                    }
                });
            });
            requestAnimationFrame(animate);
        }
        animate();
    </script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@800&family=Syne+Mono&family=Outfit&display=swap');
    </style>
    """
    components.html(html_code, height=400)

def get_3d_cube_component():
    """Renders the rotating 3D CSS cube for the Flatten layer."""
    html_code = """
    <div style="display: flex; justify-content: center; align-items: center; height: 300px; background: transparent;">
        <div class="cube-wrapper" style="perspective: 600px;">
            <div class="cube3d">
                <div class="cface front">5x5</div>
                <div class="cface back">5x5</div>
                <div class="cface right">x64</div>
                <div class="cface left">x64</div>
                <div class="cface top"></div>
                <div class="cface bottom"></div>
            </div>
        </div>
    </div>
    <style>
        .cube-wrapper { transform: scale(1.5); }
        .cube3d {
            width: 100px; height: 100px;
            transform-style: preserve-3d;
            animation: cubeRotate 8s linear infinite;
        }
        @keyframes cubeRotate {
            from { transform: rotateX(-20deg) rotateY(0deg); }
            to { transform: rotateX(-20deg) rotateY(360deg); }
        }
        .cface {
            position: absolute; width: 100px; height: 100px;
            border: 2px solid rgba(168,85,247,0.5);
            background: rgba(168,85,247,0.1);
            display: flex; align-items: center; justify-content: center;
            font-family: 'Syne Mono', monospace; font-size: 14px; color: #a855f7;
            box-shadow: inset 0 0 20px rgba(168,85,247,0.2);
        }
        .cface.front  { transform: translateZ(50px); }
        .cface.back   { transform: rotateY(180deg) translateZ(50px); }
        .cface.right  { transform: rotateY(90deg) translateZ(50px); }
        .cface.left   { transform: rotateY(-90deg) translateZ(50px); }
        .cface.top    { transform: rotateX(90deg) translateZ(50px); }
        .cface.bottom { transform: rotateX(-90deg) translateZ(50px); }
    </style>
    """
    components.html(html_code, height=300)

def get_dense_network_component():
    """Renders the animated SVG Dense network."""
    html_code = """
    <svg id="dense-svg" width="100%" height="300" style="background: transparent;"></svg>
    <script>
        const svg = document.getElementById('dense-svg');
        const W = svg.parentElement.offsetWidth, H = 300;
        const inN=8, hidN=6, outN=4;
        const iX=50, hX=W/2, oX=W-50;
        const getY = (i, tot) => (H/(tot+1))*(i+1);
        
        let content = '';
        for(let i=0; i<inN; i++) for(let j=0; j<hidN; j++) 
            content += `<line x1="${iX}" y1="${getY(i,inN)}" x2="${hX}" y2="${getY(j,hidN)}" stroke="#a855f7" stroke-width="0.5" opacity="0.1"><animate attributeName="opacity" values="0.1;0.4;0.1" dur="${2+Math.random()*2}s" repeatCount="indefinite"/></line>`;
        for(let j=0; j<hidN; j++) for(let k=0; k<outN; k++)
            content += `<line x1="${hX}" y1="${getY(j,hidN)}" x2="${oX}" y2="${getY(k,outN)}" stroke="#00d4ff" stroke-width="0.5" opacity="0.1"><animate attributeName="opacity" values="0.1;0.4;0.1" dur="${2+Math.random()*2}s" repeatCount="indefinite"/></line>`;
        
        for(let i=0; i<inN; i++) content += `<circle cx="${iX}" cy="${getY(i,inN)}" r="4" fill="#04040a" stroke="#a855f7" stroke-width="1"/>`;
        for(let j=0; j<hidN; j++) content += `<circle cx="${hX}" cy="${getY(j,hidN)}" r="5" fill="#04040a" stroke="#ec4899" stroke-width="1"/>`;
        svg.innerHTML = content;
    </script>
    """
    components.html(html_code, height=300)

def get_parameter_counter_component(total_params):
    """Renders the animated parameter counter."""
    html_code = f"""
    <div style="background: #0d0d1a; border: 1px solid rgba(255,255,255,0.06); border-radius: 20px; padding: 40px; text-align: center;">
        <div id="bigCounter" style="font-family: 'Syne', sans-serif; font-size: 72px; font-weight: 800; letter-spacing: -3px; background: linear-gradient(135deg, #22c55e, #00d4ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1;">0</div>
        <div style="font-family: 'Syne Mono', monospace; font-size: 12px; color: #7070a8; letter-spacing: 2px; margin-top: 10px; text-transform: uppercase;">Total Trainable Parameters</div>
    </div>
    <script>
        const target = {total_params};
        const el = document.getElementById('bigCounter');
        let current = 0;
        const step = target / 60;
        const timer = setInterval(() => {{
            current += step;
            if (current >= target) {{
                current = target;
                clearInterval(timer);
            }}
            el.textContent = Math.floor(current).toLocaleString();
        }}, 20);
    </script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@800&family=Syne+Mono&display=swap');
    </style>
    """
    components.html(html_code, height=200)

def get_quiz_component():
    """Renders the interactive quiz from the original HTML."""
    html_code = """
    <div style="background: #0d0d1a; border: 1px solid rgba(255,255,255,0.06); border-radius: 20px; padding: 40px; color: white; font-family: 'Outfit', sans-serif;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 20px;">
            <h3 style="font-family: 'Syne', sans-serif; font-size: 24px; margin: 0;">Quick-fire Quiz</h3>
            <div id="quizCounter" style="font-family: 'Syne Mono', monospace; color: #7070a8;">1 / 6</div>
        </div>
        <div id="quizQ" style="font-family: 'Syne', sans-serif; font-size: 20px; font-weight: 600; margin-bottom: 30px; line-height: 1.4;"></div>
        <div id="quizOpts" style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 30px;"></div>
        <div id="quizFb" style="display: none; padding: 20px; border-radius: 12px; font-size: 15px; line-height: 1.6; margin-bottom: 20px;"></div>
        <button id="quizNextBtn" style="display: none; font-family: 'Syne Mono', monospace; background: transparent; border: 1px solid #00d4ff; color: #00d4ff; padding: 12px 30px; border-radius: 6px; cursor: pointer; text-transform: uppercase; letter-spacing: 2px; transition: all 0.3s;">Next Question →</button>
    </div>
    <script>
        const quizzes = [
            {q:"A 28×28 image passes through Conv2D with a 3×3 filter (no padding). What is the output spatial size?", opts:["28×28","26×26","25×25","30×30"], ans:1, fb:"✓ Correct! 28 − 3 + 1 = 26. The filter can't hang off the edge — we lose 1 pixel on each side."},
            {q:"Pixel values are divided by 255 before training. Why?", opts:["To make images smaller","To normalise to 0–1 for stable training","CNNs only accept decimals","To remove noise"], ans:1, fb:"✓ Right! Large inputs cause large gradients and unstable weight updates. 0–1 range keeps training controlled."},
            {q:"Why can Conv layer outputs be negative even though pixels are 0–255?", opts:["Normalisation makes them negative","Filter weights are learned decimals — some are negative","ReLU causes negatives","MaxPooling introduces negatives"], ans:1, fb:"✓ Exactly! Filter weights (learned during training) can be negative. Positive pixel × negative weight = negative output."},
            {q:"MaxPooling2D(2,2) takes a 26×26 feature map. What is the output size?", opts:["24×24","28×28","13×13","52×52"], ans:2, fb:"✓ Correct! Pool size 2×2 halves both dimensions. 26 ÷ 2 = 13."},
            {q:"What does Flatten() actually do?", opts:["Removes negative values","Trains weights on the data","Reshapes 3D maps to a 1D vector","Applies softmax activation"], ans:2, fb:"✓ Just a reshape — no learning, no maths. It bridges convolutional layers to Dense layers."},
            {q:"Which single layer has the most parameters in this CNN?", opts:["Conv2D Layer 1 (640)","Conv2D Layer 2 (36,928)","Dense(128) (204,928)","MaxPooling (0)"], ans:2, fb:"✓ Dense(128) holds 84% of all parameters — 204,928 out of 243,786. That's exactly why conv+pooling compression matters."}
        ];
        let qIdx = 0;
        function initQuiz() {
            const q = quizzes[qIdx];
            document.getElementById('quizCounter').textContent = `${qIdx + 1} / ${quizzes.length}`;
            document.getElementById('quizQ').textContent = q.q;
            const opts = document.getElementById('quizOpts');
            opts.innerHTML = q.opts.map((o, i) => `<button class="q-opt" onclick="checkAns(${i},${q.ans},'${q.fb.replace(/'/g,"\\\\'")}')" style="background: #111120; border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 18px; color: white; cursor: pointer; text-align: left; transition: all 0.25s;">${o}</button>`).join('');
            document.getElementById('quizFb').style.display = 'none';
            document.getElementById('quizNextBtn').style.display = 'none';
        }
        window.checkAns = function(chosen, correct, fb) {
            const btns = document.querySelectorAll('.q-opt');
            btns.forEach(b => b.style.pointerEvents = 'none');
            btns[correct].style.borderColor = '#22c55e';
            btns[correct].style.background = 'rgba(34,197,94,0.1)';
            if(chosen !== correct) {
                btns[chosen].style.borderColor = '#ef4444';
                btns[chosen].style.background = 'rgba(239,68,68,0.1)';
            }
            const fbEl = document.getElementById('quizFb');
            fbEl.textContent = chosen === correct ? fb : '✗ Not quite. ' + fb.slice(2);
            fbEl.style.display = 'block';
            fbEl.style.background = chosen === correct ? 'rgba(34,197,94,0.1)' : 'rgba(239,68,68,0.1)';
            fbEl.style.borderColor = chosen === correct ? '#22c55e' : '#ef4444';
            fbEl.style.borderWidth = '1px';
            fbEl.style.borderStyle = 'solid';
            document.getElementById('quizNextBtn').style.display = 'inline-block';
        }
        document.getElementById('quizNextBtn').onclick = () => {
            qIdx++;
            if(qIdx < quizzes.length) initQuiz();
            else { qIdx = 0; initQuiz(); }
        }
        initQuiz();
    </script>
    <style>
        .q-opt:hover { border-color: #00d4ff !important; background: rgba(0,212,255,0.05) !important; transform: translateY(-2px); }
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;800&family=Syne+Mono&family=Outfit&display=swap');
    </style>
    """
    components.html(html_code, height=450)
