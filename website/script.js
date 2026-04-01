/* ===== ToxPredict — Main Script ===== */

// Wait for everything to load
window.addEventListener('load', () => {
  initLoader();
  initLenis();
  initThreeJS();
  initCursor();
  initNavbar();
  initHeroAnimations();
  initScrollAnimations();
  initAccordion();
  initTypingDemo();
  initHamburger();
});

/* ===== LOADER ===== */
function initLoader() {
  const loader = document.getElementById('loader');
  gsap.to(loader, {
    opacity: 0,
    duration: 0.6,
    delay: 1.8,
    ease: 'power2.inOut',
    onComplete: () => {
      loader.style.display = 'none';
      document.body.style.overflow = '';
    }
  });
}

/* ===== LENIS SMOOTH SCROLL ===== */
let lenis;
function initLenis() {
  lenis = new Lenis({
    lerp: 0.1,
    duration: 1.2,
    smoothWheel: true,
  });

  lenis.on('scroll', ScrollTrigger.update);
  gsap.ticker.add((time) => lenis.raf(time * 1000));
  gsap.ticker.lagSmoothing(0);
}

/* ===== THREE.JS PARTICLE NETWORK ===== */
function initThreeJS() {
  const canvas = document.getElementById('hero-canvas');
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  const isMobile = window.innerWidth < 768;
  const particleCount = isMobile ? 50 : 150;
  const positions = [];
  const velocities = [];

  for (let i = 0; i < particleCount; i++) {
    positions.push(
      (Math.random() - 0.5) * 10,
      (Math.random() - 0.5) * 10,
      (Math.random() - 0.5) * 5
    );
    velocities.push(
      (Math.random() - 0.5) * 0.005,
      (Math.random() - 0.5) * 0.005,
      (Math.random() - 0.5) * 0.002
    );
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));

  const material = new THREE.PointsMaterial({
    color: 0x0ea5e9,
    size: 0.04,
    transparent: true,
    opacity: 0.8,
  });

  const points = new THREE.Points(geometry, material);
  scene.add(points);

  // Lines for connections
  const lineGeometry = new THREE.BufferGeometry();
  const lineMaterial = new THREE.LineBasicMaterial({
    color: 0x0ea5e9,
    transparent: true,
    opacity: 0.15,
  });
  const lines = new THREE.LineSegments(lineGeometry, lineMaterial);
  scene.add(lines);

  camera.position.z = 5;

  let mouseX = 0, mouseY = 0;
  document.addEventListener('mousemove', (e) => {
    mouseX = (e.clientX / window.innerWidth - 0.5) * 2;
    mouseY = (e.clientY / window.innerHeight - 0.5) * 2;
  });

  function animate() {
    requestAnimationFrame(animate);

    const pos = geometry.attributes.position.array;
    for (let i = 0; i < particleCount * 3; i += 3) {
      pos[i] += velocities[i];
      pos[i + 1] += velocities[i + 1];
      pos[i + 2] += velocities[i + 2];

      // Boundary bounce
      if (Math.abs(pos[i]) > 5) velocities[i] *= -1;
      if (Math.abs(pos[i + 1]) > 5) velocities[i + 1] *= -1;
      if (Math.abs(pos[i + 2]) > 2.5) velocities[i + 2] *= -1;
    }
    geometry.attributes.position.needsUpdate = true;

    // Update connections
    const linePositions = [];
    const connectionDistance = 1.5;
    for (let i = 0; i < particleCount; i++) {
      for (let j = i + 1; j < particleCount; j++) {
        const dx = pos[i * 3] - pos[j * 3];
        const dy = pos[i * 3 + 1] - pos[j * 3 + 1];
        const dz = pos[i * 3 + 2] - pos[j * 3 + 2];
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (dist < connectionDistance) {
          linePositions.push(
            pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2],
            pos[j * 3], pos[j * 3 + 1], pos[j * 3 + 2]
          );
        }
      }
    }
    lineGeometry.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));

    // Subtle camera movement with mouse
    camera.position.x += (mouseX * 0.3 - camera.position.x) * 0.02;
    camera.position.y += (-mouseY * 0.3 - camera.position.y) * 0.02;

    renderer.render(scene, camera);
  }
  animate();

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });
}

/* ===== CUSTOM CURSOR ===== */
function initCursor() {
  if (window.innerWidth < 768) return;

  const cursor = document.getElementById('custom-cursor');
  const follower = document.getElementById('cursor-follower');
  let cx = 0, cy = 0, fx = 0, fy = 0;

  document.addEventListener('mousemove', (e) => {
    cx = e.clientX;
    cy = e.clientY;
  });

  function animateCursor() {
    fx += (cx - fx) * 0.15;
    fy += (cy - fy) * 0.15;

    cursor.style.transform = `translate(${cx - 4}px, ${cy - 4}px)`;
    follower.style.transform = `translate(${fx - 16}px, ${fy - 16}px)`;
    requestAnimationFrame(animateCursor);
  }
  animateCursor();

  // Hover detection
  const interactives = document.querySelectorAll('a, button, .demo-pill, .feature-card, .problem-card, .step-card, .accordion-header');
  interactives.forEach(el => {
    el.addEventListener('mouseenter', () => document.body.classList.add('cursor-hover'));
    el.addEventListener('mouseleave', () => document.body.classList.remove('cursor-hover'));
  });
}

/* ===== NAVBAR ===== */
function initNavbar() {
  const navbar = document.getElementById('navbar');
  window.addEventListener('scroll', () => {
    if (window.scrollY > 100) {
      navbar.classList.add('scrolled');
    } else {
      navbar.classList.remove('scrolled');
    }
  });
}

/* ===== HAMBURGER ===== */
function initHamburger() {
  const hamburger = document.getElementById('hamburger');
  const navLinks = document.getElementById('nav-links');
  hamburger.addEventListener('click', () => {
    hamburger.classList.toggle('active');
    navLinks.classList.toggle('open');
  });
  // Close menu on link click
  navLinks.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', () => {
      hamburger.classList.remove('active');
      navLinks.classList.remove('open');
    });
  });
}

/* ===== HERO ANIMATIONS ===== */
function initHeroAnimations() {
  gsap.registerPlugin(ScrollTrigger, TextPlugin);

  const tl = gsap.timeline({ delay: 2 });

  // Navbar stagger in
  tl.from('.nav-logo, .nav-link, .nav-cta', {
    y: -30,
    opacity: 0,
    duration: 0.6,
    stagger: 0.1,
    ease: 'power3.out',
  }, 0);

  // Hero badge
  tl.from('#hero-badge', {
    y: 20,
    opacity: 0,
    duration: 0.6,
    ease: 'power3.out',
  }, 0.3);

  // Hero words clip-path reveal
  tl.from('.hero-word', {
    clipPath: 'inset(100% 0 0 0)',
    y: 40,
    duration: 0.8,
    stagger: 0.15,
    ease: 'power4.out',
  }, 0.5);

  // Subtext
  tl.from('#hero-sub', {
    opacity: 0,
    y: 20,
    duration: 0.6,
    ease: 'power3.out',
  }, 0.9);

  // Buttons
  tl.from('#hero-buttons .btn', {
    scale: 0.8,
    opacity: 0,
    duration: 0.5,
    stagger: 0.1,
    ease: 'back.out(1.5)',
  }, 1.1);

  // Stats count up
  tl.add(() => animateCounters('.hero-stats .stat-number'), 1.3);

  // Molecule visual
  tl.from('#hero-visual', {
    opacity: 0,
    scale: 0.6,
    duration: 1,
    ease: 'power3.out',
  }, 1.5);
}

/* ===== COUNT ANIMATION ===== */
function animateCounters(selector) {
  document.querySelectorAll(selector).forEach(el => {
    const target = parseFloat(el.dataset.target);
    const decimals = parseInt(el.dataset.decimals) || 0;
    const obj = { val: 0 };
    gsap.to(obj, {
      val: target,
      duration: 2,
      ease: 'power2.out',
      onUpdate: () => {
        if (decimals > 0) {
          el.textContent = obj.val.toFixed(decimals);
        } else {
          el.textContent = Math.round(obj.val).toLocaleString();
        }
      }
    });
  });
}

/* ===== SCROLL ANIMATIONS ===== */
function initScrollAnimations() {
  // Problem cards
  gsap.from('.problem-card', {
    scrollTrigger: { trigger: '.problem-cards', start: 'top 80%' },
    y: 60,
    opacity: 0,
    duration: 0.7,
    stagger: 0.2,
    ease: 'power3.out',
  });

  gsap.from('.massive-stat', {
    scrollTrigger: { trigger: '.massive-stat', start: 'top 85%' },
    scale: 0.5,
    opacity: 0,
    duration: 1,
    ease: 'power3.out',
  });

  // Transition text
  gsap.from('.transition-text', {
    scrollTrigger: { trigger: '.transition-text', start: 'top 85%' },
    opacity: 0,
    y: 30,
    duration: 0.8,
    ease: 'power3.out',
  });

  // Step cards
  gsap.from('.step-card', {
    scrollTrigger: { trigger: '.steps-wrapper', start: 'top 80%' },
    y: 50,
    opacity: 0,
    duration: 0.6,
    stagger: 0.15,
    ease: 'power3.out',
  });

  // Browser mockup float
  gsap.to('#browser-mockup', {
    y: -10,
    duration: 4,
    ease: 'sine.inOut',
    yoyo: true,
    repeat: -1,
  });

  // Feature cards flip in
  gsap.from('.feature-card', {
    scrollTrigger: { trigger: '.features-grid', start: 'top 80%' },
    rotateY: 90,
    opacity: 0,
    duration: 0.7,
    stagger: 0.12,
    ease: 'power3.out',
    transformOrigin: 'left center',
  });

  // Bar chart
  const chartBars = document.querySelectorAll('.chart-bar');
  chartBars.forEach(bar => {
    const value = parseFloat(bar.dataset.value);
    const percentage = (value / 1) * 100;
    bar.style.setProperty('--bar-width', percentage);
  });

  ScrollTrigger.create({
    trigger: '#chart-container',
    start: 'top 80%',
    onEnter: () => {
      chartBars.forEach((bar, i) => {
        setTimeout(() => bar.classList.add('animated'), i * 80);
      });
    },
    once: true,
  });

  // Result stats count
  ScrollTrigger.create({
    trigger: '.result-stats',
    start: 'top 85%',
    onEnter: () => animateCounters('.rs-number'),
    once: true,
  });

  // Timeline SVG draw
  ScrollTrigger.create({
    trigger: '.timeline-track',
    start: 'top 80%',
    onEnter: () => {
      const progress = document.querySelector('.timeline-progress');
      if (progress) {
        progress.style.strokeDashoffset = '0';
      }
    },
    once: true,
  });

  // Timeline events
  gsap.from('.tl-event', {
    scrollTrigger: { trigger: '.timeline-events', start: 'top 80%' },
    y: 40,
    opacity: 0,
    duration: 0.6,
    stagger: 0.2,
    ease: 'power3.out',
  });

  // Limitations items
  gsap.from('.limits-col', {
    scrollTrigger: { trigger: '.limits-grid', start: 'top 80%' },
    x: (i) => i === 0 ? -40 : 40,
    opacity: 0,
    duration: 0.7,
    stagger: 0.2,
    ease: 'power3.out',
  });

  // About section
  gsap.from('.team-card', {
    scrollTrigger: { trigger: '.team-card', start: 'top 85%' },
    y: 40,
    opacity: 0,
    duration: 0.7,
    ease: 'power3.out',
  });

  // Footer CTA
  gsap.from('.footer-heading', {
    scrollTrigger: { trigger: '.footer-cta', start: 'top 80%' },
    y: 30,
    opacity: 0,
    duration: 0.7,
    ease: 'power3.out',
  });

  // Section titles
  gsap.utils.toArray('.section-title').forEach(title => {
    gsap.from(title, {
      scrollTrigger: { trigger: title, start: 'top 85%' },
      y: 30,
      opacity: 0,
      duration: 0.7,
      ease: 'power3.out',
    });
  });
}

/* ===== ACCORDION ===== */
function initAccordion() {
  document.querySelectorAll('.accordion-header').forEach(header => {
    header.addEventListener('click', () => {
      const item = header.parentElement;
      const isActive = item.classList.contains('active');

      // Close all
      document.querySelectorAll('.accordion-item').forEach(i => i.classList.remove('active'));

      // Toggle current
      if (!isActive) {
        item.classList.add('active');
      }
    });
  });
}

/* ===== TYPING DEMO ===== */
function initTypingDemo() {
  const el = document.getElementById('typing-demo');
  if (!el) return;

  const text = 'Nitrobenzene';
  let i = 0;

  function type() {
    if (i < text.length) {
      el.textContent += text.charAt(i);
      i++;
      setTimeout(type, 120);
    } else {
      setTimeout(() => {
        el.textContent = '';
        i = 0;
        type();
      }, 3000);
    }
  }

  // Start after scrolling to this section
  ScrollTrigger.create({
    trigger: '#how-it-works',
    start: 'top 80%',
    onEnter: type,
    once: true,
  });
}

/* ===== SMOOTH SCROLL FOR NAV LINKS ===== */
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', (e) => {
    e.preventDefault();
    const target = document.querySelector(anchor.getAttribute('href'));
    if (target && lenis) {
      lenis.scrollTo(target, { offset: -80 });
    }
  });
});
// ============================================
// TOXPREDICT API INTEGRATION
// ============================================

// Change this URL after you deploy to Render
// For local testing: "http://localhost:8000"
// After deploying: "https://toxpredict-api.onrender.com"
const API_URL = "http://localhost:8000";

// Keep API alive (prevents Render free tier sleep)
setInterval(() => {
  fetch(`${API_URL}/health`).catch(() => {});
}, 14 * 60 * 1000);

// Store last result for CSV download
let lastResult = null;

// ---- Tab Switching ----
function switchTab(tab) {
  document.querySelectorAll('.predict-tab').forEach(t =>
    t.classList.remove('active')
  );
  document.querySelectorAll('.tab-content').forEach(t =>
    t.classList.remove('active')
  );

  document.getElementById(`tab-${tab}`).classList.add('active');
  document.getElementById(`tab-${tab}-content`).classList.add('active');
}

// ---- Set drug from quick pill ----
function setDrug(name) {
  switchTab('name');
  document.getElementById('drug-name-input').value = name;
  document.getElementById('drug-name-input').focus();
}

// ---- Handle predict by name ----
async function handlePredict() {
  const drugName = document.getElementById('drug-name-input').value.trim();
  if (!drugName) {
    showError('Please enter a drug name.');
    return;
  }
  await runPrediction(drugName, null);
}

// ---- Handle predict by SMILES ----
async function handlePredictSmiles() {
  const smiles = document.getElementById('smiles-input').value.trim();
  if (!smiles) {
    showError('Please enter a SMILES string.');
    return;
  }
  await runPrediction(null, smiles);
}

// ---- Main prediction function ----
async function runPrediction(drugName, smiles) {
  // Show loading
  showLoading(true);
  hideResults();
  hideError();

  try {
    let data;

    if (drugName) {
      // Predict by drug name
      const response = await fetch(`${API_URL}/predict-by-name`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ drug_name: drugName })
      });
      data = await response.json();
    } else {
      // Predict by SMILES
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ smiles: smiles })
      });
      data = await response.json();
    }

    showLoading(false);

    if (!data.success) {
      showError(data.error || 'Could not process this molecule.');
      return;
    }

    // Save for CSV download
    lastResult = data;

    // Display results
    displayResults(data, drugName || 'Molecule');

  } catch (err) {
    showLoading(false);
    showError('Cannot connect to prediction server. Make sure the API is running.');
    console.error(err);
  }
}

// ---- Display all results ----
function displayResults(data, displayName) {
  const resultsEl = document.getElementById('predict-results');
  resultsEl.style.display = 'block';

  // Risk banner
  const banner   = document.getElementById('risk-banner');
  const riskPct  = (data.overall_risk * 100).toFixed(1);
  banner.className = `risk-banner ${data.risk_level.toLowerCase()}`;
  document.getElementById('risk-label').textContent =
    data.risk_level === 'HIGH'     ? '⚠️ HIGH RISK' :
    data.risk_level === 'MODERATE' ? '⚡ MODERATE RISK' :
    '✅ LOW RISK';
  document.getElementById('risk-score').textContent = riskPct + '%';
  document.getElementById('drug-name-display').textContent =
    displayName || '';

  // Radar chart
  drawRadarChart(data.predictions, data.risk_color);

  // Molecular properties
  const propsEl = document.getElementById('mol-props');
  propsEl.innerHTML = '';
  Object.entries(data.molecular_props).forEach(([key, val]) => {
    propsEl.innerHTML += `
      <div class="prop-item">
        <span class="prop-key">${key}</span>
        <span class="prop-val">${val}</span>
      </div>
    `;
  });

  // ADMET
  const admetEl = document.getElementById('admet-results');
  admetEl.innerHTML = '';
  const goodVals = ['Good', 'High', 'Likely', 'Yes', 'Low'];
  Object.entries(data.admet).forEach(([key, val]) => {
    const label = key.replace(/_/g, ' ')
                     .replace(/\b\w/g, c => c.toUpperCase());
    const isGood = goodVals.includes(val);
    admetEl.innerHTML += `
      <div class="admet-row-result">
        <span>${isGood ? '✅' : '⚠️'} ${label}</span>
        <span class="${isGood ? 'admet-val-good' : 'admet-val-bad'}">
          ${val}
        </span>
      </div>
    `;
  });

  // Per-assay breakdown
  const assayGrid = document.getElementById('assay-grid');
  assayGrid.innerHTML = '';
  Object.entries(data.predictions).forEach(([target, prob]) => {
    const pct   = (prob * 100).toFixed(1);
    const color = prob > 0.5 ? '#ef4444' :
                  prob > 0.3 ? '#f59e0b' :
                  '#10b981';
    assayGrid.innerHTML += `
      <div class="assay-card">
        <div class="assay-name">${target}</div>
        <div class="assay-score" style="color:${color}">${pct}%</div>
        <div class="assay-bar-wrap">
          <div class="assay-bar-fill"
               style="width:${pct}%;background:${color}">
          </div>
        </div>
      </div>
    `;
  });

  // Toxicophores
  const toxEl = document.getElementById('toxicophore-results');
  toxEl.innerHTML = '';
  if (data.toxicophores && Object.keys(data.toxicophores).length > 0) {
    Object.entries(data.toxicophores).forEach(([name, desc]) => {
      toxEl.innerHTML += `
        <div class="tox-alert-item">
          <span class="tox-alert-icon">⚠️</span>
          <div>
            <div class="tox-alert-name">${name}</div>
            <div class="tox-alert-desc">${desc}</div>
          </div>
        </div>
      `;
    });
  } else {
    toxEl.innerHTML =
      '<div class="tox-safe">✅ No known toxicophores detected</div>';
  }

  // GSAP animate in
  gsap.from('#predict-results', {
    opacity: 0, y: 30,
    duration: 0.6,
    ease: 'power3.out'
  });
  gsap.from('.assay-card', {
    opacity: 0, y: 20,
    stagger: 0.04,
    duration: 0.4,
    delay: 0.3,
    ease: 'power3.out'
  });

  // Scroll to results
  setTimeout(() => {
    if (lenis) {
      lenis.scrollTo(document.getElementById('predict-results'),
        { offset: -100 });
    }
  }, 300);
}

// ---- Draw Radar Chart ----
function drawRadarChart(predictions, color) {
  const ctx = document.getElementById('radar-chart');
  if (!ctx) return;

  if (window.radarInstance) {
    window.radarInstance.destroy();
  }

  window.radarInstance = new Chart(ctx, {
    type: 'radar',
    data: {
      labels: Object.keys(predictions),
      datasets: [{
        label: 'Toxicity Risk',
        data: Object.values(predictions),
        backgroundColor: color + '33',
        borderColor: color,
        borderWidth: 2,
        pointBackgroundColor: color,
        pointRadius: 4,
      }]
    },
    options: {
      scales: {
        r: {
          min: 0, max: 1,
          ticks: {
            color: '#94a3b8',
            stepSize: 0.25,
            backdropColor: 'transparent',
          },
          grid:        { color: '#1e293b' },
          angleLines:  { color: '#1e293b' },
          pointLabels: {
            color: '#f8fafc',
            font: { size: 10, family: 'JetBrains Mono' }
          }
        }
      },
      plugins: { legend: { display: false } }
    }
  });
}

// ---- Download CSV Report ----
function downloadReport() {
  if (!lastResult) return;

  const rows = [
    ['Target', 'Toxicity Probability', 'Risk Level'],
    ...Object.entries(lastResult.predictions).map(([t, v]) => [
      t,
      (v * 100).toFixed(1) + '%',
      v > 0.5 ? 'HIGH' : v > 0.3 ? 'MODERATE' : 'LOW'
    ]),
    [],
    ['Overall Risk', (lastResult.overall_risk * 100).toFixed(1) + '%',
     lastResult.risk_level],
    [],
    ['ADMET Property', 'Value'],
    ...Object.entries(lastResult.admet).map(([k, v]) => [k, v])
  ];

  const csv  = rows.map(r => r.join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = 'toxicity_report.csv';
  a.click();
  URL.revokeObjectURL(url);
}

// ---- UI Helpers ----
function showLoading(show) {
  document.getElementById('predict-loading').style.display =
    show ? 'block' : 'none';
}
function hideResults() {
  document.getElementById('predict-results').style.display = 'none';
}
function hideError() {
  document.getElementById('predict-error').style.display = 'none';
}
function showError(msg) {
  const el = document.getElementById('predict-error');
  document.getElementById('error-message').textContent = msg;
  el.style.display = 'flex';
}