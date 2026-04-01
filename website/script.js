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
  lucide.createIcons();
});

/* ===== LOADER ===== */
function initLoader() {
  const loader = document.getElementById('loader');
  gsap.to(loader, {
    opacity: 0,
    duration: 0.6,
    delay: 1.0,
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
    lerp: 0.15,
    duration: 0.8,
    smoothWheel: true,
    wheelMultiplier: 1,
    touchMultiplier: 1.5,
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
  const particleCount = isMobile ? 30 : 80;
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
    size: 0.06,
    transparent: true,
    opacity: 0.7,
    sizeAttenuation: true,
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
  }, { passive: true });

  let frameCount = 0;
  let cachedLinePositions = [];
  function animate() {
    requestAnimationFrame(animate);
    frameCount++;

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

    // Update connections every 3rd frame for performance
    if (frameCount % 3 === 0) {
      cachedLinePositions = [];
      const connDist = 2.0;
      const connDistSq = connDist * connDist;
      for (let i = 0; i < particleCount; i++) {
        for (let j = i + 1; j < particleCount; j++) {
          const dx = pos[i * 3] - pos[j * 3];
          const dy = pos[i * 3 + 1] - pos[j * 3 + 1];
          const dz = pos[i * 3 + 2] - pos[j * 3 + 2];
          const distSq = dx * dx + dy * dy + dz * dz;
          if (distSq < connDistSq) {
            cachedLinePositions.push(
              pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2],
              pos[j * 3], pos[j * 3 + 1], pos[j * 3 + 2]
            );
          }
        }
      }
      lineGeometry.setAttribute('position', new THREE.Float32BufferAttribute(cachedLinePositions, 3));
    }

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
  }, { passive: true });

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

/* ===== NAVBAR + SCROLL PROGRESS ===== */
function initNavbar() {
  const navbar = document.getElementById('navbar');
  const progressBar = document.getElementById('scroll-progress');
  const allNavLinks = document.querySelectorAll('.nav-link');
  const sections = document.querySelectorAll('section[id]');

  window.addEventListener('scroll', () => {
    if (window.scrollY > 100) {
      navbar.classList.add('scrolled');
    } else {
      navbar.classList.remove('scrolled');
    }

    // Scroll progress bar
    const scrollTop = window.scrollY;
    const docHeight = document.documentElement.scrollHeight - window.innerHeight;
    if (progressBar && docHeight > 0) {
      progressBar.style.width = ((scrollTop / docHeight) * 100) + '%';
    }

    // Active nav link highlighting
    let current = '';
    sections.forEach(section => {
      const sectionTop = section.offsetTop - 200;
      if (scrollTop >= sectionTop) {
        current = section.getAttribute('id');
      }
    });
    allNavLinks.forEach(link => {
      link.classList.remove('active');
      if (link.getAttribute('href') === '#' + current) {
        link.classList.add('active');
      }
    });
  }, { passive: true });
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

  const tl = gsap.timeline({ delay: 1.2 });

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
  const isCompare = document.getElementById('tab-compare') && document.getElementById('tab-compare').classList.contains('active');
  
  if (isCompare) {
    const input1 = document.getElementById('compare-drug1-input');
    const input2 = document.getElementById('compare-drug2-input');
    
    if (!input1.value) {
      input1.value = name;
      input1.focus();
    } else if (!input2.value) {
      input2.value = name;
      input2.focus();
    } else {
      input1.value = name;
      input2.value = '';
      input1.focus();
    }
  } else {
    switchTab('name');
    document.getElementById('drug-name-input').value = name;
    document.getElementById('drug-name-input').focus();
  }
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

// ---- Handle predict compare ----
async function handleCompare() {
  const d1 = document.getElementById('compare-drug1-input').value.trim();
  const d2 = document.getElementById('compare-drug2-input').value.trim();
  if (!d1 || !d2) {
    showError('Please enter both drug names to compare.');
    return;
  }
  await runComparePrediction(d1, d2);
}

// ---- Main compare function ----
async function runComparePrediction(drug1, drug2) {
  showLoading(true);
  hideResults();
  hideError();

  try {
    const p1 = fetch(`${API_URL}/predict-by-name`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ drug_name: drug1 })
    });
    const p2 = fetch(`${API_URL}/predict-by-name`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ drug_name: drug2 })
    });

    const [res1, res2] = await Promise.all([p1, p2]);
    const data1 = await res1.json();
    const data2 = await res2.json();

    showLoading(false);

    if (!data1.success) {
      showError(`Could not process drug 1: ${data1.error || 'Unknown error'}`);
      return;
    }
    if (!data2.success) {
      showError(`Could not process drug 2: ${data2.error || 'Unknown error'}`);
      return;
    }

    displayCompareResults(data1, data2, drug1, drug2);

  } catch (err) {
    showLoading(false);
    showError('Cannot connect to prediction server. Make sure the API is running.');
    console.error(err);
  }
}

function displayCompareResults(data1, data2, name1, name2) {
  const resultsEl = document.getElementById('compare-results');
  resultsEl.style.display = 'block';

  // Set up Risk Banners
  const setupBanner = (idx, data, name) => {
    const banner = document.getElementById(`compare-risk-banner-${idx}`);
    banner.className = `risk-banner ${data.risk_level.toLowerCase()}`;
    const riskPct = (data.overall_risk * 100).toFixed(1);
    
    document.getElementById(`compare-risk-label-${idx}`).innerHTML =
      data.risk_level === 'HIGH'     ? '<i data-lucide=\'skull\' class=\'icon-inline\'></i> HIGH RISK' :
      data.risk_level === 'MODERATE' ? '<i data-lucide=\'skull\' class=\'icon-inline\'></i> MODERATE RISK' :
      '<i data-lucide=\'check-circle\' class=\'icon-inline\'></i> LOW RISK';
    document.getElementById(`compare-risk-score-${idx}`).textContent = riskPct + '%';
    document.getElementById(`compare-drug-name-${idx}`).textContent = name;
  };
  
  setupBanner(1, data1, name1);
  setupBanner(2, data2, name2);

  if (window.lucide) {
    lucide.createIcons();
  }

  // Radar chart
  drawCompareRadarChart(data1, data2, name1, name2);

  // Molecular properties
  const propsEl = document.getElementById('compare-mol-props');
  propsEl.innerHTML = `
    <div class="prop-item" style="display:grid; grid-template-columns:1fr 1fr 1fr; align-items:center; gap:8px; border-bottom:1px solid rgba(255,255,255,0.1); padding-bottom:8px; margin-bottom:8px; font-weight:bold;">
      <span class="prop-key">Property</span>
      <span class="prop-val" style="color:#0ea5e9; text-align:right;">${name1}</span>
      <span class="prop-val" style="color:#8b5cf6; text-align:right;">${name2}</span>
    </div>
  `;
  const allKeys = [...new Set([...Object.keys(data1.molecular_props), ...Object.keys(data2.molecular_props)])];
  allKeys.forEach(key => {
    const label = key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    const val1 = data1.molecular_props[key] !== undefined ? data1.molecular_props[key] : '-';
    const val2 = data2.molecular_props[key] !== undefined ? data2.molecular_props[key] : '-';
    propsEl.innerHTML += `
      <div class="prop-item" style="display:grid; grid-template-columns:1fr 1fr 1fr; align-items:center; gap:8px;">
        <span class="prop-key" style="display:flex; align-items:center; gap:8px;"><i data-lucide="beaker" style="width:14px; height:14px; opacity:0.5;"></i> ${label}</span>
        <span class="prop-val" style="color:var(--text); text-align:right;">${val1}</span>
        <span class="prop-val" style="color:var(--text); text-align:right;">${val2}</span>
      </div>
    `;
  });

  // ADMET
  const admetEl = document.getElementById('compare-admet-results');
  admetEl.innerHTML = `
    <div class="admet-row-result" style="display:grid; grid-template-columns:1fr 1fr 1fr; align-items:center; gap:8px; border-bottom:1px solid rgba(255,255,255,0.1); padding-bottom:8px; margin-bottom:8px; font-weight:bold; font-size:14px;">
      <span>Property</span>
      <span style="color:#0ea5e9; text-align:right;">${name1}</span>
      <span style="color:#8b5cf6; text-align:right;">${name2}</span>
    </div>
  `;
  const goodVals = ['Good', 'High', 'Likely', 'Yes', 'Low'];
  const allAdmetKeys = [...new Set([...Object.keys(data1.admet), ...Object.keys(data2.admet)])];
  allAdmetKeys.forEach(key => {
    const val1 = data1.admet[key] || '-';
    const val2 = data2.admet[key] || '-';
    const label = key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    
    const style1 = goodVals.includes(val1) ? 'admet-val-good' : 'admet-val-bad';
    const style2 = goodVals.includes(val2) ? 'admet-val-good' : 'admet-val-bad';
    
    admetEl.innerHTML += `
      <div class="admet-row-result" style="display:grid; grid-template-columns:1fr 1fr 1fr; align-items:center; gap:8px;">
        <span>${label}</span>
        <span class="${style1}" style="text-align:right;">${val1}</span>
        <span class="${style2}" style="text-align:right;">${val2}</span>
      </div>
    `;
  });

  // Per-assay breakdown
  const assayGrid = document.getElementById('compare-assay-grid');
  assayGrid.innerHTML = '';
  const allTargets = [...new Set([...Object.keys(data1.predictions), ...Object.keys(data2.predictions)])];
  allTargets.forEach(target => {
    const prob1 = data1.predictions[target] || 0;
    const prob2 = data2.predictions[target] || 0;
    const pct1 = (prob1 * 100).toFixed(1);
    const pct2 = (prob2 * 100).toFixed(1);
    
    const color1 = prob1 > 0.5 ? '#ef4444' : prob1 > 0.3 ? '#f59e0b' : '#0ea5e9'; // cyan for primary
    const color2 = prob2 > 0.5 ? '#ef4444' : prob2 > 0.3 ? '#f59e0b' : '#8b5cf6'; // purple for secondary

    assayGrid.innerHTML += `
      <div class="assay-card" style="display:flex; flex-direction:column; gap:8px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div class="assay-name">${target}</div>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
            <span style="min-width:30px; font-size:12px; color:#0ea5e9; font-weight:600;">${(name1||'D1').substring(0,3).toUpperCase()}</span>
            <span style="min-width:40px; font-size:12px; color:var(--text); text-align:right;">${pct1}%</span>
            <div class="assay-bar-wrap" style="flex:1;">
              <div class="assay-bar-fill" style="width:${pct1}%;background:${color1}"></div>
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
            <span style="min-width:30px; font-size:12px; color:#8b5cf6; font-weight:600;">${(name2||'D2').substring(0,3).toUpperCase()}</span>
            <span style="min-width:40px; font-size:12px; color:var(--text); text-align:right;">${pct2}%</span>
            <div class="assay-bar-wrap" style="flex:1;">
              <div class="assay-bar-fill" style="width:${pct2}%;background:${color2}"></div>
            </div>
        </div>
      </div>
    `;
  });

  gsap.from('#compare-results', {
    opacity: 0, y: 30, duration: 0.6, ease: 'power3.out'
  });
  
  // Re-run lucide icons for newly injected elements
  if (window.lucide) {
    lucide.createIcons();
  }
  
  setTimeout(() => {
    if (lenis) {
      lenis.scrollTo(document.getElementById('compare-results'), { offset: -100 });
    }
  }, 300);
}

function drawCompareRadarChart(data1, data2, name1, name2) {
  const ctx = document.getElementById('compare-radar-chart');
  if (!ctx) return;
  if (window.compareRadarInstance) window.compareRadarInstance.destroy();

  const labels = [...new Set([...Object.keys(data1.predictions), ...Object.keys(data2.predictions)])];
  const d1 = labels.map(l => data1.predictions[l] || 0);
  const d2 = labels.map(l => data2.predictions[l] || 0);

  const c1 = '#0ea5e9'; // Cyan
  const c2 = '#8b5cf6'; // Purple

  window.compareRadarInstance = new Chart(ctx, {
    type: 'radar',
    data: {
      labels: labels,
      datasets: [
        {
          label: name1 || 'Drug 1',
          data: d1,
          backgroundColor: c1 + '33',
          borderColor: c1,
          borderWidth: 2,
          pointBackgroundColor: c1,
        },
        {
          label: name2 || 'Drug 2',
          data: d2,
          backgroundColor: c2 + '33',
          borderColor: c2,
          borderWidth: 2,
          pointBackgroundColor: c2,
        }
      ]
    },
    options: {
      scales: {
        r: {
          min: 0, max: 1,
          ticks: { color: '#94a3b8', stepSize: 0.25, backdropColor: 'transparent' },
          grid: { color: '#1e293b' },
          angleLines: { color: '#1e293b' },
          pointLabels: { color: '#f8fafc', font: { size: 10, family: 'JetBrains Mono' } }
        }
      },
      plugins: {
        legend: { display: true, labels: { color: '#fff' } },
        tooltip: {
          callbacks: {
            label: function(context) {
              return context.dataset.label + ': ' + (context.raw * 100).toFixed(1) + '%';
            }
          }
        }
      }
    }
  });
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
  document.getElementById('risk-label').innerHTML =
    data.risk_level === 'HIGH'     ? '<i data-lucide=\'skull\' class=\'icon-inline\'></i> HIGH RISK' :
    data.risk_level === 'MODERATE' ? '<i data-lucide=\'skull\' class=\'icon-inline\'></i> MODERATE RISK' :
    '<i data-lucide=\'check-circle\' class=\'icon-inline\'></i> LOW RISK';
  document.getElementById('risk-score').textContent = riskPct + '%';
  document.getElementById('drug-name-display').textContent =
    displayName || '';

  // Radar chart
  drawRadarChart(data.predictions, data.risk_color);

  // Molecular properties
  const propsEl = document.getElementById('mol-props');
  propsEl.innerHTML = '';
  Object.entries(data.molecular_props).forEach(([key, val]) => {
    const label = key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    propsEl.innerHTML += `
      <div class="prop-item">
        <span class="prop-key" style="display:flex; align-items:center; gap:8px;"><i data-lucide="beaker" style="width:14px; height:14px; opacity:0.5;"></i> ${label}</span>
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
        <span>${isGood ? '<i data-lucide="check-circle" class="icon-inline"></i>' : '<i data-lucide="skull" class="icon-inline"></i>'} ${label}</span>
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
          <i data-lucide="skull" class="tox-alert-icon"></i>
          <div>
            <div class="tox-alert-name">${name}</div>
            <div class="tox-alert-desc">${desc}</div>
          </div>
        </div>
      `;
    });
  } else {
    toxEl.innerHTML =
      '<div class="tox-safe"><i data-lucide="check-circle" class="icon-inline"></i> No known toxicophores detected</div>';
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

  // Re-run lucide icons for newly injected elements
  if (window.lucide) {
    lucide.createIcons();
  }

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
  const match = document.getElementById('compare-results');
  if (match) match.style.display = 'none';
}
function hideError() {
  document.getElementById('predict-error').style.display = 'none';
}
function showError(msg) {
  const el = document.getElementById('predict-error');
  document.getElementById('error-message').textContent = msg;
  el.style.display = 'flex';
}

/* ===== EASTER EGG: NyX-K1 ===== */
(function initEasterEgg() {
  // 1. Keyboard sequence: type "nyxk1"
  let keysPressed = '';
  const secretCode = 'nyxk1';
  
  window.addEventListener('keydown', (e) => {
    // Only track alphanumeric keys to avoid capturing modifiers or special commands
    if (e.key.length === 1 && e.key.match(/[a-z0-9]/i)) {
      keysPressed += e.key.toLowerCase();
      if (keysPressed.length > secretCode.length) {
        keysPressed = keysPressed.slice(-secretCode.length);
      }
      if (keysPressed === secretCode) {
        triggerEasterEgg();
        keysPressed = ''; 
      }
    } else {
      keysPressed = ''; // reset on space, backspace, etc.
    }
  });

  // 2. Click counter: Tab the logo 5 times rapidly
  const logo = document.getElementById('nav-logo');
  if (logo) {
    let clickCount = 0;
    let clickTimeout = null;
    
    logo.addEventListener('click', (e) => {
      e.preventDefault();
      clickCount++;
      
      if (clickTimeout) clearTimeout(clickTimeout);
      
      // Reset count after 1 second of inactivity
      clickTimeout = setTimeout(() => {
        clickCount = 0;
      }, 1000);

      if (clickCount === 5) {
        triggerEasterEgg();
        clickCount = 0;
      } else if (clickCount === 1) { // Normal link behavior on single click if it actually scrolled somewhere
        window.scrollTo({ top: 0, behavior: 'smooth' });
      }
    });
  }

  function triggerEasterEgg() {
    // Blast a cool console message for developers
    console.log('%c👾 ToxPredict Secret Unlocked! 👾', 'color: #8b5cf6; font-size: 20px; font-weight: bold; text-shadow: 0 0 10px #8b5cf6;');
    console.log('%cProudly contributed by NyX-K1', 'color: #0ea5e9; font-size: 16px;');
    
    // Popup toast notification
    const toast = document.createElement('div');
    toast.innerHTML = '<i data-lucide="sparkles" style="vertical-align: middle; margin-right: 8px;"></i> Easter Egg Found! Redirecting to NyX-K1...';
    toast.style.position = 'fixed';
    toast.style.bottom = '20px';
    toast.style.right = '20px';
    toast.style.background = 'linear-gradient(135deg, #2563eb, #8b5cf6)';
    toast.style.color = '#fff';
    toast.style.padding = '12px 24px';
    toast.style.borderRadius = '12px';
    toast.style.zIndex = '99999';
    toast.style.fontFamily = "'Space Grotesk', sans-serif";
    toast.style.fontWeight = 'bold';
    toast.style.boxShadow = '0 10px 30px rgba(139, 92, 246, 0.5)';
    toast.style.transform = 'translateY(100px)';
    toast.style.transition = 'transform 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275)';
    document.body.appendChild(toast);
    
    // Refresh SVG icons for the newly injected sparkles
    if (window.lucide) window.lucide.createIcons();
    
    // Slide up animation
    setTimeout(() => { toast.style.transform = 'translateY(0)'; }, 50);
    
    // Redirect out to GitHub
    setTimeout(() => {
      window.open('https://github.com/NyX-K1', '_blank');
    }, 1200);
    
    // Cleanup the DOM toast
    setTimeout(() => {
      toast.style.transform = 'translateY(100px)';
      setTimeout(() => toast.remove(), 500);
    }, 4000);
  }
})();