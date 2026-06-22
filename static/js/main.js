// ── Navbar toggle (mobile) ──────────────────────────────────────────────
const navToggle  = document.getElementById('navToggle');
const navLinks   = document.getElementById('navLinks');
if (navToggle) {
  navToggle.addEventListener('click', () => navLinks.classList.toggle('open'));
}

// Mobile dropdown toggle
document.querySelectorAll('.dropdown > a').forEach(a => {
  a.addEventListener('click', e => {
    if (window.innerWidth <= 768) {
      e.preventDefault();
      a.parentElement.classList.toggle('open');
    }
  });
});

// ── Counter animation ───────────────────────────────────────────────────
function animateCounter(el) {
  const target = parseFloat(el.dataset.target);
  const isFloat = el.dataset.float === '1';
  const dur = 1200;
  const start = performance.now();
  const update = (now) => {
    const p = Math.min((now - start) / dur, 1);
    const ease = 1 - Math.pow(1 - p, 3);
    const val = target * ease;
    el.textContent = isFloat ? val.toFixed(3) : Math.round(val);
    if (p < 1) requestAnimationFrame(update);
  };
  requestAnimationFrame(update);
}
document.querySelectorAll('[data-target]').forEach(el => {
  const obs = new IntersectionObserver(entries => {
    if (entries[0].isIntersecting) { animateCounter(el); obs.disconnect(); }
  });
  obs.observe(el);
});

// ── Scroll fade-in ──────────────────────────────────────────────────────
const fadeEls = document.querySelectorAll('.fade-in');
if (fadeEls.length) {
  const io = new IntersectionObserver(entries => {
    entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add('visible'); io.unobserve(e.target); }});
  }, { threshold: 0.1 });
  fadeEls.forEach(el => io.observe(el));
}
