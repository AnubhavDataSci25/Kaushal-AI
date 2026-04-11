/* ============================================================
   Kaushal.AI – Frontend Application Script
   ============================================================ */

(function () {
  'use strict';

  // ── Experience range slider ──────────────────────────────
  const STAGGER_DELAY_MS = 80;   // stagger between fade-in cards
  const SCROLL_DELAY_MS  = 200;  // delay before auto-scrolling to result

  const slider = document.getElementById('experience_years');
  const expLabel = document.getElementById('expLabel');
  if (slider && expLabel) {
    slider.addEventListener('input', () => {
      const yrs = parseInt(slider.value, 10);
      expLabel.textContent = yrs + ' yr' + (yrs === 1 ? '' : 's');
    });
  }

  // ── Skill chips toggle ────────────────────────────────────
  const skillCountEl = document.getElementById('skillCount');

  function updateSkillCount() {
    const total = document.querySelectorAll('.skill-chip input:checked').length;
    if (skillCountEl) skillCountEl.textContent = total + ' selected';
    const errMsg = document.getElementById('skillsErrorMsg');
    if (errMsg && total > 0) errMsg.classList.remove('visible');
  }

  document.querySelectorAll('.skill-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const cb = chip.querySelector('input[type="checkbox"]');
      setTimeout(() => {
        chip.classList.toggle('selected', cb.checked);
        updateSkillCount();
      }, 0);
    });
  });

  // ── Client-side validation & loading overlay ─────────────
  const form = document.getElementById('careerForm');
  const loadingOverlay = document.getElementById('loadingOverlay');
  const submitBtn = document.getElementById('submitBtn');

  if (form) {
    form.addEventListener('submit', function (e) {
      // Bootstrap native validation
      form.classList.add('was-validated');

      // Skills must-have check
      const checkedSkills = document.querySelectorAll('.skill-chip input:checked').length;
      const skillsErr = document.getElementById('skillsErrorMsg');

      if (!form.checkValidity() || checkedSkills === 0) {
        e.preventDefault();
        e.stopPropagation();
        if (checkedSkills === 0 && skillsErr) {
          skillsErr.classList.add('visible');
          skillsErr.scrollIntoView({ behavior: 'smooth', block: 'center' });
        } else {
          // scroll to first invalid field
          const firstInvalid = form.querySelector(':invalid');
          if (firstInvalid) firstInvalid.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
        return;
      }

      // Show loading overlay
      if (loadingOverlay) loadingOverlay.classList.add('active');
      if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Analyzing…';
      }
    });
  }

  // ── Smooth scroll for CTA ─────────────────────────────────
  const ctaBtn = document.getElementById('ctaBtn');
  if (ctaBtn) {
    ctaBtn.addEventListener('click', () => {
      const formSection = document.getElementById('formSection');
      if (formSection) formSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  }

  // ── Scroll-triggered fade-in ──────────────────────────────
  const fadeEls = document.querySelectorAll('.fade-in-up');
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry, i) => {
        if (entry.isIntersecting) {
          setTimeout(() => entry.target.classList.add('visible'), i * STAGGER_DELAY_MS);
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.12 });
    fadeEls.forEach(el => observer.observe(el));
  } else {
    // Fallback for older browsers
    fadeEls.forEach(el => el.classList.add('visible'));
  }

  // ── Scroll to result on page load (if prediction exists) ──
  const resultEl = document.querySelector('.result-section');
  if (resultEl) {
    window.addEventListener('DOMContentLoaded', () => {
      setTimeout(() => resultEl.scrollIntoView({ behavior: 'smooth', block: 'start' }), SCROLL_DELAY_MS);
    });
  }

  // ── Navbar shadow on scroll ────────────────────────────────
  const navbar = document.querySelector('.navbar-kaushal');
  if (navbar) {
    window.addEventListener('scroll', () => {
      if (window.scrollY > 20) {
        navbar.style.boxShadow = '0 4px 24px rgba(108,99,255,0.13)';
      } else {
        navbar.style.boxShadow = '0 2px 16px rgba(108,99,255,0.07)';
      }
    });
  }

})();
