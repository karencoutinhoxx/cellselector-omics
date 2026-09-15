import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useEffect } from 'react'
import { api } from '../api/client'
import DNAHelix from '../components/DNAHelix'

const LOREM_BODY = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.'

const PILLARS = [
  { title: 'Lorem Ipsum',        body: LOREM_BODY },
  { title: 'Dolor Sit',          body: LOREM_BODY },
  { title: 'Amet Consectetur',   body: LOREM_BODY },
]

const STEPS = [
  { n: '01', title: 'Lorem ipsum dolor sit amet' },
  { n: '02', title: 'Consectetur adipiscing elit' },
  { n: '03', title: 'Sed do eiusmod tempor' },
  { n: '04', title: 'Incididunt ut labore' },
]

const SOURCES = [
  { key: 'D1', name: 'Lorem Ipsum' },
  { key: 'D2', name: 'Lorem Ipsum' },
  { key: 'D3', name: 'Lorem Ipsum' },
  { key: 'D4', name: 'Lorem Ipsum' },
  { key: 'D5', name: 'Lorem Ipsum' },
]

export default function Home() {
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null)
  const [health, setHealth] = useState<any>(null)
  const [stats, setStats]   = useState<any>(null)

  useEffect(() => {
    api.health().then(setHealth).catch(() => {})
    api.stats().then(setStats).catch(() => {})
  }, [])

  const statItems = [
    { value: health?.cell_lines ? health.cell_lines.toLocaleString() : '2,076', label: 'LOREM' },
    { value: String(health?.datasets ?? 4),                                      label: 'IPSUM' },
    { value: '222M',                                                              label: 'DOLOR' },
    { value: stats?.validation?.hit_at_10 != null ? `${Math.round(stats.validation.hit_at_10 * 100)}%` : '4/5', label: 'SIT' },
  ]

  return (
    <div className="bg-white">

      {/* ── Hero ── */}
      <section style={{ position: 'relative', height: '100vh', overflow: 'hidden', background: 'white' }}>

        {/* Zone 1 — DNA helix, left 55% */}
        <div
          style={{ position: 'absolute', left: 0, top: 0, width: '55%', height: '100%', overflow: 'hidden' }}
          onMouseMove={e => {
            const r = e.currentTarget.getBoundingClientRect()
            setMousePos({ x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height })
          }}
          onMouseLeave={() => setMousePos(null)}
        >
          <DNAHelix mousePos={mousePos} />
        </div>

        {/* Thin fade divider */}
        <div
          style={{
            position: 'absolute', left: '55%', top: '10%',
            height: '80%', width: 1, pointerEvents: 'none',
            background: 'linear-gradient(to bottom, transparent, #D2D2D7 20%, #D2D2D7 80%, transparent)',
          }}
        />

        {/* Zone 2 — text, right 45%, vertically centered */}
        <div
          style={{
            position: 'absolute', right: 0, top: 0,
            width: '45%', height: '100%',
            display: 'flex', flexDirection: 'column', justifyContent: 'center',
            padding: '0 60px 0 40px',
            background: 'white',
          }}
        >
          <p style={{
            fontFamily: 'ui-monospace, monospace', color: '#6E6E73',
            fontSize: 11, letterSpacing: '0.15em', textTransform: 'uppercase',
            marginBottom: 24,
          }}>
            University of Bristol × AstraZeneca
          </p>
          <h1 style={{ color: '#1D1D1F', fontSize: 64, fontWeight: 800, lineHeight: 1.05, marginBottom: 20 }}>
            Find the<br />right cell<br />line.
          </h1>
          <p style={{ color: '#6E6E73', fontSize: 17, lineHeight: 1.6, maxWidth: 380, marginBottom: 40 }}>
            Multi-omics recommendation across 2,076 human cell lines.
            Evidence-backed. Citable.
          </p>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
            <Link
              to="/search"
              style={{
                background: '#1D1D1F', color: 'white',
                borderRadius: 50, padding: '14px 28px',
                fontWeight: 600, fontSize: 15,
                textDecoration: 'none', display: 'inline-block',
              }}
            >
              Launch Tool →
            </Link>
            <Link
              to="/about"
              style={{
                background: 'transparent', border: '1.5px solid #D2D2D7',
                color: '#1D1D1F', borderRadius: 50, padding: '14px 28px',
                fontSize: 15, textDecoration: 'none', display: 'inline-block',
              }}
              onMouseEnter={e => { (e.currentTarget as HTMLElement).style.borderColor = '#1D1D1F' }}
              onMouseLeave={e => { (e.currentTarget as HTMLElement).style.borderColor = '#D2D2D7' }}
            >
              About
            </Link>
          </div>
        </div>

        <div
          className="absolute bottom-8 left-1/2 -translate-x-1/2 text-[#6E6E73] text-xs font-mono tracking-widest animate-bounce z-10"
          style={{ pointerEvents: 'none' }}
        >
          ↓ SCROLL
        </div>
      </section>

      {/* ── Stats ── */}
      <section className="bg-[#F5F5F7] py-14">
        <div className="max-w-5xl mx-auto px-6">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-8">
            {statItems.map(s => (
              <div key={s.label} className="text-center">
                <div className="text-[#1D1D1F] font-bold text-3xl mb-1 tabular-nums">
                  {s.value}
                </div>
                <div className="text-[#6E6E73] text-xs uppercase tracking-widest">{s.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Three pillars ── */}
      <section className="bg-white py-20">
        <div className="max-w-5xl mx-auto px-6">
          <h2 className="text-[#1D1D1F] text-3xl font-bold mb-12 text-center">Lorem ipsum dolor sit amet</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {PILLARS.map(p => (
              <div
                key={p.title}
                className="bg-white border border-[#D2D2D7] rounded-xl p-7"
                style={{ boxShadow: '0 2px 20px rgba(0,0,0,0.06)' }}
              >
                <div className="w-8 h-0.5 bg-[#D2D2D7] mb-5" />
                <h3 className="text-[#1D1D1F] font-bold text-lg mb-3">{p.title}</h3>
                <p className="text-[#6E6E73] text-sm leading-relaxed">{p.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Steps ── */}
      <section className="bg-[#F5F5F7] py-20">
        <div className="max-w-5xl mx-auto px-6">
          <h2 className="text-[#1D1D1F] text-3xl font-bold mb-12 text-center">Lorem ipsum</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8">
            {STEPS.map((step, i) => (
              <div key={step.n} className="relative">
                {i < STEPS.length - 1 && (
                  <div className="hidden lg:block absolute top-5 left-full w-full h-px bg-[#D2D2D7] z-0 -translate-x-4" />
                )}
                <div className="w-10 h-10 bg-[#1D1D1F] text-white rounded-full flex items-center justify-center text-sm font-bold mb-4">
                  {step.n}
                </div>
                <p className="text-[#6E6E73] text-sm leading-relaxed">{step.title}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Source cards ── */}
      <section className="bg-white py-20">
        <div className="max-w-5xl mx-auto px-6">
          <h2 className="text-[#1D1D1F] text-3xl font-bold mb-12 text-center">Lorem ipsum</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
            {SOURCES.map(s => (
              <div
                key={s.key}
                className="bg-white border border-[#D2D2D7] rounded-xl p-5"
                style={{ boxShadow: '0 1px 8px rgba(0,0,0,0.04)' }}
              >
                <div className="text-[#6E6E73] font-mono font-bold text-xs mb-2">[{s.key}]</div>
                <div className="text-[#1D1D1F] font-semibold text-sm mb-1">{s.name}</div>
                <div className="text-[#6E6E73] text-xs leading-relaxed">{LOREM_BODY.slice(0, 60)}…</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className="bg-[#F5F5F7] border-t border-[#D2D2D7] py-8">
        <div className="max-w-5xl mx-auto px-6 flex flex-col sm:flex-row justify-between items-center gap-3">
          <div className="text-[#6E6E73] text-xs">
            CellSelector Omics · University of Bristol × AstraZeneca · MSc Group Project 2026
          </div>
          <div className="text-[#6E6E73] font-mono text-xs">v1.0</div>
        </div>
      </footer>
    </div>
  )
}
