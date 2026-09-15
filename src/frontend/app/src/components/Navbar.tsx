import { useEffect, useState } from 'react'
import { NavLink, useNavigate } from 'react-router-dom'

const PILL_STYLE: React.CSSProperties = {
  display: 'flex',
  gap: 8,
  padding: '10px 28px',
  borderRadius: 50,
  background: 'rgba(255,255,255,0.85)',
  backdropFilter: 'blur(20px) saturate(180%)',
  WebkitBackdropFilter: 'blur(20px) saturate(180%)',
  border: '1px solid rgba(0,0,0,0.10)',
  boxShadow: '0 8px 32px rgba(0,0,0,0.12), 0 2px 8px rgba(0,0,0,0.06), inset 0 1px 0 rgba(255,255,255,0.8)',
}

const LINK_BASE: React.CSSProperties = {
  fontSize: 14,
  fontWeight: 500,
  color: '#1D1D1F',
  textDecoration: 'none',
  borderRadius: 20,
  padding: '4px 12px',
  transition: 'background 0.2s',
}

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false)
  const navigate = useNavigate()

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20)
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  return (
    <nav
      style={{
        position: 'fixed',
        top: 0, left: 0, right: 0,
        zIndex: 50,
        transition: 'background 0.3s ease, border-color 0.3s ease, backdrop-filter 0.3s ease',
        background: scrolled ? 'rgba(255,255,255,0.85)' : 'transparent',
        backdropFilter: scrolled ? 'blur(20px)' : 'none',
        WebkitBackdropFilter: scrolled ? 'blur(20px)' : 'none',
        borderBottom: scrolled ? '1px solid rgba(0,0,0,0.06)' : '1px solid transparent',
      }}
    >
      <div style={{
        maxWidth: 1280, margin: '0 auto',
        padding: '16px 32px',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      }}>
        {/* Left — logo, outside pill */}
        <span
          onClick={() => navigate('/')}
          style={{
            fontWeight: 600, color: '#1D1D1F', fontSize: 15, letterSpacing: '-0.01em',
            cursor: 'pointer', userSelect: 'none', border: 'none', outline: 'none',
            background: 'none', textDecoration: 'none',
          }}
        >
          CellSelector Omics
        </span>

        {/* Centre — glass pill with Home + About only */}
        <div style={PILL_STYLE}>
          {[
            { to: '/',       label: 'Home',   end: true  },
            { to: '/about',  label: 'About',  end: false },
          ].map(({ to, label, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              style={({ isActive }) => ({
                ...LINK_BASE,
                fontWeight: isActive ? 600 : 500,
                background: 'transparent',
                textDecoration: 'none',
                transition: 'color 0.2s ease',
              })}
              onMouseEnter={e => { (e.currentTarget as HTMLElement).style.color = '#888888' }}
              onMouseLeave={e => { (e.currentTarget as HTMLElement).style.color = '#1D1D1F' }}
            >
              {label}
            </NavLink>
          ))}
        </div>

        {/* Right — circle arrow button */}
        <button
          onClick={() => navigate('/search')}
          aria-label="Open search tool"
          style={{
            width: 40, height: 40,
            borderRadius: '50%',
            background: '#1D1D1F',
            color: 'white',
            border: 'none',
            cursor: 'pointer',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 18,
            flexShrink: 0,
            transition: 'background 0.2s',
          }}
          onMouseEnter={e => (e.currentTarget.style.background = '#333333')}
          onMouseLeave={e => (e.currentTarget.style.background = '#1D1D1F')}
        >
          →
        </button>
      </div>
    </nav>
  )
}
