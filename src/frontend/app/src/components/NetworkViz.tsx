import { useCallback, useRef, useState } from 'react'

const W = 1200
const H = 700
const CX = W / 2
const CY = H / 2 - 10
const OUTER_R = 265
const N_OUTER = 72
const N_INNER = 18

const GENE_LABELS: Record<number, string> = {
  0:  'EGFR',
  10: 'BRCA1',
  20: 'KIT',
  30: 'AR',
  40: 'ALK',
  54: 'MET',
  64: 'ERBB2',
}

interface Dot {
  id: number
  x: number
  y: number
  label?: string
  r: number
}

type Conn = [number, number] // -1 = center

// Deterministic layout — no Math.random, same every render
function buildDots(): Dot[] {
  const dots: Dot[] = []

  for (let i = 0; i < N_OUTER; i++) {
    const angle = (i / N_OUTER) * Math.PI * 2 - Math.PI / 2
    const rv = OUTER_R + Math.sin(i * 4.1) * 22 + Math.sin(i * 7.3) * 8
    dots.push({
      id: i,
      x: CX + Math.cos(angle) * rv,
      y: CY + Math.sin(angle) * rv,
      label: GENE_LABELS[i],
      r: 2 + Math.abs(Math.sin(i * 1.7)) * 1.8,
    })
  }

  for (let i = 0; i < N_INNER; i++) {
    const angle = (i / N_INNER) * Math.PI * 2 + 0.3
    const rv = 65 + Math.abs(Math.sin(i * 2.3)) * 120
    dots.push({
      id: N_OUTER + i,
      x: CX + Math.cos(angle) * rv,
      y: CY + Math.sin(angle) * rv,
      r: 1.5 + Math.abs(Math.sin(i * 3.1)) * 1.2,
    })
  }

  return dots
}

function buildConns(): Conn[] {
  const c: Conn[] = []

  // All outer dots → center
  for (let i = 0; i < N_OUTER; i++) c.push([i, -1])

  // Adjacent ring connections
  for (let i = 0; i < N_OUTER; i++) c.push([i, (i + 1) % N_OUTER])

  // Cross-connections every 6th dot
  for (let i = 0; i < N_OUTER; i += 6) {
    c.push([i, (i + 7) % N_OUTER])
    c.push([i, (i + 14) % N_OUTER])
  }

  // Inner dots → center + nearest outer
  for (let i = 0; i < N_INNER; i++) {
    c.push([N_OUTER + i, -1])
    c.push([N_OUTER + i, Math.round((i / N_INNER) * N_OUTER) % N_OUTER])
  }

  return c
}

const DOTS = buildDots()
const CONNS = buildConns()
const DOT_MAP = new Map<number, Dot>(DOTS.map(d => [d.id, d]))

function dotOp(dot: Dot, mouse: { x: number; y: number } | null): number {
  if (!mouse) return 0.2
  const d = Math.hypot(dot.x - mouse.x, dot.y - mouse.y)
  if (d < 80) return 1.0
  if (d < 160) return 0.5
  return 0.15
}

function centerOp(mouse: { x: number; y: number } | null): number {
  if (!mouse) return 0.7
  const d = Math.hypot(CX - mouse.x, CY - mouse.y)
  if (d < 80) return 1.0
  if (d < 160) return 0.7
  return 0.5
}

function lineOp(a: number, b: number, mouse: { x: number; y: number } | null): number {
  const da = DOT_MAP.get(a)
  const db = b === -1 ? null : DOT_MAP.get(b)
  const oa = da ? dotOp(da, mouse) : 0.15
  const ob = db ? dotOp(db, mouse) : centerOp(mouse)
  return Math.max(oa, ob) * 0.35
}

export default function NetworkViz() {
  const svgRef = useRef<SVGSVGElement>(null)
  const [mouse, setMouse] = useState<{ x: number; y: number } | null>(null)

  const onMove = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    const rect = svgRef.current?.getBoundingClientRect()
    if (!rect) return
    setMouse({
      x: (e.clientX - rect.left) * (W / rect.width),
      y: (e.clientY - rect.top) * (H / rect.height),
    })
  }, [])

  const onLeave = useCallback(() => setMouse(null), [])

  // Parallax
  const px = mouse ? (mouse.x - CX) * 0.022 : 0
  const py = mouse ? (mouse.y - CY) * 0.022 : 0

  const cop = centerOp(mouse)

  return (
    <svg
      ref={svgRef}
      viewBox={`0 0 ${W} ${H}`}
      className="w-full h-full"
      preserveAspectRatio="xMidYMid meet"
      onMouseMove={onMove}
      onMouseLeave={onLeave}
      style={{ display: 'block' }}
    >
      <defs>
        <radialGradient id="glowCenter" cx="50%" cy="50%" r="50%">
          <stop offset="0%"   stopColor="white" stopOpacity="0.9" />
          <stop offset="35%"  stopColor="white" stopOpacity="0.3" />
          <stop offset="100%" stopColor="white" stopOpacity="0"   />
        </radialGradient>
      </defs>

      <g transform={`translate(${px.toFixed(2)}, ${py.toFixed(2)})`}>
        {/* Lines */}
        {CONNS.map(([a, b], i) => {
          const da = DOT_MAP.get(a)
          if (!da) return null
          const x2 = b === -1 ? CX : (DOT_MAP.get(b)?.x ?? CX)
          const y2 = b === -1 ? CY : (DOT_MAP.get(b)?.y ?? CY)
          const op = lineOp(a, b, mouse)
          return (
            <line
              key={i}
              x1={da.x.toFixed(1)} y1={da.y.toFixed(1)}
              x2={x2.toFixed(1)}   y2={y2.toFixed(1)}
              stroke="white"
              strokeWidth="0.6"
              opacity={op}
            />
          )
        })}

        {/* Center glow */}
        <circle cx={CX} cy={CY} r="55" fill="url(#glowCenter)" opacity={cop * 0.55} />
        <circle cx={CX} cy={CY} r="7"  fill="white" opacity={cop} />

        {/* Dots + labels */}
        {DOTS.map(dot => {
          const op = dotOp(dot, mouse)
          const near = mouse !== null && Math.hypot(dot.x - mouse.x, dot.y - mouse.y) < 80
          const r = near ? dot.r * 1.5 : dot.r
          const labelOp = Math.min(op * 1.15, 1)
          const rightSide = dot.x >= CX
          return (
            <g key={dot.id} transform={`translate(${dot.x.toFixed(1)}, ${dot.y.toFixed(1)})`}>
              <circle cx={0} cy={0} r={r} fill="white" opacity={op} />
              {dot.label && (
                <text
                  x={rightSide ? 10 : -10}
                  y={4}
                  textAnchor={rightSide ? 'start' : 'end'}
                  fill="white"
                  opacity={labelOp}
                  fontSize="11"
                  fontFamily="ui-monospace, monospace"
                  fontWeight={near ? 'bold' : '500'}
                  style={{ pointerEvents: 'none', letterSpacing: '0.05em' }}
                >
                  {dot.label}
                </text>
              )}
            </g>
          )
        })}
      </g>
    </svg>
  )
}
