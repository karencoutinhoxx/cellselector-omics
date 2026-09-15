import { useCallback, useRef, useState } from 'react'

const SVG_W = 900
const SVG_H = 560
const CY = SVG_H / 2
const AMPLITUDE = 95
const CYCLES = 3
const N = 100          // segments
const BASE_R = 3.5
const RUNG_STEP = 5    // rung every N nodes

const LABELED: Record<number, string> = {
  8:  'EGFR',
  22: 'HCC827',
  35: 'BRCA1',
  48: 'KIT',
  60: 'AR',
  73: 'MCF-7',
  86: 'ALK',
}

interface Node {
  idx: number
  x: number
  y1: number
  y2: number
  label?: string
}

// Build node positions once outside component (deterministic)
function makeNodes(): Node[] {
  const pad = 28
  const nodes: Node[] = []
  for (let i = 0; i <= N; i++) {
    const t = i / N
    const x = pad + t * (SVG_W - 2 * pad)
    const phase = t * Math.PI * 2 * CYCLES
    nodes.push({
      idx:   i,
      x,
      y1: CY - AMPLITUDE * Math.sin(phase),
      y2: CY + AMPLITUDE * Math.sin(phase),
      label: LABELED[i],
    })
  }
  return nodes
}

// Catmull-Rom spline for smooth sine-wave path
function catmullRom(pts: { x: number; y: number }[]): string {
  const d = [`M ${pts[0].x.toFixed(1)} ${pts[0].y.toFixed(1)}`]
  for (let i = 1; i < pts.length; i++) {
    const p0 = pts[Math.max(0, i - 2)]
    const p1 = pts[i - 1]
    const p2 = pts[i]
    const p3 = pts[Math.min(pts.length - 1, i + 1)]
    const cp1x = p1.x + (p2.x - p0.x) / 6
    const cp1y = p1.y + (p2.y - p0.y) / 6
    const cp2x = p2.x - (p3.x - p1.x) / 6
    const cp2y = p2.y - (p3.y - p1.y) / 6
    d.push(`C ${cp1x.toFixed(1)} ${cp1y.toFixed(1)} ${cp2x.toFixed(1)} ${cp2y.toFixed(1)} ${p2.x.toFixed(1)} ${p2.y.toFixed(1)}`)
  }
  return d.join(' ')
}

const NODES = makeNodes()
const STRAND1_D = catmullRom(NODES.map(n => ({ x: n.x, y: n.y1 })))
const STRAND2_D = catmullRom(NODES.map(n => ({ x: n.x, y: n.y2 })))
const DOT_NODES = NODES.filter(n => n.idx % 5 === 0)
const RUNG_NODES = NODES.filter(n => n.idx % RUNG_STEP === 0 && n.idx > 0 && n.idx < N)

function dotStyle(x: number, y: number, mouse: { x: number; y: number } | null) {
  if (!mouse) return { r: BASE_R, op: 0.3 }
  const d = Math.hypot(x - mouse.x, y - mouse.y)
  if (d < 60) return { r: BASE_R * 2.2, op: 1.0 }
  if (d < 120) return { r: BASE_R * 1.5, op: 0.6 }
  return { r: BASE_R, op: 0.3 }
}

function rungOp(n: Node, mouse: { x: number; y: number } | null): number {
  if (!mouse) return 0.2
  const midY = (n.y1 + n.y2) / 2
  const d = Math.hypot(n.x - mouse.x, midY - mouse.y)
  if (d < 80)  return 0.85
  if (d < 150) return 0.45
  return 0.2
}

export default function DNAHelixHoriz() {
  const svgRef = useRef<SVGSVGElement>(null)
  const [mouse, setMouse] = useState<{ x: number; y: number } | null>(null)

  const onMove = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    const rect = svgRef.current?.getBoundingClientRect()
    if (!rect) return
    setMouse({
      x: (e.clientX - rect.left) * (SVG_W / rect.width),
      y: (e.clientY - rect.top) * (SVG_H / rect.height),
    })
  }, [])

  const onLeave = useCallback(() => setMouse(null), [])

  return (
    <>
      <style>{`
        @keyframes strand-pulse {
          0%, 100% { opacity: 0.72; }
          50%       { opacity: 0.95; }
        }
        @keyframes dot-bob {
          0%, 100% { transform: translateY(0px);  }
          50%       { transform: translateY(7px);  }
        }
        .helix-strand {
          animation: strand-pulse 5s ease-in-out infinite;
        }
        .helix-dot {
          animation: dot-bob 5s ease-in-out infinite;
          transform-box: fill-box;
          transform-origin: center;
        }
      `}</style>

      <svg
        ref={svgRef}
        viewBox={`0 0 ${SVG_W} ${SVG_H}`}
        className="w-full h-full"
        preserveAspectRatio="xMidYMid meet"
        onMouseMove={onMove}
        onMouseLeave={onLeave}
        style={{ display: 'block' }}
        aria-hidden="true"
      >
        {/* Rungs */}
        {RUNG_NODES.map(n => (
          <line
            key={`r-${n.idx}`}
            x1={n.x.toFixed(1)} y1={n.y1.toFixed(1)}
            x2={n.x.toFixed(1)} y2={n.y2.toFixed(1)}
            stroke="white"
            strokeWidth="1.1"
            opacity={rungOp(n, mouse)}
          />
        ))}

        {/* Strand 1 — top */}
        <path
          d={STRAND1_D}
          fill="none"
          stroke="white"
          strokeWidth="1.8"
          className="helix-strand"
        />

        {/* Strand 2 — bottom */}
        <path
          d={STRAND2_D}
          fill="none"
          stroke="white"
          strokeWidth="1.8"
          className="helix-strand"
          style={{ animationDelay: '2.5s' }}
        />

        {/* Strand-1 dots + labels */}
        {DOT_NODES.map(n => {
          const { r, op } = dotStyle(n.x, n.y1, mouse)
          const near = mouse !== null && Math.hypot(n.x - mouse.x, n.y1 - mouse.y) < 60
          const labelY = Math.max(14, n.y1 - r - 7)
          return (
            <g key={`s1-${n.idx}`}>
              <circle
                cx={n.x.toFixed(1)}
                cy={n.y1.toFixed(1)}
                r={r}
                fill="white"
                opacity={op}
                className="helix-dot"
                style={{ animationDelay: `${-(n.idx * 52)}ms` }}
              />
              {n.label && (
                <text
                  x={n.x.toFixed(1)}
                  y={labelY.toFixed(1)}
                  textAnchor="middle"
                  fill="white"
                  opacity={near ? 1 : 0}
                  fontSize="8.5"
                  fontFamily="ui-monospace, monospace"
                  fontWeight="600"
                  letterSpacing="0.05em"
                  style={{ pointerEvents: 'none', transition: 'opacity 0.12s' }}
                >
                  {n.label}
                </text>
              )}
            </g>
          )
        })}

        {/* Strand-2 dots */}
        {DOT_NODES.map(n => {
          const { r, op } = dotStyle(n.x, n.y2, mouse)
          return (
            <circle
              key={`s2-${n.idx}`}
              cx={n.x.toFixed(1)}
              cy={n.y2.toFixed(1)}
              r={r}
              fill="white"
              opacity={op}
              className="helix-dot"
              style={{ animationDelay: `${-(n.idx * 52 + 2500)}ms` }}
            />
          )
        })}
      </svg>
    </>
  )
}
