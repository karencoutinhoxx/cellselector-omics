import { useEffect, useRef } from 'react'

// ── Geometry ──────────────────────────────────────────────────────────────────
const VB_W = 800
const VB_H = 900
const START_X = 0,   START_Y = 850
const END_X   = 800, END_Y   = 50
const N       = 100
const AMP     = 90
const CYCLES  = 3.5
const SPEED   = 0.008  // rad per frame — slow idle spin

// Diagonal axis angle → perpendicular direction
const AXIS_ANGLE = Math.atan2(END_Y - START_Y, END_X - START_X) // ≈ -45°
const PERP_ANGLE = AXIS_ANGLE + Math.PI / 2                      // ≈  45°
const COS_P      = Math.cos(PERP_ANGLE)                          // ≈ 0.707
const SIN_P      = Math.sin(PERP_ANGLE)                          // ≈ 0.707

const GENE_LABELS: Record<number, string> = {
  0: 'EGFR', 12: 'HCC827', 24: 'BRCA1', 36: 'KIT', 48: 'AR', 60: 'ALK', 72: 'MCF-7',
}

const ALL_IDX   = Array.from({ length: N + 1 }, (_, i) => i)
const NODE_IDX  = ALL_IDX.filter(i => i % 4 === 0)                    // 26 per strand
const RUNG_IDX  = ALL_IDX.filter(i => i % 6 === 0 && i > 0 && i < N) // 16 rungs
const LABEL_IDX = Object.keys(GENE_LABELS).map(Number)                // 7 labels

// ── Catmull-Rom spline ────────────────────────────────────────────────────────
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

// ── Props ─────────────────────────────────────────────────────────────────────
interface Props {
  mousePos: { x: number; y: number } | null  // fractional (0–1) relative to container
}

// ── Component ─────────────────────────────────────────────────────────────────
export default function DNAHelix({ mousePos }: Props) {
  const strand1Ref = useRef<SVGPathElement>(null)
  const strand2Ref = useRef<SVGPathElement>(null)
  const dot1Refs   = useRef<(SVGCircleElement | null)[]>([])
  const dot2Refs   = useRef<(SVGCircleElement | null)[]>([])
  const rungRefs   = useRef<(SVGLineElement | null)[]>([])
  const labelGRefs = useRef<(SVGGElement | null)[]>([])

  const phaseRef = useRef(0)
  const mouseRef = useRef<{ x: number; y: number } | null>(null)
  const rafRef   = useRef<number>(0)

  // Keep mouseRef in sync with prop — avoids stale closure in the RAF loop
  useEffect(() => {
    mouseRef.current = mousePos
  }, [mousePos])

  useEffect(() => {
    const animate = () => {
      phaseRef.current += SPEED
      const ph = phaseRef.current
      const m  = mouseRef.current
      const mx = m ? m.x * VB_W : null
      const my = m ? m.y * VB_H : null

      // Compute all N+1 strand positions
      const pos1: { x: number; y: number }[] = []
      const pos2: { x: number; y: number }[] = []
      for (let i = 0; i <= N; i++) {
        const t  = i / N
        const ax = START_X + t * (END_X - START_X)
        const ay = START_Y + t * (END_Y - START_Y)
        const p  = t * CYCLES * 2 * Math.PI + ph
        const s  = AMP * Math.sin(p)
        pos1.push({ x: ax + COS_P * s, y: ay + SIN_P * s })
        pos2.push({ x: ax - COS_P * s, y: ay - SIN_P * s })
      }

      // Strand paths
      strand1Ref.current?.setAttribute('d', catmullRom(pos1))
      strand2Ref.current?.setAttribute('d', catmullRom(pos2))

      // Nodes — depth illusion via z = cos(phase)
      NODE_IDX.forEach((idx, ni) => {
        const t = idx / N
        const p = t * CYCLES * 2 * Math.PI + ph
        const z = Math.cos(p)
        const pt1 = pos1[idx], pt2 = pos2[idx]

        const r1base  = Math.max(1, 3 + z * 2)
        const r2base  = Math.max(1, 3 - z * 2)
        const op1base = Math.max(0.3, Math.min(1, 0.7 + z * 0.3))
        const op2base = Math.max(0.3, Math.min(1, 0.7 - z * 0.3))

        let r1 = r1base, op1 = op1base
        let r2 = r2base, op2 = op2base

        if (mx !== null && my !== null) {
          if (Math.hypot(pt1.x - mx, pt1.y - my) < 70) { r1 = r1base * 2; op1 = 1 }
          if (Math.hypot(pt2.x - mx, pt2.y - my) < 70) { r2 = r2base * 2; op2 = 1 }
        }

        const c1 = dot1Refs.current[ni]
        if (c1) {
          c1.setAttribute('cx', pt1.x.toFixed(1))
          c1.setAttribute('cy', pt1.y.toFixed(1))
          c1.setAttribute('r',  r1.toFixed(1))
          c1.setAttribute('opacity', op1.toFixed(2))
        }
        const c2 = dot2Refs.current[ni]
        if (c2) {
          c2.setAttribute('cx', pt2.x.toFixed(1))
          c2.setAttribute('cy', pt2.y.toFixed(1))
          c2.setAttribute('r',  r2.toFixed(1))
          c2.setAttribute('opacity', op2.toFixed(2))
        }
      })

      // Rungs — green base pairs
      RUNG_IDX.forEach((idx, ri) => {
        const t   = idx / N
        const p   = t * CYCLES * 2 * Math.PI + ph
        const z   = Math.cos(p)
        const rOp = Math.max(0.25, 0.55 + z * 0.3)
        const line = rungRefs.current[ri]
        if (line) {
          line.setAttribute('x1', pos1[idx].x.toFixed(1))
          line.setAttribute('y1', pos1[idx].y.toFixed(1))
          line.setAttribute('x2', pos2[idx].x.toFixed(1))
          line.setAttribute('y2', pos2[idx].y.toFixed(1))
          line.setAttribute('stroke', '#2D6A4F')
          line.setAttribute('opacity', rOp.toFixed(2))
        }
      })

      // Gene labels — revealed on mouse proximity
      LABEL_IDX.forEach((idx, li) => {
        const pt   = pos1[idx]
        const show = mx !== null && my !== null && Math.hypot(pt.x - mx, pt.y - my) < 70
        const lg   = labelGRefs.current[li]
        if (lg) {
          lg.setAttribute('opacity', show ? '1' : '0')
          if (show) {
            ;(lg.children[0] as SVGRectElement)?.setAttribute('x', (pt.x - 24).toFixed(1))
            ;(lg.children[0] as SVGRectElement)?.setAttribute('y', (pt.y - 18).toFixed(1))
            ;(lg.children[1] as SVGTextElement)?.setAttribute('x', pt.x.toFixed(1))
            ;(lg.children[1] as SVGTextElement)?.setAttribute('y', (pt.y - 8).toFixed(1))
          }
        }
      })

      rafRef.current = requestAnimationFrame(animate)
    }

    rafRef.current = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(rafRef.current)
  }, [])

  return (
    <svg
      viewBox={`0 0 ${VB_W} ${VB_H}`}
      style={{ display: 'block', width: '100%', height: '100%' }}
      preserveAspectRatio="xMidYMid meet"
      aria-hidden="true"
    >
      {/* Rungs — drawn behind strands */}
      {RUNG_IDX.map((idx, ri) => (
        <line
          key={`rung-${idx}`}
          ref={el => { rungRefs.current[ri] = el }}
          x1="-2000" y1="-2000" x2="-2000" y2="-2000"
          stroke="#2D6A4F" strokeWidth="2"
        />
      ))}

      {/* Strand 1 — red */}
      <path
        ref={strand1Ref} d="M -2000 -2000"
        fill="none" stroke="#E63946" strokeWidth="3"
        strokeLinecap="round" opacity="0.9"
      />

      {/* Strand 2 — blue */}
      <path
        ref={strand2Ref} d="M -2000 -2000"
        fill="none" stroke="#457B9D" strokeWidth="3"
        strokeLinecap="round" opacity="0.9"
      />

      {/* Strand 1 nodes */}
      {NODE_IDX.map((idx, ni) => (
        <circle
          key={`d1-${idx}`}
          ref={el => { dot1Refs.current[ni] = el }}
          cx="-2000" cy="-2000" r="3"
          fill="white" stroke="#E63946" strokeWidth="1.5"
        />
      ))}

      {/* Strand 2 nodes */}
      {NODE_IDX.map((idx, ni) => (
        <circle
          key={`d2-${idx}`}
          ref={el => { dot2Refs.current[ni] = el }}
          cx="-2000" cy="-2000" r="3"
          fill="white" stroke="#457B9D" strokeWidth="1.5"
        />
      ))}

      {/* Gene labels — opacity driven by RAF hover check */}
      {LABEL_IDX.map((idx, li) => (
        <g
          key={`label-${idx}`}
          ref={el => { labelGRefs.current[li] = el }}
          opacity="0"
          style={{ pointerEvents: 'none' }}
        >
          <rect
            x="-2000" y="-2000" width="48" height="14" rx="7"
            fill="white" fillOpacity="0.95"
            filter="drop-shadow(0 1px 3px rgba(0,0,0,0.12))"
          />
          <text
            x="-2000" y="-2000"
            textAnchor="middle" fill="#1D1D1F"
            fontSize="8" fontFamily="ui-monospace, monospace" fontWeight="600"
          >
            {GENE_LABELS[idx]}
          </text>
        </g>
      ))}
    </svg>
  )
}
