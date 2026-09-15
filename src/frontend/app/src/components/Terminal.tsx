import { useEffect, useState } from 'react'

const LINES = [
  '> Connecting to omics data layer...',
  '> Loading 2,076 human cell lines...',
  '> Indexing 222,196,849 expression records...',
  '> Verifying Cellosaurus nomenclature...',
  '> Models: classical [OK]  agentic [OK]',
  '> SYSTEM_READY_FOR_QUERY',
]

const CHAR_DELAY = 38
const LINE_PAUSE = 260

export default function Terminal() {
  const [rows, setRows] = useState<string[]>([''])
  const [lineIdx, setLineIdx] = useState(0)
  const [charIdx, setCharIdx] = useState(0)
  const [done, setDone] = useState(false)

  useEffect(() => {
    if (done) return
    if (lineIdx >= LINES.length) { setDone(true); return }
    const line = LINES[lineIdx]
    if (charIdx < line.length) {
      const t = setTimeout(() => {
        setRows(prev => {
          const next = [...prev]
          next[lineIdx] = line.slice(0, charIdx + 1)
          return next
        })
        setCharIdx(c => c + 1)
      }, CHAR_DELAY)
      return () => clearTimeout(t)
    }
    const t = setTimeout(() => {
      const isLast = lineIdx === LINES.length - 1
      setLineIdx(l => l + 1)
      setCharIdx(0)
      if (!isLast) setRows(prev => [...prev, ''])
    }, LINE_PAUSE)
    return () => clearTimeout(t)
  }, [lineIdx, charIdx, done])

  return (
    <div className="bg-black border border-[#2A2A2A] rounded p-4 font-mono text-xs leading-relaxed">
      {rows.map((row, i) => {
        const isFinal = done && i === LINES.length - 1
        const isTyping = !done && i === lineIdx
        return (
          <div key={i} className={isFinal ? 'text-white' : 'text-[#888888]'}>
            {row}
            {isTyping && <span className="animate-pulse ml-0.5 text-white">█</span>}
          </div>
        )
      })}
    </div>
  )
}
