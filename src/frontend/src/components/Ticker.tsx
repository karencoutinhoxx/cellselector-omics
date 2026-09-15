const CONTENT =
  'EGFR · HCC827 · BRCA1 · MCF-7 · KIT · Kasumi-1 · AR · LNCaP · ALK · NCI-H2228 · ERBB2 · SK-BR-3 · MET · EBC-1 · TP53 · '

export default function Ticker() {
  const repeated = CONTENT.repeat(3)
  return (
    <div className="bg-[#111111] overflow-hidden py-1.5 select-none border-t border-b border-[#2A2A2A]">
      <style>{`
        @keyframes ticker-scroll {
          0%   { transform: translateX(0); }
          100% { transform: translateX(-50%); }
        }
        .ticker-track {
          display: inline-block;
          white-space: nowrap;
          animation: ticker-scroll 35s linear infinite;
        }
      `}</style>
      <span className="ticker-track font-mono text-white text-xs tracking-widest">
        <span>{repeated}</span>
        <span>{repeated}</span>
      </span>
    </div>
  )
}
