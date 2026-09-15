interface Props {
  label: string
  value: number
}

export default function ScoreBar({ label, value }: Props) {
  const pct = Math.round(Math.min(1, Math.max(0, value || 0)) * 100)
  return (
    <div className="flex items-center gap-3 py-1">
      <span className="text-[#6E6E73] text-xs w-20 flex-shrink-0 truncate">{label}</span>
      <div className="flex-1 h-1.5 bg-[#F5F5F7] rounded-full overflow-hidden">
        <div
          className="h-full bg-[#1D1D1F] rounded-full transition-all duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-[#1D1D1F] font-mono text-xs w-10 text-right flex-shrink-0">
        {(value || 0).toFixed(2)}
      </span>
    </div>
  )
}
