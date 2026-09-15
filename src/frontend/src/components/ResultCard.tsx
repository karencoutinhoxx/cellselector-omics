import { useState } from 'react'
import { api } from '../api/client'
import ScoreBar from './ScoreBar'

interface Props {
  result: any
  gene: string
  diseaseFilter?: string
  excludeGenes?: string[]
  onCellLineClick: (cvcl: string) => void
}

const SECTION_LABELS = [
  'RECOMMENDATION', 'KEY REASON', 'EVIDENCE SUMMARY',
  'TRADE-OFFS', 'BEST FOR', 'DATA CITATIONS',
  'DATA SOURCES', 'LITERATURE CITATIONS', 'LITERATURE',
]

const MONO_SECTIONS = new Set(['DATA SOURCES', 'LITERATURE CITATIONS', 'LITERATURE', 'DATA CITATIONS'])

function parseJustification(text: string): Record<string, string> {
  const sections: Record<string, string> = {}
  const lines = text.split('\n')
  let currentLabel = ''
  let currentContent: string[] = []

  for (const line of lines) {
    const trimmed = line.trim()
    if (!trimmed) continue

    const matchedLabel = SECTION_LABELS.find(lbl => trimmed.toUpperCase().includes(lbl))

    if (matchedLabel) {
      if (currentLabel) sections[currentLabel] = currentContent.join(' ').trim()
      currentLabel = matchedLabel
      const afterColon = trimmed.split(':').slice(1).join(':').trim()
      currentContent = afterColon ? [afterColon] : []
    } else if (currentLabel) {
      currentContent.push(trimmed)
    }
  }

  if (currentLabel) sections[currentLabel] = currentContent.join(' ').trim()
  return sections
}

function levelColor(level: string) {
  if (level === 'High' || level === 'Confirms') return 'text-green-700 font-semibold'
  if (level === 'Medium') return 'text-amber-600 font-semibold'
  if (level === 'Low' || level === 'Contradicts') return 'text-red-600 font-semibold'
  return 'text-gray-400'
}

const CLASS_STYLE: Record<string, string> = {
  tissue_specific:  'bg-[#E8F5E9] text-[#2D6A4F]',
  ubiquitous:       'bg-[#FFF8E1] text-[#F57F17]',
  loss_of_function: 'bg-[#FCE4EC] text-[#C62828]',
}

function Section({ label, count, children }: { label: string; count?: number; children: React.ReactNode }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="border-t border-[#F5F5F7] mt-3 pt-3">
      <button
        onClick={() => setOpen(o => !o)}
        className="flex items-center gap-1.5 text-xs text-[#6E6E73] hover:text-[#1D1D1F] transition-colors"
      >
        <span>{open ? '▾' : '▸'}</span>
        {label}
        {count !== undefined && <span className="text-[#D2D2D7] font-mono">({count})</span>}
      </button>
      {open && <div className="mt-2">{children}</div>}
    </div>
  )
}

export default function ResultCard({ result, gene, diseaseFilter, excludeGenes, onCellLineClick }: Props) {
  const [aiData, setAiData] = useState<any>(null)
  const [aiLoading, setAiLoading] = useState(false)
  const [aiError, setAiError] = useState(false)

  const isTop    = result.rank === 1
  const scorePct = Math.round((result.final_score ?? 0) * 100)

  const sources = [
    (result.hpa_score ?? 0) > 0 && 'HPA RNA',
    (result.depmap_score ?? 0) > 0 && 'DepMap',
    (result.geo_confirmation ?? 0) !== 0 && 'GEO',
    (result.protein_score ?? 0) > 0 && 'Proteomics',
  ].filter(Boolean) as string[]

  const handleAI = async () => {
    setAiLoading(true)
    setAiError(false)
    try {
      const data = await api.recommendAgentic({
        gene,
        disease_filter: diseaseFilter,
        exclude_genes: excludeGenes,
        target_cellosaurus_id: result.cellosaurus_id,
        top_n: 1,
      })
      setAiData(data)
    } catch {
      setAiError(true)
    } finally {
      setAiLoading(false)
    }
  }

  const classStyle = CLASS_STYLE[result.gene_class] ?? 'bg-[#F5F5F7] text-[#6E6E73]'

  return (
    <div
      className={`bg-white border border-[#D2D2D7] rounded-xl p-5 transition-shadow duration-200 hover:shadow-md ${
        isTop ? 'border-l-4 border-l-[#1D1D1F]' : ''
      }`}
      style={{ boxShadow: '0 1px 8px rgba(0,0,0,0.06)' }}
    >
      {/* Top row */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-start gap-3 flex-1 min-w-0">
          <span className="text-[#6E6E73] font-mono text-xl font-bold leading-none flex-shrink-0 mt-0.5">
            {String(result.rank).padStart(2, '0')}
          </span>
          <div className="min-w-0">
            <button
              onClick={() => onCellLineClick(result.cellosaurus_id)}
              className="text-[#1D1D1F] font-bold text-base leading-tight hover:text-[#6E6E73] transition-colors text-left truncate max-w-xs block"
            >
              {result.official_name}
            </button>
            <div className="text-[#6E6E73] font-mono text-xs mt-0.5">{result.cellosaurus_id}</div>
          </div>
        </div>
        <div className="text-right ml-4 flex-shrink-0">
          <div className="text-[#1D1D1F] font-mono text-2xl font-bold leading-tight">{scorePct}%</div>
          <div className="text-[#6E6E73] text-xs">Fit Score</div>
        </div>
      </div>

      {/* Badges */}
      <div className="flex flex-wrap items-center gap-1.5 mb-3">
        {result.gene_class && (
          <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${classStyle}`}>
            {(result.gene_class as string).replace(/_/g, ' ')}
          </span>
        )}
        {result.gene_role && (
          <span className="text-xs text-[#6E6E73] italic">
            {result.gene_role}
          </span>
        )}
        {[
          { label: 'RNA',     val: result.rna_score },
          { label: 'PROTEIN', val: result.protein_score },
          { label: 'QUALITY', val: result.quality_score },
          { label: 'CONTEXT', val: result.context_score },
          { label: 'PATHWAY', val: result.pathway_activity_score },
        ].map(({ label, val }) => (
          <span key={label} className="bg-[#F5F5F7] text-[#6E6E73] text-xs px-2 py-0.5 rounded font-mono">
            {(val ?? 0).toFixed(2)} {label}
          </span>
        ))}
      </div>

      {/* Source chips */}
      {sources.length > 0 && (
        <div className="flex flex-wrap gap-1.5 mb-3">
          {sources.map(s => (
            <span key={s} className="text-xs border border-[#D2D2D7] text-[#6E6E73] px-2 py-0.5 rounded-full">
              {s}
            </span>
          ))}
        </div>
      )}

      {/* Evidence line */}
      <div className="text-[#6E6E73] text-xs mb-2">
        {[result.disease, result.lineage, result.n_sources ? `${result.n_sources} RNA src` : null]
          .filter(Boolean)
          .join(' · ')}
      </div>

      {/* Doubling time */}
      {result.growth_properties?.doubling_time && (
        <div className="text-xs text-[#6E6E73] mt-1">
          Doubling time: {result.growth_properties.doubling_time.min}
          –{result.growth_properties.doubling_time.max}{' '}
          {result.growth_properties.doubling_time.unit}s
        </div>
      )}

      {/* Exclusion warnings */}
      {result.exclusion_warnings?.length > 0 && (
        <div className="mb-2 space-y-0.5">
          {result.exclusion_warnings.map((w: any, i: number) => (
            <div key={i} className="text-[#F57F17] text-xs">⚠ {w.message}</div>
          ))}
        </div>
      )}

      {/* Alternatives */}
      {result.alternatives?.length > 0 && (
        <Section label="Similar alternatives" count={result.alternatives.length}>
          <div className="space-y-2">
            {result.alternatives.map((alt: any, i: number) => (
              <div key={i} className="bg-[#F5F5F7] rounded-lg p-3 text-xs">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[#1D1D1F] font-medium">{alt.official_name}</span>
                  <span className="text-[#1D1D1F] font-mono">{Math.round((alt.similarity_score ?? 0) * 100)}%</span>
                </div>
                <div className="text-[#6E6E73] mb-1 font-mono">{alt.cellosaurus_id}</div>
                <div className="text-[#6E6E73] italic leading-relaxed">{alt.similarity_reason}</div>
                {alt.note && <div className="text-[#6E6E73] mt-1">{alt.note}</div>}
                <a
                  href={alt.cellosaurus_url ?? `https://www.cellosaurus.org/${alt.cellosaurus_id}`}
                  target="_blank" rel="noopener noreferrer"
                  className="text-[#457B9D] hover:underline mt-1 inline-block"
                >
                  Cellosaurus ↗
                </a>
              </div>
            ))}
          </div>
        </Section>
      )}

      {/* Score breakdown — all detailed evidence consolidated here (bars +
          quality/context reasoning + per-source percentiles + rank
          comparison) so the collapsed card stays scannable across 10+
          results, and depth is only a click away. */}
      <Section label="Score breakdown">
        <div className="space-y-2">
          <ScoreBar label="RNA"     value={result.rna_score     ?? 0} />
          <ScoreBar label="Protein" value={result.protein_score ?? 0} />
          <div>
            <ScoreBar label="Quality" value={result.quality_score ?? 0} />
            {result.quality_explanation && (
              <div className="text-xs text-gray-400 pl-3 mt-0.5">
                └─ {result.quality_explanation}
              </div>
            )}
          </div>
          <div>
            <ScoreBar label="Context" value={result.context_score ?? 0} />
            {result.context_explanation && (
              <div className="text-xs text-gray-400 pl-3 mt-0.5">
                └─ {result.context_explanation}
              </div>
            )}
          </div>
          <div>
            <ScoreBar label="Pathway" value={result.pathway_activity_score ?? 0} />
            {result.pathway_genes_total > 0 && (
              <div className="text-xs text-gray-400 pl-3 mt-0.5">
                └─ {result.pathway_genes_expressed}/{result.pathway_genes_total} pathway-neighbor genes also expressed here
              </div>
            )}
          </div>
        </div>

        {/* Per-source evidence — label + exact percentile, so cross-source
            agreement/disagreement is visible without losing precision to a
            coarse Low/Medium/High bucket. */}
        <div className="grid grid-cols-4 gap-3 mt-4 pt-3 border-t border-[#F5F5F7]">
          {[
            { key: 'hpa_evidence', name: 'HPA RNA' },
            { key: 'depmap_evidence', name: 'DepMap' },
            { key: 'geo_evidence', name: 'GEO' },
            { key: 'protein_evidence', name: 'Proteomics' },
          ].map(({ key, name }) => {
            const ev = result[key]
            if (!ev) return null
            return (
              <div key={key} className="text-xs">
                <div className="text-[#6E6E73]">{name}</div>
                <div className={levelColor(ev.label)}>
                  {ev.label}
                </div>
                {ev.percentile && (
                  <div className="text-[10px] text-gray-400">
                    {ev.percentile}
                  </div>
                )}
              </div>
            )
          })}
        </div>

        {result.vs_next_rank && (
          <div className="mt-3 pt-3 border-t border-[#F5F5F7] text-xs text-[#6E6E73] italic">
            {result.vs_next_rank}
          </div>
        )}
      </Section>

      {/* AI Justification */}
      <div className="border-t border-[#F5F5F7] mt-3 pt-3">
        {!aiData ? (
          <button
            onClick={handleAI}
            disabled={aiLoading}
            className="flex items-center gap-2 text-xs border border-[#D2D2D7] text-[#6E6E73] px-3 py-1.5 rounded-lg hover:border-[#1D1D1F] hover:text-[#1D1D1F] transition-colors disabled:opacity-50"
          >
            {aiLoading ? (
              <>
                <span className="w-3 h-3 border border-[#D2D2D7] border-t-[#1D1D1F] rounded-full animate-spin" />
                Generating AI justification…
              </>
            ) : (
              'Get AI Justification →'
            )}
          </button>
        ) : (
          <div className="space-y-3">
            {aiData.results?.[0]?.justification && (() => {
              const raw = typeof aiData.results[0].justification === 'string'
                ? aiData.results[0].justification
                : JSON.stringify(aiData.results[0].justification, null, 2)
              const parsed = parseJustification(raw)
              const mainKeys = SECTION_LABELS.filter(k => parsed[k] && !MONO_SECTIONS.has(k))
              const monoKeys = SECTION_LABELS.filter(k => parsed[k] && MONO_SECTIONS.has(k))
              const hasAny   = mainKeys.length > 0 || monoKeys.length > 0

              return hasAny ? (
                <div className="bg-[#F5F5F7] rounded-lg p-3 space-y-2.5">
                  {mainKeys.map(k => (
                    <div key={k}>
                      <div className="text-[10px] font-semibold tracking-wider text-[#6E6E73] uppercase mb-0.5">{k}</div>
                      <p className="text-xs text-[#1D1D1F] leading-relaxed">{parsed[k]}</p>
                    </div>
                  ))}
                  {monoKeys.map(k => (
                    <div key={k} className="border-t border-[#D2D2D7] pt-2">
                      <div className="text-[10px] font-semibold tracking-wider text-[#6E6E73] uppercase mb-0.5">{k}</div>
                      <pre className="text-[10px] text-[#6E6E73] font-mono whitespace-pre-wrap leading-relaxed">{parsed[k]}</pre>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="bg-[#F5F5F7] rounded-lg p-3 text-xs text-[#1D1D1F] leading-relaxed">{raw}</div>
              )
            })()}
            {aiData.dataset_citations && (
              <div className="space-y-0.5">
                {aiData.dataset_citations.map((c: any) => (
                  <div key={c.key} className="text-[#6E6E73] text-xs font-mono">[{c.key}] {c.name}</div>
                ))}
              </div>
            )}
          </div>
        )}
        {aiError && (
          <div className="text-[#C62828] text-xs mt-1">
            AI justification failed — check that the Ollama server is running.
          </div>
        )}
      </div>
    </div>
  )
}
