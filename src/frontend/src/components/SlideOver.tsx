import { useEffect, useState } from 'react'
import { api } from '../api/client'
import ScoreBar from './ScoreBar'
import LoadingSpinner from './LoadingSpinner'

interface Props {
  cellosaurus_id: string
  onClose: () => void
}

const COVERAGE_LABELS: Record<string, string> = {
  has_hpa_expr:     'HPA RNA',
  has_depmap_expr:  'DepMap RNA',
  has_geo_expr:     'GEO',
  has_proteomics:   'Proteomics',
  has_metabolomics: 'Metabolomics',
  has_mirna:        'miRNA',
  has_mutations:    'Mutations',
  has_fusions:      'Fusions',
}

export default function SlideOver({ cellosaurus_id, onClose }: Props) {
  const [data, setData] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(false)

  useEffect(() => {
    setLoading(true); setError(false); setData(null)
    api.cellLine(cellosaurus_id)
      .then(d => setData(d))
      .catch(() => setError(true))
      .finally(() => setLoading(false))
  }, [cellosaurus_id])

  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose])

  return (
    <>
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/25 z-40 backdrop-blur-sm" onClick={onClose} />

      {/* Panel */}
      <div
        className="fixed right-0 top-0 h-full w-96 bg-white z-50 overflow-y-auto"
        style={{ borderLeft: '1px solid #D2D2D7', boxShadow: '-8px 0 32px rgba(0,0,0,0.08)' }}
      >
        <div className="p-6">
          {/* Header */}
          <div className="flex items-start justify-between mb-6">
            <div className="flex-1 min-w-0">
              {loading ? (
                <LoadingSpinner text="Fetching cell line…" />
              ) : error ? (
                <p className="text-[#C62828] text-sm">Failed to load cell line data.</p>
              ) : (
                <>
                  <h2 className="text-[#1D1D1F] font-bold text-xl leading-tight truncate pr-2">
                    {data?.official_name}
                  </h2>
                  <span className="text-[#6E6E73] font-mono text-sm">{cellosaurus_id}</span>
                </>
              )}
            </div>
            <button
              onClick={onClose}
              className="text-[#6E6E73] hover:text-[#1D1D1F] text-lg leading-none ml-2 mt-0.5 flex-shrink-0 transition-colors"
              aria-label="Close"
            >
              ✕
            </button>
          </div>

          {data && (
            <>
              {/* Context */}
              <div className="mb-6">
                <div className="text-[#6E6E73] text-xs uppercase tracking-widest mb-2">Context</div>
                <div className="text-[#1D1D1F] text-sm">{data.disease || '—'}</div>
                <div className="text-[#6E6E73] text-xs mt-0.5">{data.lineage || '—'}</div>
              </div>

              {/* Coverage */}
              <div className="mb-6">
                <div className="text-[#6E6E73] text-xs uppercase tracking-widest mb-3">Data Coverage</div>
                <div className="grid grid-cols-2 gap-y-2 gap-x-4">
                  {Object.entries(COVERAGE_LABELS).map(([key, label]) => {
                    const has = Boolean(data.data_coverage?.[key])
                    return (
                      <div key={key} className="flex items-center gap-2">
                        <span className={`text-sm font-bold ${has ? 'text-[#2D6A4F]' : 'text-[#D2D2D7]'}`}>
                          {has ? '✓' : '✗'}
                        </span>
                        <span className="text-xs text-[#6E6E73]">{label}</span>
                      </div>
                    )
                  })}
                </div>
              </div>

              {/* Genomic features */}
              {data.genomic_features && Object.keys(data.genomic_features).length > 0 && (
                <div className="mb-6">
                  <div className="text-[#6E6E73] text-xs uppercase tracking-widest mb-3">Genomic Features</div>
                  {Object.entries(data.genomic_features as Record<string, number | null>).map(([key, val]) =>
                    val != null ? <ScoreBar key={key} label={key} value={val} /> : null
                  )}
                </div>
              )}

              <div className="mb-5 text-xs text-[#6E6E73]">
                Evidence entries: <span className="text-[#1D1D1F] font-mono">{data.evidence_count ?? '—'}</span>
              </div>

              <a
                href={data.cellosaurus_url}
                target="_blank" rel="noopener noreferrer"
                className="inline-block border border-[#D2D2D7] text-[#1D1D1F] text-sm px-4 py-2 rounded-lg hover:border-[#1D1D1F] transition-colors"
              >
                View on Cellosaurus ↗
              </a>
            </>
          )}
        </div>
      </div>
    </>
  )
}
