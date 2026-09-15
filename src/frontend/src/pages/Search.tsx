import { useEffect, useRef, useState } from 'react'
import { api } from '../api/client'
import ResultCard from '../components/ResultCard'
import SlideOver from '../components/SlideOver'

const LOADING_LINES = [
  '> Resolving gene symbol…',
  '> Computing RNA expression scores…',
  '> Applying protein expression weights…',
  '> Running GEO cross-validation…',
  '> Ranking 2,076 cell lines…',
  '> Computing similarity alternatives…',
  '> Analysis complete.',
]

export default function Search() {
  const [gene, setGene] = useState('')
  const [geneInfo, setGeneInfo] = useState<any>(null)
  const [geneLoading, setGeneLoading] = useState(false)
  const [diseaseFilter, setDiseaseFilter] = useState('')
  const [lineageFilter, setLineageFilter] = useState('')
  const [excludeGenes, setExcludeGenes] = useState('')
  const [topN, setTopN] = useState(10)
  const [allResults, setAllResults] = useState<any>(null)
  const [pathwayResults, setPathwayResults] = useState<any[] | null>(null)
  const [loading, setLoading] = useState(false)
  const [loadLine, setLoadLine] = useState(0)
  const [selectedCVCL, setSelectedCVCL] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [exportOpen, setExportOpen] = useState(false)
  const exportMenuRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!exportOpen) return
    const handler = (e: MouseEvent) => {
      if (exportMenuRef.current && !exportMenuRef.current.contains(e.target as Node)) {
        setExportOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [exportOpen])

  useEffect(() => {
    if (!gene.trim()) { setGeneInfo(null); return }
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(async () => {
      setGeneLoading(true)
      try {
        const r = await api.searchGene(gene.trim().toUpperCase())
        setGeneInfo(r)
      } catch {
        setGeneInfo(null)
      } finally {
        setGeneLoading(false)
      }
    }, 300)
  }, [gene])

  useEffect(() => {
    if (!loading) { setLoadLine(0); return }
    const id = setInterval(() => setLoadLine(l => Math.min(l + 1, LOADING_LINES.length - 1)), 650)
    return () => clearInterval(id)
  }, [loading])

  const handleSearch = async () => {
    const g = gene.trim().toUpperCase()
    if (!g) return

    const excludeList = excludeGenes
      ? excludeGenes.split(',').map(s => s.trim().toUpperCase()).filter(Boolean)
      : []
    if (excludeList.includes(g)) {
      setError(`Cannot search for ${g} and exclude it at the same time.`)
      return
    }

    setLoading(true); setAllResults(null); setPathwayResults(null); setError(null)
    try {
      const diseaseFilterVal = diseaseFilter.trim() || undefined
      // Pathway-connected recommendations are fetched alongside the main
      // search, not on a separate user action. A failure here (e.g. the
      // gene isn't yet ingested into the graph) shouldn't break the main
      // results, so it's caught independently rather than via the outer catch.
      const [r, pw] = await Promise.all([
        api.recommendClassical({
          gene: g,
          disease_filter: diseaseFilterVal,
          lineage_filter: lineageFilter.trim() || undefined,
          exclude_genes: excludeList.length ? excludeList : undefined,
          top_n: 50,
        }),
        api.cellLinesViaPathway(g, diseaseFilterVal, 5).catch(() => null),
      ])
      if (r.detail) throw new Error(r.detail)
      setAllResults(r)
      setPathwayResults(pw?.results ?? null)
    } catch (e: any) {
      setError(e?.message ?? 'Request failed. Is the API running on port 8001?')
    } finally {
      setLoading(false)
    }
  }

  const exportJSON = () => {
    if (!allResults) return
    const blob = new Blob([JSON.stringify(allResults, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `cellselector_${gene}_${new Date().toISOString().split('T')[0]}.json`
    a.click(); URL.revokeObjectURL(url)
  }

  const exportCSV = () => {
    if (!allResults?.results) return
    const headers = ['rank','cellosaurus_id','official_name','final_score','rna_score',
      'protein_score','quality_score','context_score','n_sources','gene_class','disease','lineage']
    const rows = (allResults.results as any[]).map((r: any) => headers.map(h => r[h] ?? '').join(','))
    const csv = [headers.join(','), ...rows].join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `cellselector_${gene}_${new Date().toISOString().split('T')[0]}.csv`
    a.click(); URL.revokeObjectURL(url)
  }

  const exportPDF = async () => {
    if (!allResults?.session_id) {
      alert('No session available — run a search first')
      return
    }
    const response = await fetch(`/recommend/export/pdf?session_id=${allResults.session_id}`)
    if (!response.ok) {
      console.error('PDF export failed', response.status)
      return
    }
    const blob = await response.blob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `cellselector_${gene}_${new Date().toISOString().split('T')[0]}.pdf`
    a.click()
    URL.revokeObjectURL(url)
  }

  const displayedResults = allResults
    ? { ...allResults, results: (allResults.results as any[]).slice(0, topN) }
    : null

  // Actual weights the API used for this query (learned or fixed) — the
  // scoring panel reads real numbers from here rather than hardcoding them,
  // since learned weights (the default) don't match any fixed percentage.
  const w = allResults?.weights_used

  const geneFound = geneInfo?.found === true
  const sourcesFound = geneFound
    ? Object.entries(geneInfo.sources as Record<string, boolean>)
        .filter(([, v]) => v).map(([k]) => k).join(' · ')
    : ''

  const inputCls = `w-full bg-white border border-[#D2D2D7] text-[#1D1D1F] px-4 py-2.5 rounded-xl text-sm focus:outline-none focus:border-[#1D1D1F] transition-colors placeholder-[#D2D2D7]`

  return (
    <div className="min-h-screen bg-white pt-24">
      {/* Search panel */}
      <div className="bg-white border-b border-[#D2D2D7]">
        <div className="max-w-3xl mx-auto px-6 pb-8">
          <p className="text-[#6E6E73] text-xs tracking-[0.2em] uppercase mb-2">Cell Line Recommender</p>
          <h1 className="text-[#1D1D1F] text-3xl font-bold mb-8">Search Tool</h1>

          {/* Gene input */}
          <div className="mb-5">
            <label className="text-[#6E6E73] text-xs uppercase tracking-widest block mb-2">Gene Name</label>
            <div className="relative">
              <input
                type="text"
                value={gene}
                onChange={e => setGene(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleSearch()}
                placeholder="e.g. EGFR, BRCA1, KIT"
                autoFocus
                className="w-full bg-white border border-[#D2D2D7] text-[#1D1D1F] font-mono text-lg px-4 py-3 rounded-xl focus:outline-none focus:border-[#1D1D1F] transition-colors placeholder-[#D2D2D7]"
                style={{ boxShadow: gene ? '0 0 0 3px rgba(29,29,31,0.06)' : undefined }}
              />
              {geneLoading && (
                <div className="absolute right-3.5 top-4">
                  <div className="w-4 h-4 border border-[#D2D2D7] border-t-[#1D1D1F] rounded-full animate-spin" />
                </div>
              )}
            </div>
            {geneInfo && !geneLoading && (
              <div className={`mt-2 text-xs font-mono ${geneFound ? 'text-[#2D6A4F]' : 'text-[#C62828]'}`}>
                {geneFound
                  ? `✓ ${gene.toUpperCase()} — ${sourcesFound} — ${geneInfo.total_cell_lines_with_data?.toLocaleString()} cell lines`
                  : `✗ ${gene.toUpperCase()} not found in any omics source`}
              </div>
            )}
          </div>

          {/* Filters */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-5">
            {[
              { label: 'Disease Filter',           value: diseaseFilter, set: setDiseaseFilter, ph: 'e.g. lung, breast' },
              { label: 'Tissue Type',               value: lineageFilter, set: setLineageFilter, ph: 'e.g. lung, breast, epithelial' },
              { label: 'Exclude Genes (comma sep)', value: excludeGenes,  set: setExcludeGenes,  ph: 'e.g. KRAS, NRAS' },
            ].map(({ label, value, set, ph }) => (
              <div key={label}>
                <label className="text-[#6E6E73] text-xs uppercase tracking-widest block mb-2">{label}</label>
                <input
                  type="text" value={value}
                  onChange={e => set(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && handleSearch()}
                  placeholder={ph}
                  className={inputCls}
                />
              </div>
            ))}
          </div>

          <div className="flex items-center gap-4 flex-wrap">
            <div className="flex items-center gap-2">
              <label className="text-[#6E6E73] text-xs uppercase tracking-widest">Top</label>
              <select
                value={topN}
                onChange={e => setTopN(Number(e.target.value))}
                className="bg-white border border-[#D2D2D7] text-[#1D1D1F] px-3 py-2 rounded-xl text-sm focus:outline-none"
              >
                {[5, 10, 20, 50].map(n => <option key={n} value={n}>{n}</option>)}
              </select>
            </div>
            <button
              onClick={handleSearch}
              disabled={!gene.trim() || loading}
              className="flex-1 bg-[#1D1D1F] text-white font-semibold py-3 rounded-xl hover:bg-[#333333] transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {loading ? 'Running…' : 'Run Analysis →'}
            </button>
          </div>
        </div>
      </div>

      <div className="max-w-3xl mx-auto px-6 py-8">
        {/* Loading terminal */}
        {loading && (
          <div className="bg-[#F5F5F7] border border-[#D2D2D7] rounded-xl p-4 font-mono text-xs mb-8">
            {LOADING_LINES.slice(0, loadLine + 1).map((line, i) => (
              <div key={i} className="text-[#6E6E73]">
                {line}
                {i === loadLine && i < LOADING_LINES.length - 1 && (
                  <span className="animate-pulse ml-0.5 text-[#1D1D1F]">█</span>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="bg-[#FCE4EC] border border-[#F8BBD9] rounded-xl p-4 text-[#C62828] text-sm mb-6">
            {error}
          </div>
        )}

        {/* Results */}
        {allResults && !loading && (
          <>
            {/* Scoring transparency panel */}
            <div className="bg-gray-50 rounded-lg p-4 mb-4 text-xs text-[#6E6E73] leading-relaxed">
              <div className="font-semibold text-black mb-2">
                How Fit Score is computed
              </div>
              <p>
                Each cell line is scored across five weighted
                dimensions: <strong>RNA Expression</strong>{' '}
                ({w?.rna ? (w.rna * 100).toFixed(0) : '?'}%)
                measures transcript abundance across HPA and
                DepMap; <strong>Protein</strong>{' '}
                ({w?.protein ? (w.protein * 100).toFixed(0) : '?'}%) measures
                protein abundance from CCLE proteomics;{' '}
                <strong>Data Quality</strong>{' '}
                ({w?.quality ? (w.quality * 100).toFixed(0) : '?'}%) reflects
                cross-source agreement and data completeness;{' '}
                <strong>Context</strong>{' '}
                ({w?.context ? (w.context * 100).toFixed(0) : '?'}%) rewards disease
                and tissue match to your search filter;{' '}
                <strong>Pathway Activity</strong>{' '}
                ({w?.pathway ? (w.pathway * 100).toFixed(0) : '10'}%) measures
                how many genes sharing a KEGG pathway with your
                target are also expressed in this cell line. GEO
                expression acts as a confirmatory bonus
                (up to +10%). Weights are optimised by maximising
                Mean Reciprocal Rank against 25 validated
                gene-cell-line associations from the literature.
              </p>
              <p className="mt-2">
                Fit Score combines five evidence dimensions with
                weights that are automatically optimized per gene
                class using Mean Reciprocal Rank against 25
                validated gene-cell-line associations. The system
                discovered that pathway activity improves ranking
                accuracy for hormone receptors and broadly-expressed
                genes, but can introduce noise for receptor tyrosine
                kinases where direct expression is already the
                decisive signal — so pathway weight is tuned
                independently per gene class. The percentages above
                reflect the weights actually applied to{' '}
                {allResults.query?.gene ?? gene}, a{' '}
                {displayedResults?.results?.[0]?.gene_class?.replace(/_/g, ' ') ?? 'classified'}{' '}
                gene.
              </p>
            </div>

            <div className="flex items-center justify-between mb-6">
              <div>
                <div className="text-[#1D1D1F] font-bold text-lg">
                  Showing {displayedResults?.results.length} of {allResults.results?.length} loaded for{' '}
                  <span className="font-mono">{allResults.query?.gene}</span>
                  {allResults.query?.disease_filter && (
                    <span className="text-[#6E6E73] text-sm font-normal ml-2">
                      in {allResults.query.disease_filter}
                    </span>
                  )}
                </div>
                <div className="text-[#6E6E73] text-xs mt-0.5 font-mono">
                  {allResults.metadata?.total_candidates?.toLocaleString()} total candidates scored
                  {allResults.metadata?.execution_time_ms && ` · ${allResults.metadata.execution_time_ms}ms`}
                </div>
              </div>
              <div className="relative" ref={exportMenuRef}>
                <button
                  onClick={() => setExportOpen(o => !o)}
                  className="text-xs border border-[#D2D2D7] text-[#6E6E73] hover:text-[#1D1D1F] hover:border-[#1D1D1F] px-3 py-1.5 rounded-lg transition-colors"
                >
                  Export ▾
                </button>
                {exportOpen && (
                  <div className="absolute right-0 mt-2 w-32 bg-white border border-[#D2D2D7] rounded-lg shadow-lg z-10 overflow-hidden">
                    {([['JSON', exportJSON], ['CSV', exportCSV], ['PDF', exportPDF]] as [string, () => void][]).map(([label, fn]) => (
                      <button
                        key={label}
                        onClick={() => { fn(); setExportOpen(false) }}
                        className="w-full text-left px-4 py-2 hover:bg-[#F5F5F7] text-sm text-[#1D1D1F] transition-colors"
                      >
                        {label}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>

            <div className="space-y-4">
              {(displayedResults!.results as any[]).map((r: any) => (
                <ResultCard
                  key={r.cellosaurus_id}
                  result={r}
                  gene={allResults.query?.gene ?? gene}
                  diseaseFilter={allResults.query?.disease_filter}
                  excludeGenes={
                    excludeGenes
                      ? excludeGenes.split(',').map(s => s.trim().toUpperCase()).filter(Boolean)
                      : undefined
                  }
                  onCellLineClick={setSelectedCVCL}
                />
              ))}
            </div>

            {/* Pathway-Connected Recommendations */}
            <div className="mt-8 pt-8 border-t border-gray-200">
              <h2 className="text-lg font-semibold mb-1">
                Pathway-Connected Recommendations
              </h2>
              <p className="text-xs text-[#6E6E73] mb-4">
                Cell lines strong for genes that share a
                biological pathway with {allResults.query?.gene ?? gene}. These expand
                your experimental options beyond direct{' '}
                {allResults.query?.gene ?? gene} expression.
              </p>
              {pathwayResults === null ? (
                <div className="text-xs text-[#6E6E73]">
                  No pathway-connected data available for this gene yet — the knowledge
                  graph currently only covers a curated set of validation genes.
                </div>
              ) : pathwayResults.length === 0 ? (
                <div className="text-xs text-[#6E6E73]">
                  No pathway-connected cell lines found for this gene.
                </div>
              ) : (
                pathwayResults.map((r: any) => {
                  const top = r.connecting_genes?.[0]
                  const isDirect = top?.is_target
                  return (
                    <div
                      key={r.cellosaurus_id}
                      className="border rounded-lg p-3 mb-2 cursor-pointer hover:border-[#1D1D1F] transition-colors"
                      onClick={() => setSelectedCVCL(r.cellosaurus_id)}
                    >
                      <div className="flex justify-between">
                        <div>
                          <span className="font-medium">
                            {r.official_name ?? r.cellosaurus_id}
                          </span>
                          {top && (
                            <span
                              className={`ml-2 text-xs px-2 py-0.5 rounded ${
                                isDirect ? 'bg-amber-50 text-amber-700' : 'bg-blue-50 text-blue-700'
                              }`}
                            >
                              {isDirect ? `direct: ${top.gene}` : `via ${top.gene}`}
                            </span>
                          )}
                        </div>
                        <span className="text-sm">
                          score {(r.max_score ?? 0).toFixed(2)}
                        </span>
                      </div>
                    </div>
                  )
                })
              )}
            </div>
          </>
        )}

        {/* Empty state */}
        {!allResults && !loading && !error && (
          <div className="text-center py-24">
            <div className="text-[#D2D2D7] text-6xl mb-5">⬡</div>
            <div className="text-[#6E6E73] text-sm">
              Enter a gene symbol above and press{' '}
              <span className="text-[#1D1D1F] font-semibold">Run Analysis</span>
            </div>
          </div>
        )}
      </div>

      {selectedCVCL && (
        <SlideOver cellosaurus_id={selectedCVCL} onClose={() => setSelectedCVCL(null)} />
      )}
    </div>
  )
}
