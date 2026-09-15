# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import re, json, os
from functools import reduce

st.set_page_config(page_title="CellLineFinder", page_icon="🧬", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&family=IBM+Plex+Mono:wght@500&display=swap');
#MainMenu,footer,header{visibility:hidden;}
.stApp{background:#F1F5F9;}
.block-container{padding-top:2rem;max-width:900px;}
*{font-family:-apple-system,BlinkMacSystemFont,'SF Pro Display','SF Pro Text','Helvetica Neue',Helvetica,Arial,sans-serif;}
.head{display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #E2E8F0;padding-bottom:1.3rem;margin-bottom:1.3rem;}
.bt h1{font-size:1.4rem;font-weight:700;color:#0F172A;letter-spacing:-0.3px;margin:0;}
.bt p{font-size:0.82rem;color:#64748B;font-weight:400;margin:1px 0 0 0;}
.headstats{display:flex;gap:1.8rem;}
.hs{text-align:right;}
.hs .n{font-size:1.1rem;font-weight:700;color:#0F172A;font-family:'IBM Plex Mono';}
.hs .l{font-size:0.66rem;color:#94A3B8;text-transform:uppercase;letter-spacing:0.6px;}
.stTextInput input{background:#fff !important;border:1px solid #CBD5E1 !important;border-radius:9px !important;padding:0.85rem 1.1rem !important;font-size:1rem !important;color:#0F172A !important;font-family:'IBM Plex Mono' !important;}
.stTextInput input:focus{border-color:#0D9488 !important;box-shadow:0 0 0 3px rgba(13,148,136,0.12) !important;}
.stTextInput input::placeholder{color:#94A3B8 !important;}
.stButton button{border-radius:9px !important;font-weight:600 !important;padding:0.7rem 0 !important;border:1px solid #CBD5E1 !important;background:#fff !important;color:#334155 !important;}
.stButton button:hover{border-color:#0D9488 !important;color:#0D9488 !important;}
.stButton button[kind="primary"]{background:#0F172A !important;color:#fff !important;border-color:#0F172A !important;}
.parsed{font-size:0.82rem;color:#64748B;margin:0.6rem 0 0 0;font-family:'IBM Plex Mono';}
.parsed b{color:#0D9488;}
.resbar{font-size:0.85rem;color:#64748B;margin:1.3rem 0 0.9rem 0;font-weight:500;}
.resbar b{color:#0F172A;}
.summary{background:#ECFEFF;border:1px solid #A5F3FC;border-radius:12px;padding:1rem 1.2rem;margin:0.6rem 0 1.2rem 0;font-size:0.9rem;color:#155E63;line-height:1.6;}
.summary .st{font-weight:700;color:#0E7490;text-transform:uppercase;font-size:0.72rem;letter-spacing:0.6px;margin-bottom:0.4rem;}
.card{background:#fff;border:1px solid #E2E8F0;border-radius:12px;padding:1.2rem 1.4rem;margin-bottom:0.8rem;}
.card.top{border-left:3px solid #0D9488;}
.crow{display:flex;align-items:center;gap:1rem;}
.rank{font-family:'IBM Plex Mono';font-weight:600;font-size:0.9rem;color:#94A3B8;min-width:30px;}
.rank.hi{color:#0D9488;}
.name{font-size:1.2rem;font-weight:700;color:#0F172A;}
.cid{font-family:'IBM Plex Mono';font-size:0.75rem;color:#94A3B8;}
.score{margin-left:auto;text-align:right;}
.score .n{font-size:1.5rem;font-weight:800;color:#0F172A;font-family:'IBM Plex Mono';line-height:1;}
.score .l{font-size:0.64rem;color:#94A3B8;text-transform:uppercase;letter-spacing:0.6px;margin-top:3px;}
.metrics{display:flex;margin-top:1rem;border:1px solid #EEF2F6;border-radius:8px;overflow:hidden;}
.m{flex:1;padding:0.6rem 0.9rem;border-right:1px solid #EEF2F6;background:#FAFBFC;}
.m:last-child{border-right:none;}
.m .v{font-family:'IBM Plex Mono';font-weight:600;font-size:0.98rem;color:#0F172A;}
.m .k{font-size:0.64rem;color:#94A3B8;text-transform:uppercase;letter-spacing:0.5px;margin-top:1px;}
.m .v.teal{color:#0D9488;}
.contrib{display:flex;gap:8px;margin-top:0.9rem;flex-wrap:wrap;}
.cbox{border:1px solid #E2E8F0;border-radius:8px;padding:5px 10px;font-size:0.72rem;min-width:82px;}
.cbox .cn{color:#94A3B8;font-family:'IBM Plex Mono';font-size:0.64rem;text-transform:uppercase;letter-spacing:0.4px;}
.cbox .cl{font-weight:700;margin-top:2px;}
.cl.high{color:#0F766E;} .cl.med{color:#B9770E;} .cl.low{color:#94A3B8;} .cl.none{color:#CBD5E1;}
.ev{margin-top:0.85rem;padding-top:0.85rem;border-top:1px solid #EEF2F6;font-size:0.85rem;color:#64748B;line-height:1.5;}
.ev b{color:#334155;font-weight:600;}
.ev .d{color:#0D9488;font-weight:600;}
.chip{display:inline-block;font-size:0.7rem;font-weight:600;padding:2px 9px;border-radius:20px;margin-left:6px;background:#E6F5F3;color:#0F766E;}
.excl-warn{display:inline-block;font-size:0.7rem;font-weight:600;padding:2px 9px;border-radius:20px;margin-left:6px;background:#FEF9C3;color:#854D0E;}
.excl-note{margin-top:0.5rem;padding:0.45rem 0.8rem;background:#FFFBEB;border:1px solid #FDE68A;border-radius:7px;font-size:0.78rem;color:#92400E;}
.stExpander{border:1px solid #E2E8F0 !important;border-radius:9px !important;background:#F0FDFA !important;margin-top:0.7rem !important;}
.stExpander summary{font-weight:600 !important;color:#0F766E !important;font-size:0.9rem !important;}
.alt-row{display:flex;align-items:center;gap:0.7rem;padding:0.45rem 0;border-bottom:1px solid #EEF2F6;}
.alt-row:last-child{border-bottom:none;}
.alt-sim{font-family:'IBM Plex Mono';font-size:0.88rem;font-weight:700;color:#0D9488;min-width:38px;}
.alt-name{font-weight:600;font-size:0.88rem;color:#0F172A;}
.alt-cid{font-family:'IBM Plex Mono';font-size:0.7rem;color:#94A3B8;margin-left:0.3rem;}
.alt-chips{display:flex;gap:4px;flex-wrap:wrap;margin-left:auto;}
.alt-chip{font-size:0.64rem;font-weight:600;padding:1px 7px;border-radius:20px;background:#E6F5F3;color:#0F766E;}
.alt-reason{font-size:0.76rem;color:#64748B;margin-top:2px;}

.stExpander p, .stExpander li, .stExpander span, .stExpander div{color:#0F172A !important;}
.stExpander strong, .stExpander b{color:#0F766E !important;}
.stExpander [data-testid="stMarkdownContainer"]{color:#0F172A !important;}
.stExpander [data-testid="stMarkdownContainer"] *{color:#0F172A !important;}
.stExpander [data-testid="stMarkdownContainer"] strong{color:#0F766E !important;}
</style>
""", unsafe_allow_html=True)

DISEASE_WORDS=["lung","breast","colon","colorectal","ovarian","prostate","pancreatic","leukemia","leukaemia","lymphoma","melanoma","glioma","glioblastoma","liver","kidney","gastric","stomach","bladder","cervical","brain","neuroblastoma","sarcoma","carcinoma","skin","blood","bone","esophageal","thyroid"]
PQ="outputs/parquet/"

@st.cache_data
def load_data():
    lk=pd.read_parquet("outputs/cell_line_lookup.parquet")
    hpa=dict(zip(lk["hpa_name"].dropna(), lk.loc[lk["hpa_name"].notna(),"cellosaurus_id"]))
    meta=lk.set_index("cellosaurus_id")[["disease","lineage"]].to_dict("index")
    samp=pd.read_csv("data/nomenclature/9_DepMap_sample_info.csv", low_memory=False)
    ach=dict(zip(samp["DepMap_ID"], samp["RRID"]))
    geo=pd.read_csv("data/nomenclature/10_GEOInfo.txt", sep="\t", low_memory=False)
    gsm=dict(zip(geo["Geo_accession"], geo["Cellosaurus_ID"]))
    m=pd.read_parquet("outputs/master_with_confidence.parquet")
    for c in ["has_mutations","has_fusions"]:
        if c not in m: m[c]=False
    keep=[k for k in ["cellosaurus_id","official_name","confidence","evidence_count","has_mutations","has_fusions"] if k in m.columns]
    return hpa,ach,gsm,m[keep].drop_duplicates("cellosaurus_id"),meta

@st.cache_data
def load_gene_set():
    g=pd.read_parquet(PQ+"gene_expr_hpa_preprocessed.parquet", columns=["gene_symbol"])
    return set(g["gene_symbol"].dropna().unique())

hpa_to_id,ach_to_id,gsm_to_id,CONF,META=load_data()
GENES=load_gene_set()

def load_justifications(gene):
    path=f"outputs/agentic_results_{gene}.json"
    if not os.path.exists(path): return {},""
    try:
        with open(path,encoding="utf-8") as f: data=json.load(f)
    except Exception: return {},""
    just={r.get("cellosaurus_id"):r.get("justification","") for r in data.get("results",[]) if r.get("cellosaurus_id")}
    return just,data.get("comparative_summary","")

def load_alternatives(gene):
    path=f"outputs/agentic_results_{gene}.json"
    if not os.path.exists(path): return {}
    try:
        with open(path,encoding="utf-8") as f: data=json.load(f)
    except Exception: return {}
    return {r.get("cellosaurus_id"):r.get("alternatives",[]) for r in data.get("results",[]) if r.get("cellosaurus_id")}

def parse_query(text):
    dl=text.lower()
    disease=next((w for w in DISEASE_WORDS if w in dl), None)
    found=[]
    for tok in re.findall(r"[A-Za-z0-9\-]+", text):
        u=tok.upper()
        if u in GENES and u not in found:
            found.append(u)
        if len(found)>=2: break
    gene = found[0] if found else None
    gene2 = found[1] if len(found)>1 else None
    return gene, gene2, disease

@st.cache_data
def score_source(path, gene, which):
    idmap={"hpa":hpa_to_id,"depmap":None,"geo":gsm_to_id,"prot":ach_to_id}[which]
    col="cellosaurus_id" if which=="depmap" else "original_id"
    df=pd.read_parquet(path, columns=[col,"gene_symbol","expression_value"])
    rows=df[df["gene_symbol"]==gene]
    if len(rows)==0: return None
    per=rows.groupby(col)["expression_value"].mean().reset_index()
    per["score"]=per["expression_value"]/per["expression_value"].max()
    per["cellosaurus_id"]=per[col].map(idmap) if idmap is not None else per[col]
    per=per.dropna(subset=["cellosaurus_id"])
    return per.groupby("cellosaurus_id")["score"].max().reset_index()

def _excl_rna_score(excl_gene):
    """HPA + DepMap averaged expression score for an exclusion gene (returns 0-1)."""
    parts=[]
    for path,which in [(PQ+"gene_expr_hpa_preprocessed.parquet","hpa"),(PQ+"gene_expr_depmap_preprocessed.parquet","depmap")]:
        s=score_source(path,excl_gene,which)
        if s is not None: parts.append(s.rename(columns={"score":which}))
    if not parts: return None
    merged=reduce(lambda a,b:a.merge(b,on="cellosaurus_id",how="outer"),parts)
    cols=[c for c in ["hpa","depmap"] if c in merged.columns]
    merged["excl_score"]=merged[cols].mean(axis=1,skipna=True).fillna(0.0)
    return merged[["cellosaurus_id","excl_score"]]

def recommend(gene, disease_filter=None, top_n=10, exclude_genes=None):
    parts={}
    for path,which in [(PQ+"gene_expr_hpa_preprocessed.parquet","hpa"),(PQ+"gene_expr_depmap_preprocessed.parquet","depmap"),(PQ+"gene_expr_geo_preprocessed.parquet","geo"),(PQ+"gene_expr_ccle_proteomics_preprocessed.parquet","prot")]:
        s=score_source(path,gene,which)
        if s is not None: parts[which]=s.rename(columns={"score":which})
    if not parts: return None,0
    c=reduce(lambda a,b:a.merge(b,on="cellosaurus_id",how="outer"),parts.values())
    for w in ["hpa","depmap","geo","prot"]:
        if w not in c: c[w]=float("nan")
    c["expr_score"]=c[["hpa","depmap","geo","prot"]].mean(axis=1,skipna=True)
    c["n_sources"]=c[["hpa","depmap","geo","prot"]].notna().sum(axis=1)
    r=c.merge(CONF,on="cellosaurus_id",how="left")
    r["disease"]=r["cellosaurus_id"].map(lambda x:(META.get(x) or {}).get("disease",""))
    r["lineage"]=r["cellosaurus_id"].map(lambda x:(META.get(x) or {}).get("lineage",""))
    if disease_filter:
        mask=r["disease"].fillna("").str.lower().str.contains(disease_filter.lower())|r["lineage"].fillna("").str.lower().str.contains(disease_filter.lower())
        r=r[mask]
    r["final_score"]=r["expr_score"]*r["confidence"]
    # Exclusion gene penalties: final_score *= (1 - excl_rna_score * 0.5)
    if exclude_genes:
        for eg in exclude_genes:
            excl_df=_excl_rna_score(eg)
            ecol=f"excl_{eg}"
            if excl_df is not None:
                r=r.merge(excl_df.rename(columns={"excl_score":ecol}),on="cellosaurus_id",how="left")
            if ecol not in r.columns: r[ecol]=0.0
            else: r[ecol]=r[ecol].fillna(0.0)
            # REMOVE cells that express the excluded gene above threshold
            r=r[r[ecol]<=0.5]
        r["exclusion_warning"]=False
    else:
        r["exclusion_warning"]=False
    return r.sort_values("final_score",ascending=False).head(top_n),len(r)



def recommend_two(geneA, geneB, disease_filter=None, top_n=10):
    rA, _ = recommend(geneA, disease_filter, top_n=100000)
    rB, _ = recommend(geneB, disease_filter, top_n=100000)
    if rA is None or rB is None or len(rA)==0 or len(rB)==0:
        return None, 0
    a = rA[["cellosaurus_id","official_name","expr_score","confidence","disease","lineage","n_sources","evidence_count","has_mutations","has_fusions"]].rename(columns={"expr_score":"scoreA"})
    b = rB[["cellosaurus_id","scoreA"]].rename(columns={"scoreA":"scoreB"}) if False else rB[["cellosaurus_id","expr_score"]].rename(columns={"expr_score":"scoreB"})
    m = a.merge(b, on="cellosaurus_id", how="inner")
    if len(m)==0:
        return None, 0
    m["expr_score"] = (m["scoreA"] + m["scoreB"]) / 2
    m["final_score"] = m["expr_score"] * m["confidence"]
    m = m.sort_values("final_score", ascending=False)
    return m.head(top_n), len(m)

@st.cache_data
def load_all():
    m=pd.read_parquet("outputs/master_with_confidence.parquet")
    lk=pd.read_parquet("outputs/cell_line_lookup.parquet")[["cellosaurus_id","disease","lineage"]]
    d=m.merge(lk,on="cellosaurus_id",how="left")
    cols=[c for c in ["official_name","cellosaurus_id","confidence","evidence_count","disease","lineage","has_hpa_expr","has_depmap_expr","has_geo_expr","has_proteomics","has_mutations","has_fusions"] if c in d.columns]
    d=d[cols].drop_duplicates("cellosaurus_id")
    return d.rename(columns={"official_name":"Cell line","cellosaurus_id":"Cellosaurus ID","confidence":"Confidence","evidence_count":"Evidence","disease":"Disease","lineage":"Lineage","has_hpa_expr":"HPA","has_depmap_expr":"DepMap","has_geo_expr":"GEO","has_proteomics":"Proteomics","has_mutations":"Mutations","has_fusions":"Fusions"})

# ---------- Header ----------
st.markdown('<div class="head"><div class="brand"><div class="bt"><h1>CellLine<span>Finder</span></h1><p>Multi-omics cell line recommendation</p></div></div><div class="headstats"><div class="hs"><div class="n">2,076</div><div class="l">cell lines</div></div><div class="hs"><div class="n">4</div><div class="l">datasets</div></div><div class="hs"><div class="n">4/5</div><div class="l">validated</div></div></div></div>', unsafe_allow_html=True)

# ---------- SEARCH PAGE ----------
if True:
    sc1,sc2=st.columns([4,1])
    query=sc1.text_input("q", value="show me EGFR lung cancer lines", label_visibility="collapsed", placeholder="Enter a gene or ask in plain English").strip()
    sc2.button("Find", key="do_search", use_container_width=True, type="primary")
    st.markdown('<p style="font-size:0.82rem;color:#64748B;margin:0.6rem 0 0.2rem 0;"><b style="color:#0F172A;">Exclude genes</b> (optional) - leave out cell lines that also strongly express these genes, useful for studying your target gene in isolation</p>', unsafe_allow_html=True)
    excl_raw=st.text_input("excl", placeholder="e.g. TP53, MYC", label_visibility="collapsed").strip()
    exclude_genes=[g.strip().upper() for g in excl_raw.split(",") if g.strip()] if excl_raw else []
    if query:
        gene,gene2,disease=parse_query(query)
        if gene is None:
            st.warning("No recognised gene found. Try a gene symbol such as EGFR or TP53.")
        else:
            st.markdown(f'<p class="parsed">Detected gene <b>{gene}</b>'+(f' &middot; tissue <b>{disease}</b>' if disease else '')+(f' &middot; excluding <b>{", ".join(exclude_genes)}</b>' if exclude_genes else '')+'</p>', unsafe_allow_html=True)
            JUST,SUMMARY=load_justifications(gene)
            ALTS=load_alternatives(gene)
            r,total=recommend(gene,disease,exclude_genes=exclude_genes)
            if not JUST and r is not None and len(r)>0:
                st.info("No AI explanations saved for this gene yet.")
                if st.button(f"Generate AI explanations for {gene}", key="gen_ai"):
                    with st.spinner(f"Generating AI explanations for {gene} (this takes a moment)..."):
                        try:
                            from src.models.agentic.pipeline import run as _run_ai
                            _run_ai(gene, disease_filter=disease, top_n=10)
                            st.success("Done. Reloading...")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Could not generate: {e}. Make sure Ollama is running.")
            if r is None or len(r)==0:
                st.warning(f"No results for {gene}"+(f" in {disease}" if disease else "")+".")
            else:
                ctx=f" in <b>{disease}</b>" if disease else ""
                st.markdown(f'<div class="resbar">Showing top {len(r)} of <b>{total}</b> cell lines for {gene}{ctx}</div>', unsafe_allow_html=True)
                if SUMMARY:
                    st.markdown('<div class="summary"><div class="st">AI overview - comparing the top cells</div>'+SUMMARY.replace(chr(10),"<br>")+'</div>', unsafe_allow_html=True)
                for i,(_,row) in enumerate(r.iterrows(),1):
                    strength="strongly" if row["expr_score"]>=0.5 else "moderately"
                    has_mut=bool(row.get("has_mutations",False)); has_fus=bool(row.get("has_fusions",False))
                    def contrib(label,w):
                        v=row.get(w)
                        if v is None or pd.isna(v): lvl,cls="No data","none"
                        elif v>=0.66: lvl,cls="High","high"
                        elif v>=0.33: lvl,cls="Medium","med"
                        else: lvl,cls="Low","low"
                        return f'<div class="cbox"><div class="cn">{label}</div><div class="cl {cls}">{lvl}</div></div>'
                    breakdown=contrib("HPA RNA","hpa")+contrib("DepMap","depmap")+contrib("GEO","geo")+contrib("Proteomics","prot")
                    excl_warn=bool(row.get("exclusion_warning",False))
                    excl_badge='<span class="excl-warn">⚠ excludes: '+", ".join(exclude_genes)+'</span>' if excl_warn else ''
                    excl_detail=""
                    if excl_warn and exclude_genes:
                        parts_excl=[]
                        for eg in exclude_genes:
                            sc=row.get(f"excl_{eg}",None)
                            if sc is not None and sc>0.5:
                                parts_excl.append(f"{eg} ({sc:.2f})")
                        if parts_excl:
                            excl_detail=f'<div class="excl-note">⚠ Also expresses: {", ".join(parts_excl)} — may confound {gene} experiments</div>'
                    chips='<span class="chip">complete model</span>' if (has_mut and has_fus) else ''
                    evc=int(row["evidence_count"]) if "evidence_count" in row and pd.notna(row["evidence_count"]) else 0
                    dis=row.get("disease") or ""; lin=row.get("lineage") or ""
                    dis=dis if isinstance(dis,str) and dis and dis!="nan" else ""
                    lin=lin if isinstance(lin,str) and lin and lin!="nan" else ""
                    evp=[]
                    if dis: evp.append(f'<span class="d">{dis}</span>')
                    if lin: evp.append(f'{lin} lineage')
                    evp.append(f'expresses {gene} {strength} across {int(row["n_sources"])} of 4 datasets')
                    if has_mut: evp.append("mutation reported")
                    if has_fus: evp.append("fusion reported")
                    ev='<div class="ev"><b>Evidence:</b> '+" &middot; ".join(evp)+f'. Backed by {evc} of 3 nomenclature sources.{chips}</div>'
                    cc="card top" if i<=3 else "card"; rc="rank hi" if i<=3 else "rank"
                    st.markdown(f'<div class="{cc}"><div class="crow"><span class="{rc}">{i:02d}</span><div><span class="name">{row["official_name"]}</span> <span class="cid">{row["cellosaurus_id"]}</span>{excl_badge}</div><div class="score"><div class="n">{row["final_score"]:.2f}</div><div class="l">Fit score</div></div></div><div class="metrics"><div class="m"><div class="v teal">{row["expr_score"]:.2f}</div><div class="k">Expression</div></div><div class="m"><div class="v">{row["confidence"]:.0%}</div><div class="k">Confidence</div></div><div class="m"><div class="v">{int(row["n_sources"])}/4</div><div class="k">Sources</div></div><div class="m"><div class="v">{evc}/3</div><div class="k">Evidence</div></div></div><div class="contrib">{breakdown}</div>{ev}{excl_detail}</div>', unsafe_allow_html=True)
                    jtext=JUST.get(row["cellosaurus_id"],"")
                    if jtext and jtext.strip() and "RECOMMENDATION" in jtext.upper():
                        with st.expander("Why this cell? (AI explanation)"):
                            shown=False
                            for line in jtext.split(chr(10)):
                                line=line.strip()
                                if not line: continue
                                printed=False
                                for lbl in ["RECOMMENDATION","KEY REASON","EVIDENCE SUMMARY","TRADE-OFFS","BEST FOR"]:
                                    if lbl in line.upper():
                                        txt=line.split(":",1)[-1].strip()
                                        if txt:
                                            st.markdown(f"**{lbl.title()}:** {txt}")
                                            shown=True
                                        printed=True; break
                                if not printed and len(line)>3:
                                    st.markdown(line); shown=True
                            if not shown:
                                st.caption("No detailed explanation available for this cell.")
                    row_alts=ALTS.get(row["cellosaurus_id"],[])
                    if row_alts:
                        with st.expander(f"Similar alternatives ({len(row_alts)})"):
                            st.caption("Cell lines with the most similar multi-omics profile — useful as experimental backups or orthogonal validation.")
                            rows_html=""
                            for alt in row_alts:
                                sim=alt.get("similarity_score",0)
                                aname=alt.get("official_name",alt.get("cellosaurus_id",""))
                                acid=alt.get("cellosaurus_id","")
                                shared=alt.get("shared_data_types",[])
                                reason=alt.get("similarity_reason","")
                                chips="".join(f'<span class="alt-chip">{dt}</span>' for dt in shared[:4])
                                rows_html+=f'<div class="alt-row"><span class="alt-sim">{sim:.2f}</span><div><span class="alt-name">{aname}</span><span class="alt-cid">{acid}</span><div class="alt-reason">{reason}</div></div><div class="alt-chips">{chips}</div></div>'
                            st.markdown(f'<div style="padding:0.2rem 0">{rows_html}</div>', unsafe_allow_html=True)

