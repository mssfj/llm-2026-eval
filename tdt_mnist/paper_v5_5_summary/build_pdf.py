"""Standalone handoff summary from existing theory/results. No experiment execution."""
from pathlib import Path
import csv,json,hashlib,html,re,statistics,io
import pymupdf as fitz
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,PageBreak,Table,TableStyle,Spacer
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;DOC=ROOT/'doc'
MD=DOC/'TDT-v5.5要約版_引き継ぎ資料.md';DEST=MD.with_suffix('.pdf')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return list(csv.DictReader(p.open()))
results=ROOT/'tdt_mnist/results';text=MD.read_text();sources=[MD,Path(__file__).resolve(),ROOT/'tdt_mnist/paper_v5_5/build_pdf.py']
for name,file in [('paper_v5','TDT-v5_本文.txt'),('paper_v5_1','TDT-v5.1_追加本文.txt'),('paper_v5_2','TDT-v5.2_追加本文.txt'),('paper_v5_3','TDT-v5.3_追加本文.txt'),('paper_v5_4','TDT-v5.4_追加本文.txt'),('paper_v5_5','TDT-v5.5_追加本文.txt')]:sources.append(ROOT/'tdt_mnist'/name/file)
for name in ['train.py','residual_stream.py','residual_followup_models.py','gpu_evaluation_engines.py','run_gpu_e17a.py','audit_gpu_e17a.py','activation_quantization.py','GPU_E17A_REPRODUCTION_PREREGISTRATION.md','E18_E20_PREREGISTRATION.md','ENGINE_OPTIMIZATION_SUMMARY.md']:sources.append(ROOT/'tdt_mnist'/name)
computed=[]
for directory in ['residual-stream-a8-e17-20260908','residual-followups-e18-e20-20260908']:
 root=results/directory;per=read(root/'per_seed/results.csv');agg=read(root/'aggregate/results.csv');sources.extend([root/'per_seed/results.csv',root/'aggregate/results.csv'])
 for a in agg:
  vals=[float(r['test_accuracy_percent']) for r in per if r['condition']==a['condition']];assert len(vals)==3
  mean=statistics.mean(vals);sd=statistics.stdev(vals)
  assert abs(mean-float(a['test_mean_percent']))<1e-10 and abs(sd-float(a['test_sample_std_percent']))<1e-10
  assert f'{mean:.3f}±{sd:.3f}' in text,a['condition']
  computed.append(dict(condition=a['condition'],seed_count=3,test_mean_percent=mean,test_sample_sd_percent=sd,source=str((root/'per_seed/results.csv').relative_to(ROOT))))
gpu=results/'gpu-e17a-reproduction-20260908';gper=[json.loads((gpu/f'per_seed/seed{s}/summary.json').read_text()) for s in range(3)]
for s in range(3):sources.append(gpu/f'per_seed/seed{s}/summary.json')
gvals=[100*r['test']['accuracy'] for r in gper];gmean=statistics.mean(gvals);gsd=statistics.stdev(gvals)
assert f'{gmean:.3f}±{gsd:.3f}' in text
assert [r['first_firing_divergence'] for r in gper]==[4,21,10]
computed.append(dict(condition='GPU_E17a',seed_count=3,test_mean_percent=gmean,test_sample_sd_percent=gsd,source=str(gpu.relative_to(ROOT))+'/per_seed/seed*/summary.json'))
stop=results/'fast-engine-e17a-20260908'
s=json.loads((stop/'status.json').read_text());assert s['stopped'] and not s['active_seeds'] and [r['logged_intervals'] for r in s['runs']]==[12000,5282,5465]
for name in ['status.json','stop_audit.json','stop_manifest_sha256.json']:sources.append(stop/name)
for directory in ['allocation-ablations-16blocks-20260908','gpu-evaluation-16blocks-20260908']:
 for name in ['aggregate.csv','audit.json','README.md']:sources.append(results/directory/name)
sources.extend([gpu/'report.json',gpu/'audit.json',results/'depth-precision-16layer-100k-20260907/aggregate.csv',ROOT/'tdt_mnist/paper_v5/normalized_records.json'])
for name in ['a3-ablation-100k-20260907','depth-100k-20260907','depth-activation-100k-20260907','a3-improvements-16layer-20260907','backprop-a3-inference-16x79-20260907','qat-a3-16x79-20260907']:
 assert (results/name).is_dir(),name
 # Primary aggregate schema differs across experiment generations; retain the paper text plus available aggregates.
 for f in (results/name).glob('*aggregate*.csv'):sources.append(f)
with (OUT/'key_results.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(computed[0]));w.writeheader();w.writerows(computed)
sources.append(OUT/'key_results.csv')
# Reuse only the existing typography/Markdown helpers, not its report or experiment code.
helper=(ROOT/'tdt_mnist/paper_v5_5/build_pdf.py').read_text();helper=helper[helper.index("pdfmetrics.registerFont"):helper.index('def label(')]
exec(helper)
def footer(c,d):
 c.setFont('CJK',8);c.setFillColor(blue);c.drawString(48,815,'TDT-v5.5要約版 | 理論・主要実験・LLM引き継ぎ | 2026-09-08');c.drawRightString(547,25,str(d.page))
p('TDT-v5.5要約版\n他LLMへの引き継ぎ資料','title')
md(MD)
main=OUT/'summary-body.pdf'
Doc(str(main),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=48,bottomMargin=43,title='TDT-v5.5要約版：理論・主要結果・LLM引き継ぎ',author='TDT experimental study').build(story,onFirstPage=footer,onLaterPages=footer)
hashes={str(f.relative_to(ROOT)):sha(f) for f in sorted(set(sources))}
(OUT/'handoff_sources_sha256.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2)+'\n')
pdf=fitz.open(main);pdf.embfile_add('handoff.md',MD.read_bytes(),filename=MD.name)
pdf.embfile_add('handoff_sources_sha256.json',(OUT/'handoff_sources_sha256.json').read_bytes(),filename='handoff_sources_sha256.json')
pdf.embfile_add('key_results.csv',(OUT/'key_results.csv').read_bytes(),filename='key_results.csv')
pdf.set_metadata({'title':'TDT-v5.5要約版 — v5理論とv5.5までの主要実験・LLM引き継ぎ','author':'TDT experimental study','subject':'Standalone handoff: theory assumptions, E1-E20, CPU/GPU engines, limitations and artifact map','creator':'ReportLab / PyMuPDF'})
pdf.save(DEST,garbage=4,deflate=True);pdf.close();pdf=fitz.open(DEST)
outside=[]
for i,page in enumerate(pdf):
 for block in page.get_text('dict')['blocks']:
  for line in block.get('lines',[]):
   for span in line['spans']:
    if not page.rect.contains(fitz.Rect(span['bbox'])):outside.append([i+1,span['text']])
 page.get_pixmap(matrix=fitz.Matrix(.7,.7)).save(OUT/f'preview-{i+1:02}.png')
assert not outside,outside
assert pdf.embfile_get('handoff.md')==MD.read_bytes()
extracted='\n'.join(page.get_text() for page in pdf)
for t in ['90.637','95.987','90.587','select_actions','G_minus','5,282','5,465','graph.replay()','未判定']:assert t in extracted,t
assert '\ufffd' not in extracted
(OUT/'extracted_text.txt').write_text(extracted)
validation=dict(passed=True,pages=len(pdf),recomputed_test_conditions=len(computed),source_files=len(hashes),source_hashes_verified=all(sha(ROOT/f)==h for f,h in hashes.items()),markdown_attachment_exact=True,no_text_outside_page=True,new_training=False,test_reevaluated=False,pdf_sha256=sha(DEST),markdown_sha256=sha(MD))
(OUT/'pdf_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n');print(json.dumps(validation,ensure_ascii=False,indent=2))
