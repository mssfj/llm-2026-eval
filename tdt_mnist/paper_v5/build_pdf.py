"""Build TDT-v5 from the original v4 PDF and all nine completed experiment groups.

Run with reportlab and pymupdf installed. The font defaults to PyMuPDF's bundled
CJK font; all experiment numbers are read from saved CSV/JSON, never transcribed.
"""
from pathlib import Path
import csv
import hashlib
import html
import io
import json
import statistics as st
import zipfile

import pymupdf as fitz
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RESULTS = ROOT / 'tdt_mnist/results'
ORIGINAL = ROOT / 'TDT-v4_離散状態遷移学習理論.pdf'
DEST = ROOT / 'TDT-v5_離散状態遷移学習理論.pdf'
GROUPS = [
 ('E1', 'mnist-1000-seed0', '1k・単一seedの初期動作確認', 1000, 3000),
 ('E2', 'grid-20260905', '1k・ブロック×閾値の比較', 1000, 3000),
 ('E3', 'counter-comparison-20260905', '1k・カウンタあり／なし', 1000, 3000),
 ('E4', 'threshold-1-32-20260905', '1k・閾値1～32の全点掃引', 1000, 3000),
 ('E5', 'mnist-10000-grid-20260905', '10k・ブロック×閾値の比較', 10000, 3000),
 ('E6', 'mnist-100000-grid-20260905', '100k・ブロック×閾値の比較', 100000, 3000),
 ('E7', 'activation-grid-10k-20260905', '10k・活性化精度の比較', 10000, 3000),
 ('E8', 'length-grid-100k-20260905', '100k・大ブロックと学習区間', 100000, None),
 ('E9', 'blocks-8-16-32-100k-12k-20260905', '100k・12,000区間の小ブロック比較', 100000, 12000),
]


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


records = []
group_data = {}
source_hashes = {}
for gid, name, title, params, steps in GROUPS:
    directory = RESULTS / name
    for f in directory.rglob('*'):
        if f.is_file() and f.suffix in ('.csv', '.json', '.md', '.py'):
            source_hashes[str(f.relative_to(ROOT))] = digest(f)
    if gid == 'E1':
        s = json.loads((directory / 'summary.json').read_text())
        rows = [dict(seed=0, block_size=1, threshold=4,
                     initial_val_loss=s['initial_validation']['loss'],
                     initial_val_accuracy=s['initial_validation']['accuracy'],
                     val_loss=s['final_validation']['loss'], val_accuracy=s['final_validation']['accuracy'],
                     test_accuracy=s['test']['accuracy'], test_loss=s['test']['loss'],
                     total_fires=s['total_fires'], train_forward_calls=s['train_forward_calls'])]
        aggregates = []
    else:
        rows = read_csv(directory / 'per_seed.csv')
        aggregates = read_csv(directory / 'aggregate.csv')
    extra = json.loads((directory / 'summaries.json').read_text()) if (directory / 'summaries.json').exists() else []
    normalized = []
    for index, raw in enumerate(rows):
        r = dict(raw)
        r.update(group=gid, record_id=f'{gid}-{index+1:03d}', params=params,
                 steps=int(raw.get('steps', steps)), seed=int(raw['seed']),
                 block_size=int(raw.get('block_size', 32 if gid == 'E7' else 8)),
                 threshold=int(raw.get('threshold', 8)), precision=raw.get('precision', 'a32'),
                 method=raw.get('method', 'counter'), measurements=16 if gid == 'E1' else 64)
        for k in ['initial_val_loss', 'initial_val_accuracy', 'val_loss', 'val_accuracy', 'test_accuracy', 'total_fires']:
            r[k] = float(r[k])
        assert r['val_loss'] < r['initial_val_loss'], r['record_id']
        assert r['val_accuracy'] > r['initial_val_accuracy'], r['record_id']
        assert int(r['train_forward_calls']) == 2*r['measurements']*r['steps']
        if extra:
            s = next(s for s in extra if s['seed'] == r['seed'] and
                     (s.get('method') == r['method'] if gid == 'E3' else s.get('precision') == r['precision']))
            counter = s.get('counter_distribution')
            if counter:
                counter.setdefault('saturated_fraction', counter['saturated_count'] / counter['count'])
                for k in ['min', 'max', 'mean', 'abs_max', 'abs_mean', 'capacity', 'saturated_count', 'saturated_fraction']:
                    r['counter_' + k] = (min(map(int, counter['histogram'])) if k == 'min' else max(map(int, counter['histogram'])) if k == 'max' else counter[k])
        if 'counter_saturated_count' in r:
            assert int(r['counter_saturated_count']) == 0
        normalized.append(r)
        records.append(r)
    # Recompute displayed means and sample SDs directly from per-seed records.
    for a in aggregates:
        members = [r for r in normalized if all(str(r[k]) == str(a[k]) for k in
                   ['steps', 'block_size', 'threshold', 'method', 'precision'] if k in a)]
        assert len(members) == int(a['seeds']), (gid, a)
        for metric in ['val_accuracy', 'test_accuracy', 'val_loss', 'total_fires']:
            values = [float(r[metric]) for r in members]
            assert abs(st.mean(values)-float(a[metric+'_mean'])) < 1e-10
            assert abs(st.stdev(values)-float(a[metric+'_std'])) < 1e-10
    group_data[gid] = (directory, normalized, aggregates)

assert len(records) == 244
assert sum(len(group_data[g][2]) or 1 for g, *_ in GROUPS) == 82
font_path = OUT / 'CJK-font.ttf'
font_path.write_bytes(fitz.Font('japan').buffer)
pdfmetrics.registerFont(TTFont('CJK', str(font_path)))
pdfmetrics.registerFont(TTFont('Latin', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
BLUE = colors.HexColor('#173c61')
styles = {
 'body': ParagraphStyle('body', fontName='CJK', fontSize=9, leading=15, wordWrap='CJK', spaceAfter=7),
 'small': ParagraphStyle('small', fontName='CJK', fontSize=7.1, leading=10.5, wordWrap='CJK', spaceAfter=4),
 'h1': ParagraphStyle('h1', fontName='CJK', fontSize=17, leading=24, textColor=BLUE, spaceAfter=14, keepWithNext=True),
 'h2': ParagraphStyle('h2', fontName='CJK', fontSize=12, leading=18, textColor=BLUE, spaceBefore=10, spaceAfter=8, keepWithNext=True),
 'title': ParagraphStyle('title', fontName='CJK', fontSize=25, leading=39, textColor=BLUE, spaceAfter=20),
 'cell': ParagraphStyle('cell', fontName='CJK', fontSize=7.1, leading=10, wordWrap='CJK'),
}
story = []
text_log = []
headings = []


def para(text, style='body'):
    text_log.append(text)
    # Use a Latin/math font for Greek and mathematical symbols lacking in CJK font.
    escaped = html.escape(str(text)).replace('\n', '<br/>')
    for symbol in set(text):
        if symbol in 'αΘλφΣμσΔ∈√τεℓ−ᵀ':
            escaped = escaped.replace(symbol, f'<font name="Latin">{symbol}</font>')
    p = Paragraph(escaped, styles[style])
    story.append(p)
    return p


def section(title):
    if story:
        story.append(PageBreak())
    p = para(title, 'h1')
    p.bookmark = f'section{len(headings)}'
    headings.append(title)


def table(headers, rows, widths=None):
    content = [[Paragraph(html.escape(str(v)), styles['cell']) for v in row] for row in [headers, *rows]]
    t = Table(content, colWidths=widths, repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e6eef5')),
        ('TEXTCOLOR', (0, 0), (-1, 0), BLUE),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LINEBELOW', (0, 0), (-1, 0), .6, BLUE),
        ('LINEBELOW', (0, 1), (-1, -1), .25, colors.HexColor('#d5dce3')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f6f8fa')]),
        ('LEFTPADDING', (0, 0), (-1, -1), 5), ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.extend([t, Spacer(1, 10)])
    text_log.append('\n'.join(' | '.join(map(str, row)) for row in [headers, *rows]))


def figure(path, caption, max_height=260):
    if not path.exists():
        return
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    scale = min(495/w, max_height/h)
    story.append(Image(str(path), width=w*scale, height=h*scale))
    para(caption, 'small')


def pm(row, metric, percent=False):
    multiplier = 100 if percent else 1
    places = 2 if percent else (1 if metric == 'total_fires' else 4)
    return f"{multiplier*float(row[metric+'_mean']):.{places}f} ± {multiplier*float(row[metric+'_std']):.{places}f}"


def condition(r):
    if r.get('method') == 'no_counter':
        return '即時更新・Cなし'
    if 'precision' in r:
        return 'W3' + r['precision'].upper()
    prefix = f"L={r['steps']}, " if 'steps' in r else ''
    return prefix + f"b={r.get('block_size', 8)}, θ={r.get('threshold', 8)}"


def aggregate_table(gid):
    _, _, ag = group_data[gid]
    table(['条件', '検証精度 %', 'テスト精度 %', '検証loss', '発火数'],
          [[condition(a), pm(a,'val_accuracy',True), pm(a,'test_accuracy',True),
            pm(a,'val_loss'), pm(a,'total_fires')] for a in ag], [110,98,98,93,96])


para('TECHNICAL PAPER / VERSION 5', 'small')
story.append(Spacer(1, 70))
para('TDT-v5\n離散状態遷移に基づく\n前進評価型3値学習\nフレームワーク', 'title')
para('MNIST実証編を統合した改訂版', 'h2')
para('Ternary Dynamics Theory\nForward-Only Discrete Learning with Edge Evidence', 'body')
story.append(Spacer(1, 32))
para('2026年9月5日 実験記録統合\n9実験群・82条件行・244 run記録（再現実行を含む）')
para('基礎文書：TDT-v4_離散状態遷移学習理論.pdf\n改訂範囲：理論と実装の対応、全条件の定量結果、seed別記録、カウンタ・活性化診断、今後の検証課題。')
para('v4の数式・参考文献を含む原文全10ページは、版間の追跡のため付録Dに収録する。CSV・JSON・生成時の資料一式はPDF内のZIP添付および同名の外部ZIPに収録する。', 'small')

section('要旨とv5で更新した結論')
para('TDT-Dは、重みの離散成分を学習開始から{-1, 0, +1}に保ち、候補対の前進損失差を確率的な3値票へ変換し、辺カウンタの証拠から隣接重み遷移を選ぶ学習則である。v5では、2026年9月5日に完了したMNISTの9実験群を統合した。潜在浮動小数点重み、逆伝播、STEを使わない同期実装について、1k・10k・100kパラメータで初期状態から検証損失と精度が改善することを確認した。')
para('全244 run記録で最終検証損失が初期値より低下し、検証精度が上昇した。ただし同一seed・同一設定の再現実行を含むため、244回の独立な統計的反復とは解釈しない。複数seed比較は原則seed=0,1,2の平均と標本標準偏差であり、同じデータ分割を使う。')
para('100k・FP32・12,000区間・発火閾値8では、測定したブロック8/16/32/64/128/256のうち16が最も高い平均検証精度87.80±0.26%を示し、テスト精度は88.55±0.22%だった。ブロック32との差は小さい。10kで活性化をA3まで量子化しても学習は進んだが、テスト精度はA32の84.64±0.73%から65.21±1.54%へ低下した。')
para('これらは「MNISTの小規模モデルにおける前進評価型の直接三値学習」の成立を支持する。iid初通過理論、局所遷移場の低実効次元仮説、相互作用比、巨大Transformerへの拡張、総学習コスト・帯域・通信の優位性を検証したものではない。本日の目的は学習成立の確認であり、コスト優位の判定は実施していない。')
para('構成', 'h2')
para('1. 理論・実装の対応\n2. 共通実験条件\n3. 全実験の登録簿\n4. E1～E9の全条件結果\n5. 横断的解釈と未検証事項\n付録A. 全seedの精度・損失・発火数\n付録B. 全seedのカウンタ統計・層別更新数\n付録C. 再現方法・資料とハッシュ\n付録D. TDT-v4原文（歴史的基礎文書）')

section('1. 理論・実装の対応')
para('1.1 状態と前向き計算', 'h2')
para('v4の定義を引き継ぎ、各層の実効重みは W_l = α_l T_l、T_l ∈ {-1,0,+1} とする。状態にはTだけでなく、辺カウンタC、層スケールα、票スケールS、乱数・ブロック選択などの内部状態を含める。合法な重み遷移は -1 ↔ 0 ↔ +1 であり、-1と+1の直接反転は行わない。')
para('今回の層スケールは α_l = 1 / √(d_l(1−π_0))、π_0=1/3、gain=1で固定した。学習パラメータ数は三値重みの個数であり、固定スケールを数えない。重みはINT8コンテナに保存し、forwardでFP32に変換して固定αを掛ける。低ビット専用GEMMやビット詰めの評価は行わない。')
para('1kモデル：logits = α_1 T_1 x（100→10、1線形層）。\n10k/100kモデル：h = ReLU(α_1 T_1 x)、logits = α_2 T_2 h（90→100/1000→10、2線形層）。\n交差エントロピー損失をFP32で計算する。発火はニューロン活性化ではなく、三値重みの隣接状態への更新を意味する。')
para('1.2 候補対・票・カウンタ', 'h2')
para('各区間でb個の座標を重複なしで無作為抽出する。各測定で現在値に接続する辺を1本選び、0なら(-1,0)または(0,+1)を等確率で選ぶ。辺の向きφを独立な±1で定め、全選択座標を同時に横断する候補T+とT−を生成する。ブロック内の全重みを同一方向へ結び付けるblock tyingではない。bはブロックの個数ではなく同時に摂動する座標数である。')
para('同じミニバッチで y = loss(T+) − loss(T−) を計算し、s_e = −y φ_e / S とする。sを[-1,1]へクリップし、その絶対値の確率で符号±1、それ以外は0を票q_eとして返す。確率丸めはクリップ後の値に対して条件付き不偏であり、クリップ前の信号全体の不偏性は主張しない。')
para('各測定で C_e ← clip(C_e + q_e, −127,127)。リークλ=1。辺の低い端点から高い端点への支持を正とする。区間末に、現在端点から外向きの支持supportが閾値θ以上の辺を候補にし、score = support × S / max(測定回数,1) の大きい順に最大1座標を更新する。0の2辺が競合する場合も1辺のみ選ぶ。検証損失による更新の採否選別はしない。')
para('1.3 v4の一般形から今回の同期版への限定', 'h2')
table(['項目','今回の実装'],[
 ['発火時刻','K測定を完了した区間末だけで判定。途中で閾値に達しても即時発火しない。'],
 ['リセット','発火の有無を問わず、毎区間の終了時に全カウンタを破棄。区間をまたぐ証拠を保持しない。'],
 ['票の陳腐化','区間内はTとSを固定し、区間間で全証拠を捨てる。非同期版のage・version管理は未実験。'],
 ['Sの更新','初期0.02、下限1e−5。区間内の|y|の上側中央値を用い、区間間のみEMA係数0.1で更新。'],
 ['状態の精度','T: INT8三値、C: INT8、辺測定回数: INT32、αとS: FP32。浮動小数点の潜在重みは持たない。'],
 ['更新上限','最大1重み/区間。カウンタなし対照のみ、各測定で最大1重み/測定。'],
 ], [100,395])

section('1.4 理論式の継承と実験上の扱い')
para('v4の固定長iid票では、票の平均μ・分散σ²に対して Var(q_mean(K))=σ²/K、SNR_K=|μ|√K/σ となる。初通過の命題はこれと別であり、p+>p−>0、固定状態、iid票、C_0=0、対称吸収境界±Θという条件に限定される。')
para('ε = 1 / [1 + (p+ / p−)^Θ]\nE[τ] = Θ(1−2ε)/(p+−p−)')
para('今回の「K回後だけ判定し全リセット」という打ち切り方式は、この吸収型初通過過程ではない。閾値θ=1も毎測定の即時更新と同じではない。実票のiid性、誤方向初通過率、待ち時間式の一致は評価していない。特にK=64かつ各票が±1/0で毎区間ゼロから始まるため、|C|≤64であり、容量±127に届かないことは構造的にも予測される。飽和0だけから長時間積分でのC8十分性を結論しない。')
para('離散行動の真の損失差をΔ_i(a)とし、stayを含む合法行動集合A_iに対して R_i = Δ_i(選択行動) − min_{a∈A_i} Δ_i(a) を局所regretとする。正解時の平均改善G+と誤行動時の平均悪化上限G−に関する条件が成り立つとき、期待降下の十分条件は p > G−/(G++G−) である。単なる正解率50%超を一般的な合格条件にはしない。')
para('また、領域Aからの初退出まで条件付き損失ドリフトが−γ以下で、損失に下界F_infがあれば、v4の有限停止時刻評価 E[min(τ_A,m)] ≤ (F(w_0)−F_inf)/γ を継承する。これは生涯総発火数、大域収束、循環排除、局所最適への到達を保証しない。今回の最終損失低下は、この条件付きドリフト仮定を実証したものではない。')
para('v4の局所遷移場モデル y = dᵀφ + η_int + ε_data、低実効次元仮説、相互作用比R_int、共通seedによるscalar all-reduce、コストモデルと引用文献は付録Dに原文を保存した。今回の結果から、これらを証明済み・実証済みへ変更しない。')

section('2. 共通実験条件と評価方法')
table(['項目','条件'],[
 ['データ','MNIST公式訓練集合から訓練10,000件と独立検証1,000件。公式テスト10,000件。'],
 ['前処理','画像を255で割り、適応平均プーリング、(x−0.1307)/0.3081で正規化。1kは10×10、10k/100kは9×10。'],
 ['構造','1k:100→10、10k:90→100→10、100k:90→1000→10。バイアスなし。隠れ層はReLU。'],
 ['学習','ミニバッチ128、復元抽出。標準K=64、最大1座標更新、λ=1、INT8カウンタ。E1のみK=16。'],
 ['反復','E1はseed=0のみ。他はseed=0,1,2。データ分割seed=0、バッチ乱数seed=seed+100000。'],
 ['E1の例外','初期実装の乱数列を使用し、バッチ乱数の独立指定前。単一seedの動作確認として別に扱う。'],
 ['評価','検証は初期・500区間ごと・最終。テストは各事前指定学習長の終了後1回。oracleはE1以外無効。'],
 ['表の値','精度は%、平均±標本標準偏差。lossは交差エントロピー。発火数は隣接重み更新イベント数。'],
 ['計算量の記録','訓練forward=2KL。同じLなら同じ候補対数。Lを変える比較は学習量の比較であり、総コスト優劣は判定しない。'],
 ],[100,395])
para('1区間はデータ全体を1巡するepochではなく、K回の候補対測定と区間末の更新をまとめた単位である。コードは勾配計算を無効にし、backward・STE・勾配オプティマイザを使わない。3seedはデータ分割の反復ではない。複数条件を検証データで探索したため、その最高平均を独立な未知データ性能の保証とは扱わない。')
para('カウンタ統計は各区間で一度以上測定された辺のリセット直前値を全区間で集計する。未訪問ゼロは除外する。符号付き平均と絶対値平均を分け、最大は原則全seed最大の|C|を示す。飽和とは|C|=127であり、発火閾値への到達とは別である。初期E1には追加前のカウンタ分布ログがないため「未記録」とする。カウンタなし対照は「存在しない」であり、ゼロと補完しない。')

section('3. 本日完了した全実験の登録簿')
table(['ID','実験群','パラメータ','条件行','run数'],
      [[g,t,f'{p:,}',len(group_data[g][2]) or 1,len(group_data[g][1])] for g,n,t,p,s in GROUPS],
      [30,260,75,65,65])
para('合計82条件行・244 run記録。E4はE2の一部閾値を、E3のcounterはE2の1条件を、E7のA32はE5の1条件を再実行している。同一設定の再現結果を削除せず各実験群に収録する一方、独立したseed数を水増ししない。E8の子フォルダは親の集計と重複するため、登録簿では親の27 runのみ数える。')
para('実行中断された部分ログや、起動前に取り消された設定は完了結果に含めない。最新のE9は12,000区間のみであり、E9として3,000・6,000区間の別実験を行ったとは記載しない。学習途中の検証曲線は最終結果と区別する。')

interpretations = {
 'E2': '全36 runで改善。検証精度の最高平均はb=8・θ=8の79.73±0.55%。大きいブロックが常に良いわけではなく、b=32・θ=32では17.70±4.33%にとどまった。',
 'E3': '同じ384,000回の訓練forwardで、counterは検証79.73%、no_counterは49.27%。ただしcounterはK回重みを固定して区間末に更新、no_counterは非ゼロ票で各測定直後に更新するため、更新機会が64倍異なる。これは証拠蓄積と待機を合わせた比較であり、蓄積だけの純粋な因果効果を分離した実験ではない。θ=8はcounterにのみ適用する。',
 'E4': '閾値1～32を1刻みで全点実行。平均検証精度はθ=8が最高だが、θ=5～9は79.47～79.73%と近接する。高閾値では発火が減り、θ=32の平均発火数は85.3回。固定K・区間末リセットの下での結果である。',
 'E5': '90→100→10の2層で両層の更新を確認。最高平均はb=32・θ=8の検証83.03±0.75%、テスト84.64±0.73%。1kとは入力形状と層数も異なるため、パラメータ数だけの効果とは解釈しない。',
 'E6': '90→1000→10の2層で両層の更新を確認。最高平均はb=32・θ=8の検証79.30±1.71%、テスト80.09±0.08%。10kの同条件より検証精度が3.73ポイント低いが、同じ3,000区間・最大1座標更新での比較であり、十分学習後の表現能力の優劣を示さない。',
 'E7': '全15 runで検証損失と精度が改善。A32は以前の10k・b=32・θ=8と全3seedで損失・精度・発火数・層別更新数まで一致。A3でも学習するが、今回の量子化方式では精度低下が大きい。',
 'E8': '100k・FP32・θ=8に固定し、b=64/128/256とL=3000/6000/12000の全9条件を3seedで実行。全ブロックで長く学習すると平均精度が向上した。同じseed・blockの学習経過は異なる長さの実験間で一致し、27組の接頭区間照合を通過した。',
 'E9': '100k・FP32・θ=8・L=12000に固定。平均検証精度はb=16が最高だが、b=32との差は検証0.23、テスト0.13ポイントと小さい。発火数はbに伴い増えるが精度は単調ではない。全9 runで両層の更新とカウンタ飽和0を確認した。',
}

for gid, name, title, params, steps in GROUPS:
    section(f'4.{int(gid[1:])} {gid}：{title}')
    directory, rows, ag = group_data[gid]
    para(f'出典：tdt_mnist/results/{name}/', 'small')
    if gid == 'E1':
        r=rows[0]
        para('seed=0、1,000重み、b=1、θ=4、K=16、L=3000、batch=128。')
        table(['指標','初期','最終'],[
          ['検証loss',f"{r['initial_val_loss']:.4f}",f"{r['val_loss']:.4f}"],
          ['検証精度 %',f"{100*r['initial_val_accuracy']:.2f}",f"{100*r['val_accuracy']:.2f}"],
          ['テスト精度 %','—',f"{100*r['test_accuracy']:.2f}"],
          ['総発火数','0',int(r['total_fires'])]], [165,165,165])
        para('前進評価のみでの学習動作を最初に確認した。100区間ごとのoracle監査と発火が重なったのは4件のみで、改善率75%・平均損失変化−0.000242は参考値にとどまる。Stage Aの95%信頼区間を伴うaction品質合格判定を満たしたとは扱わない。カウンタの最大・平均・分布は、この初期runでは未記録。')
        continue
    if gid == 'E7':
        para('N=10000、b=32、θ=8、K=64、L=3000。各精度で初期状態から学習する。量子化する場所は、正規化画像入力とReLU後の隠れ層出力、すなわち各線形層への入力のみ。')
        table(['設定','表現'],[
         ['W3A32','FP32、そのまま。'],['W3A16','FP16に丸め、FP32へ復元して線形計算。'],
         ['W3A8','整数コード−127～127、255段階、FP32スケール。'],
         ['W3A4','整数コード−7～7、15段階、FP32スケール。'],
         ['W3A3','コード−1/0/+1の3値。3ビットではない。FP32スケール。']], [90,405])
        para('整数系は各サンプル・各層のscale=max(abs(x))/qmaxを使い、q=clip(round(x/scale),−qmax,qmax)、復元値scale×qをFP32で計算する。最近傍丸めで同距離は偶数。全ゼロ行のscaleは1。INT8容器を使い、sub-byte packingはしない。積和、中間線形出力、ReLU、最終logit、lossはFP32。ReLU後のA3は非負のため実際には0/+1だけを使う。')
    aggregate_table(gid)
    para(interpretations[gid])
    if gid == 'E7':
        diagnostics = read_csv(directory/'activation_diagnostics.csv')
        dr=[]
        for precision in ['a32','a16','a8','a4','a3']:
            fields=[]
            for layer in [0,1]:
                rr=[d for d in diagnostics if d['precision']==precision and d['stage']=='final' and int(d['layer'])==layer]
                fields.extend([f"{100*st.mean(float(d['zero_fraction']) for d in rr):.2f}",
                               f"{st.mean(float(d['relative_squared_error']) for d in rr):.6g}"])
            dr.append(['W3'+precision.upper(),*fields])
        table(['設定','入力ゼロ率 %','入力相対二乗誤差','隠れゼロ率 %','隠れ相対二乗誤差'],dr,[85,95,110,95,110])
        para('表は最終検証時の3seed平均。量子化直前と復元後の二乗誤差を量子化直前の信号エネルギーで割る。A32モデルとの誤差ではない。診断用FP64集計は読み取り専用で学習へ戻さない。初期・最終の全60診断行と整数コード分布は添付データに保存。A3の大きな情報損失は精度低下と整合するが、この診断だけで因果を完全に分離したとはしない。')
    images = {'E4':'threshold_comparison.png','E7':'activation_comparison.png','E8':'comparison.png'}
    if gid in images:
        figure(directory/images[gid],f'図：{title}。誤差棒は保存レポートで定義された3seedの標本標準偏差。',300)

section('5. 横断的解釈：何が確認できたか')
para('5.1 同一12,000区間での100kブロック比較', 'h2')
merged = [a for g in ['E9','E8'] for a in group_data[g][2] if int(a.get('steps',12000))==12000]
merged.sort(key=lambda a:int(a['block_size']))
table(['block','検証精度 %','テスト精度 %','平均発火数'],
      [[a['block_size'],pm(a,'val_accuracy',True),pm(a,'test_accuracy',True),pm(a,'total_fires')] for a in merged],
      [65,145,145,140])
para('全6条件のモデル・データ・区間数・閾値・K・batchは同じ。今回の平均は16が最高であり、32と近接する。64以上はブロックを大きくするほど精度が低かった。ただし局所行動正解率や相互作用残差を測定していないため、この傾向だけで低実効次元仮説や相互作用支配を証明しない。')
para('5.2 カウンタと低精度活性化', 'h2')
para('カウンタあり対照の改善は、時間積分と区間末更新を組み合わせた設計が、今回の即時更新対照より良かったことを示す。カウンタ方式の一般的優位や、STE-Adam/SGD・Bopに対する優位性を示さない。閾値の最適値はK、モデル、block、更新上限、リセット方式に依存し得る。')
para('W3A3で損失低下を確認したことは、三値重みと指定した三値活性化表現の下でも本実装が学習できるという実証である。一方、ReLU後は実効2値となり、補助スケール・積和・lossはFP32のままである。全演算が三値のシステムや、FP32相当精度を達成したとは表現しない。')
table(['v4の主張・課題','v5での位置づけ'],[
 ['直接三値・forward-only学習','MNISTの1k/10k/100kで実証。244記録で初期から改善。'],
 ['H1：C8十分性','今回の有限K・全リセットでは飽和なし。C16比較・長時間積分・初通過誤率は未検証。'],
 ['H2：低実効次元','学習成立は確認したがd_eff推定や識別可能性の直接実証はない。'],
 ['H3：相互作用制御','block依存を観測。R_int、gain別regret、集合更新の相互作用は未測定。'],
 ['H4：分散通信優位','単一CPUプロセスの各runで学習。共通seed分散学習・通信量比較は未実施。'],
 ['Stage A～F','MNIST予備検証。v4指定規模・信頼区間・baselineを満たす正式なStage通過宣言はしない。'],
 ['学習コスト','効率比較は今回の対象外。記録されたforward回数は条件一致確認用。'],
 ['新規性・収束','本実験は先行研究調査や収束証明ではない。v4の限定命題と研究上の位置づけを維持。'],
 ],[130,365])
para('次に必要な検証は、独立oracleによるgain別の行動品質、カウンタ待機と蓄積の要因分離、C8/C16・リセット・リークの比較、入力・隠れ層別の量子化アブレーション、深いモデル・別データへの拡張である。現時点で学習成立は確認できたが、すべての理論仮説が確認されたわけではない。')

section('付録A：全seedの精度・損失・発火数')
para('各runを省略せず掲載する。IDは添付normalized_records.jsonと対応する。検証精度とテスト精度は%、lossは交差エントロピー。条件はb/θ/L/A（block/閾値/区間数/活性化）で表す。counterなしはθを適用しない。')
for gid,name,title,params,steps in GROUPS:
    para(f'{gid} {title}', 'h2')
    rr=group_data[gid][1]
    data=[]
    for r in rr:
        cond=f"{r['block_size']}/{r['threshold']}/{r['steps']}/{r['precision'].upper()}"
        if r['method']=='no_counter': cond='8/—/3000/A32 Cなし'
        data.append([r['record_id'],r['seed'],cond,f"{r['initial_val_loss']:.4f}→{r['val_loss']:.4f}",
                     f"{100*r['initial_val_accuracy']:.2f}→{100*r['val_accuracy']:.2f}",
                     f"{100*r['test_accuracy']:.2f}",int(r['total_fires'])])
    table(['ID','seed','b/θ/L/A','検証loss 初→終','検証精度 初→終','test %','発火数'],data,[45,27,105,104,99,51,64])

section('付録B：全seedのカウンタ統計・層別更新数')
para('Cmin/Cmaxは符号付き最小・最大、平均Cと平均|C|は各runの測定済み辺分布、容量は±127。以下の記録済み全カウンタrunで飽和観測数は0。|C|最大はmin/maxからも復元できる。層更新数は入力側から第1層/第2層、未記録は「—」。全ヒストグラム、飽和更新回数、観測数、活性化コード分布などは添付CSV/JSONに保存する。')
for gid,name,title,params,steps in GROUPS:
    para(f'{gid} {title}', 'h2')
    data=[]
    for r in group_data[gid][1]:
        if 'counter_mean' not in r:
            data.append([r['record_id'],'未記録' if gid=='E1' else '存在なし','—','—','—','—','—'])
        else:
            layers='/'.join(str(int(float(r[k]))) for k in ['layer_0_fires','layer_1_fires'] if k in r) or '—'
            data.append([r['record_id'],f"{int(r['counter_min'])}/{int(r['counter_max'])}",
                         f"{float(r['counter_mean']):.4f}",f"{float(r['counter_abs_mean']):.4f}",
                         int(r['counter_abs_max']),int(r['counter_saturated_count']),layers])
    table(['ID','Cmin/Cmax','平均C','平均|C|','最大|C|','飽和件数','層別更新数'],data,[55,85,72,72,62,59,90])

section('付録C：再現方法・資料と照合')
para('全表の数値は保存済みCSV/JSONから生成し、243件の複数seed記録と1件の初期runを統合した。各群の平均・標本標準偏差をper_seed.csvから再計算し、aggregate.csvの精度・検証loss・発火数と1e−10未満の差で一致することを確認した。重複する子集計を二重計上しない。')
para('資料ZIPには本日の9実験群のCSV/JSON/README、ソーススナップショット、現行再現スクリプト、正規化済み244レコード、SHA256一覧を含む。PDFビューアの添付ファイルからtdt_v5_experiment_data.zipを取り出せる。同じ内容を外部ファイルTDT-v5_実験データ.zipにも保存した。重みチェックポイントと大きな全区間ログのローカル保存先は各manifest/READMEを参照する。')
para('共通環境：Python 3.12系、PyTorch 2.12.0a0+0291f960b6.nv26.04.48445190、CPU、各runはthreads=1。異なるPyTorch/CPU環境では完全一致しない場合がある。データはMNISTキャッシュを使用。実行設定の詳細は各config.jsonまたはmanifest.jsonが正本である。')
para('再現コマンド（リポジトリのルート、各保存先は新規ディレクトリ）', 'h2')
commands = [
 'python tdt_mnist/train.py --seed 0 --steps 3000 --train-size 10000 --val-size 1000 --eval-every 500 --output-dir runs/e1',
 'python tdt_mnist/sweep.py --blocks 1 8 32 --thresholds 4 8 16 32 --steps 3000 --output-dir runs/e2 --report-dir results/e2',
 'python tdt_mnist/compare_counters.py --output-dir runs/e3 --report-dir results/e3',
 'python tdt_mnist/sweep.py --blocks 8 --thresholds '+ ' '.join(map(str,range(1,33))) +' --output-dir runs/e4 --report-dir results/e4',
 'python tdt_mnist/sweep.py --pool-shape 9 10 --hidden-size 100 --expected-params 10000 --blocks 1 8 32 --thresholds 4 8 16 --output-dir runs/e5 --report-dir results/e5',
 'python tdt_mnist/sweep.py --pool-shape 9 10 --hidden-size 1000 --expected-params 100000 --blocks 1 8 32 --thresholds 4 8 16 --output-dir runs/e6 --report-dir results/e6',
 'python tdt_mnist/sweep_activations.py --output-dir runs/e7 --report-dir results/e7',
 'python tdt_mnist/sweep_lengths.py --blocks 64 128 256 --lengths 3000 6000 12000 --output-dir runs/e8 --report-dir results/e8',
 'python tdt_mnist/sweep_lengths.py --blocks 8 16 32 --lengths 12000 --output-dir runs/e9 --report-dir results/e9',
]
for (gid,*_),cmd in zip(GROUPS,commands):
    para(gid+': '+cmd, 'small')
para('E1は初期バージョンの記録であり、完全一致が必要なら保存された初期実装の設定・乱数方式を参照する。上のコマンドは同じ実験条件の再実行用で、バージョンをまたぐビット単位再現を保証しない。現行の複数seed実験の詳細な引数と当時のソースハッシュはmanifestに保存されている。')
para('v4原文SHA256：'+digest(ORIGINAL), 'small')
para('本PDFの生成ソース：tdt_mnist/paper_v5/build_pdf.py。本文テキスト、全run正規化データ、照合記録も同ディレクトリに保存。参考文献[1]～[13]は付録Dのv4原文に収録し、今回新たな先行研究レビューは加えていない。')

section('付録D：TDT-v4原文・全10ページ')
para('以下は改訂の基礎となったv4原文の忠実な複製であり、原版のページ番号・数式・参考文献を保持する。v4の「実験結果を報告するものではない」という文書位置づけはv4執筆時点の記述である。v5では本編のMNIST実験結果と限定された結論が追加されている。')
para('この付録はv5の実験結果を上書きするものではない。v4の一般アルゴリズムと今回の同期・全リセット実装との相違は第1節を参照する。')


class Doc(SimpleDocTemplate):
    def afterFlowable(self, flowable):
        if hasattr(flowable, 'bookmark'):
            self.canv.bookmarkPage(flowable.bookmark)
            self.canv.addOutlineEntry(flowable.getPlainText(), flowable.bookmark, level=0)


def page_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('CJK',7)
    canvas.setFillColor(colors.HexColor('#536779'))
    if doc.page>1:
        canvas.drawString(48,815,'TDT-v5  |  MNIST実験統合版  |  2026-09-05')
    canvas.drawRightString(547,26,str(doc.page))
    canvas.restoreState()


intermediate = OUT/'v5-main.pdf'
Doc(str(intermediate), pagesize=A4, leftMargin=48, rightMargin=48, topMargin=47, bottomMargin=45,
    title='TDT-v5：離散状態遷移に基づく前進評価型3値学習フレームワーク',
    author='TDT experimental research').build(story,onFirstPage=page_footer,onLaterPages=page_footer)
(OUT/'TDT-v5_本文.txt').write_text('\n\n'.join(text_log))
(OUT/'normalized_records.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
source_hashes[str(ORIGINAL.relative_to(ROOT))] = digest(ORIGINAL)
(OUT/'source_hashes.json').write_text(json.dumps(source_hashes,ensure_ascii=False,indent=2)+'\n')
audit = dict(experiment_groups=9, condition_rows=82, run_records=244, includes_repeated_conditions=True,
             initial_loss_improved=244, initial_accuracy_improved=244, aggregate_recalculation_passed=True,
             missing_counter_distribution=['E1'], non_applicable_counter_distribution=['E3 no_counter'],
             date='2026-09-05')
(OUT/'data_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
zip_path=ROOT/'TDT-v5_実験データ.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as archive:
    for path in source_hashes:
        if Path(path).suffix != '.pdf':
            archive.write(ROOT/path,path)
    for name in ['normalized_records.json','source_hashes.json','data_audit.json','build_pdf.py','TDT-v5_本文.txt']:
        archive.write(OUT/name,'paper_v5/'+name)
    for f in (ROOT/'tdt_mnist').glob('*.py'):
        archive.write(f,'current_code/'+f.name)
pdf=fitz.open(intermediate)
main_pages=len(pdf)
toc=pdf.get_toc()
original=fitz.open(ORIGINAL)
pdf.insert_pdf(original)
toc.append([1,'付録D：v4原文の開始',main_pages+1])
pdf.set_toc(toc)
pdf.embfile_add('tdt_v5_experiment_data.zip',zip_path.read_bytes(),filename=zip_path.name,
                desc='All nine completed 2026-09-05 experiment groups; CSV, JSON, source records, audit.')
pdf.set_metadata({'title':'TDT-v5：離散状態遷移学習理論 — MNIST実験統合版',
                  'subject':'2026-09-05: 9 experiment groups, 244 run records',
                  'keywords':'TDT, MNIST, ternary, forward-only, counter, FP32',
                  'creator':'TDT-v5 build_pdf.py'})
pdf.save(DEST,garbage=4,deflate=True)
print(json.dumps({'pdf':str(DEST),'main_pages':main_pages,'original_pages':len(original),
                  'total_pages':len(pdf),'bytes':DEST.stat().st_size,'zip_bytes':zip_path.stat().st_size},indent=2))
