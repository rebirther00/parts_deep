"""Rebuild data-only figures and an editable Word draft; no model results invented."""
import json
import re
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from door8.core import ROOT, dump, outpath, read_json, setup_runtime, snapshot, write_csv
setup_runtime()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def save(fig,name):
    p=outpath('figures');p.mkdir(exist_ok=True)
    for ext in ('pdf','svg','png'):fig.savefig(p/(name+'.'+ext),bbox_inches='tight',dpi=220)
    plt.close(fig)


def flow(name,labels,caption):
    fig,ax=plt.subplots(figsize=(11,2.8));ax.set_xlim(0,11);ax.set_ylim(0,2.5);ax.axis('off')
    width=10.5/len(labels)
    for i,text in enumerate(labels):
        x=.25+i*width
        box=FancyBboxPatch((x,.85),width-.3,1.2,boxstyle='round,pad=0.04',edgecolor='#32576F',facecolor='#EAF1F5',linewidth=1.2)
        ax.add_patch(box);ax.text(x+(width-.3)/2,1.45,text,ha='center',va='center',fontsize=10)
        if i<len(labels)-1:ax.annotate('',xy=(x+width-.07,1.45),xytext=(x+width-.25,1.45),arrowprops={'arrowstyle':'->','color':'#32576F'})
    ax.text(5.5,.35,caption,ha='center',va='center',fontsize=10,color='#333333');save(fig,name)


def figures():
    _,s,_,_=snapshot();classes=list(s['classes']);counts=[s['classes'][c]['sessions'] for c in classes]
    labels=[c.replace('_door','').replace('E30_E38','E30/E38').replace('_',' ') for c in classes]
    fig,ax=plt.subplots(figsize=(8,4.8));bars=ax.barh(labels[::-1],counts[::-1],color='#376B84')
    ax.bar_label(bars,padding=4);ax.set_xlim(0,80);ax.set_xlabel('Capture sessions');ax.set_title('Retrospective field dataset: 361 sessions, 8 classes')
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.text(.02,.01,'E23 excluded; physical-instance identity is not established.',fontsize=9,color='#555555');save(fig,'class_sessions')
    flow('data_flow',['379 recorded\nsessions','Exclude 18\nsessions','361 eligible\nsessions','7,104 sampled\nRGB-depth pairs','Session-group\n5-fold CV'],
         'Exclusions: 7 previously invalid + 11 human-confirmed class-correction sessions (including 6 E23).')
    flow('method_pipeline',['RGB image','Hole heatmaps\n4 + 1 + 1 points','Geometry\nvalidity checks','Pixel / depth\ndistance','CAD match\nor abstention'],
         'Detector: manual LAB labels only. Scale calibration: field TRAIN sessions only. TEST: evaluation only.')
    write_csv('reports/class_counts.csv',[{'class':c,**s['classes'][c]} for c in classes])
    write_csv('reports/split_counts.csv',[{'fold':f,'role':role,'sessions':v['sessions'],'frames':v['frames']}
              for f,parts in s['split_counts'].items() for role,v in parts.items()])
    comparison=ROOT/'reports/comparison_f0_s42/comparison.json'
    if comparison.exists():
        import numpy as np
        rows=read_json(comparison)['table']
        names={'rgb':'RGB','rgbe':'RGB + edge','holes_pixel':'Holes + pixel',
               'holes_depth':'Holes + depth','mobile_sam_dimensions':'MobileSAM + shape'}
        labels=[names[r['method']] for r in rows];x=np.arange(len(rows));fig,ax=plt.subplots(figsize=(9,4.5))
        a=ax.bar(x-.18,[r['frame_accuracy_all']*100 for r in rows],.36,label='Frame accuracy (all inputs)',color='#376B84')
        b=ax.bar(x+.18,[r['frame_coverage']*100 for r in rows],.36,label='Frame coverage',color='#A9BCC6')
        ax.bar_label(a,fmt='%.1f',padding=3,fontsize=8);ax.bar_label(b,fmt='%.1f',padding=3,fontsize=8)
        ax.set_xticks(x,labels);ax.set_ylim(0,113);ax.set_ylabel('Percent');ax.spines[['top','right']].set_visible(False)
        ax.set_title('First formal test fold only: 74 sessions / 1,478 frames / seed 42')
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.10),ncol=2,frameon=False,fontsize=9)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);save(fig,'first_fold_accuracy')


def plain(s):
    s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'\1 (\2)',s)
    return s.replace('**','').replace('`','')


def docx():
    from docx import Document
    from docx.shared import Inches,Pt,Cm
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    doc=Document();sec=doc.sections[0];sec.top_margin=Cm(2);sec.bottom_margin=Cm(2);sec.left_margin=Cm(2.2);sec.right_margin=Cm(2.2)
    for name in ['Normal','Title','Heading 1','Heading 2','Heading 3']:
        style=doc.styles[name];style.font.name='Malgun Gothic';style._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'맑은 고딕')
    doc.styles['Normal'].font.size=Pt(10);doc.styles['Normal'].paragraph_format.line_spacing=1.25
    sec.header.paragraphs[0].text='연구용 초안 v1 | 정식 비교 결과 및 전향적 평가 미완료'
    p=sec.footer.paragraphs[0];p.alignment=2
    f=OxmlElement('w:fldSimple');f.set(qn('w:instr'),'PAGE');p._p.append(f)
    lines=(ROOT/'manuscript/논문초안_v1.md').read_text().splitlines();i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[plain(x.strip()) for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[:\- ]+',c) for c in cells):rows.append(cells)
                i+=1
            table=doc.add_table(rows=1,cols=len(rows[0]));table.style='Light Shading Accent 1'
            for j,c in enumerate(rows[0]):table.rows[0].cells[j].text=c
            for row in rows[1:]:
                cells=table.add_row().cells
                for j,c in enumerate(row):cells[j].text=c
            continue
        if line.startswith('# '):doc.add_heading(plain(line[2:]),0)
        elif line.startswith('## '):doc.add_heading(plain(line[3:]),1)
        elif line.startswith('### '):doc.add_heading(plain(line[4:]),2)
        elif line.startswith('> '):doc.add_paragraph(plain(line[2:]),style='Intense Quote')
        else:doc.add_paragraph(plain(line))
        i+=1
    doc.add_page_break();doc.add_heading('데이터 구성 및 방법 도식',1)
    figures=[('class_sessions','클래스별 세션 수'),('data_flow','데이터 포함 및 평가 구성'),('method_pipeline','홀 기반 품종 식별 흐름')]
    if (ROOT/'figures/first_fold_accuracy.png').exists():figures.append(('first_fold_accuracy','첫 폴드 정식 비교: 전체 반복 성능이 아님'))
    for name,caption in figures:
        doc.add_picture(str(ROOT/'figures'/f'{name}.png'),width=Cm(16));doc.add_paragraph(caption)
    doc.save(outpath('manuscript/논문초안_v1.docx'))


if __name__=='__main__':figures();docx();print('Wrote data figures and editable manuscript DOCX.')
