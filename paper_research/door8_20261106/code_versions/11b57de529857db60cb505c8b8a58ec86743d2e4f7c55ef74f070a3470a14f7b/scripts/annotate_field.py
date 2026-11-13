"""Local human annotation UI. Reads original images; writes only research labels.

Left click: place selected point. Wheel: zoom. Right drag: pan. V: invisible.
S: save and next. Numbers 1..6: select point. F: fit image. No automatic labels.
"""
import argparse,datetime,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from door8.core import ROOT,dump,now,outpath,read_json,read_jsonl,sha256
from door8.annotations import KEYS,validate_label


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--web',action='store_true',help='DISPLAY 없이 로컬 브라우저에서 주석')
    parser.add_argument('--port',type=int,default=8765,help='브라우저 주석 포트 (기본 8765)')
    args=parser.parse_args()
    rows=read_jsonl(ROOT/'data/field40_v2_queue.jsonl');labeldir=outpath('data/annotations/field40_v2')
    if args.list:
        print(f'Human labels: {sum((labeldir/(r["annotation_id"]+".json")).exists() for r in rows)}/{len(rows)}');return
    if args.web:
        from annotate_field_web import serve
        serve(args.port);return
    import tkinter as tk
    from tkinter import ttk,messagebox
    from PIL import Image,ImageTk
    try:win=tk.Tk()
    except tk.TclError as exc:
        if 'display' not in str(exc).lower():raise
        print('GUI 화면(DISPLAY)에 연결할 수 없어 브라우저 주석 모드로 전환합니다.',flush=True)
        from annotate_field_web import serve
        serve(args.port);return
    win.title('Door8 — 수동 현장 홀 좌표 (원본 읽기 전용)');win.geometry('1400x980')
    state={'index':0,'scale':1.,'ox':0.,'oy':0.,'image':None,'photo':None,'row':None,'pan':None}
    author=tk.StringVar();selected=tk.StringVar(value=KEYS[0]);status=tk.StringVar();title=tk.StringVar()
    top=ttk.Frame(win);top.pack(fill='x',padx=8,pady=6)
    ttk.Label(top,text='작업자:').pack(side='left');ttk.Entry(top,textvariable=author,width=16).pack(side='left')
    ttk.Combobox(top,textvariable=selected,values=KEYS,state='readonly',width=20).pack(side='left',padx=8)
    ttk.Label(win,textvariable=title).pack(fill='x')
    ttk.Label(win,text='좌클릭: 좌표 | 휠: 확대 | 우클릭 드래그: 이동 | V: 가려짐 | S: 저장 후 다음 | 1~6: 점 선택 | F: 전체 보기').pack(fill='x')
    canvas=tk.Canvas(win,bg='#222222',highlightthickness=0);canvas.pack(fill='both',expand=True)
    ttk.Label(win,textvariable=status).pack(fill='x',padx=8,pady=6)

    def redraw(_=None):
        im=state['image']
        if im is None:return
        canvas.delete('all');sc=state['scale'];ox,oy=state['ox'],state['oy'];cw,ch=canvas.winfo_width(),canvas.winfo_height()
        left=max(0,int(-ox/sc));top=max(0,int(-oy/sc));right=min(im.width,int((cw-ox)/sc)+1);bottom=min(im.height,int((ch-oy)/sc)+1)
        if right>left and bottom>top:
            crop=im.crop((left,top,right,bottom));crop=crop.resize((max(1,round(crop.width*sc)),max(1,round(crop.height*sc))))
            state['photo']=ImageTk.PhotoImage(crop);canvas.create_image(ox+left*sc,oy+top*sc,anchor='nw',image=state['photo'])
        for i,k in enumerate(KEYS):
            p=state['row']['points'][k]
            if p is not None:
                x,y=p[0]*sc+ox,p[1]*sc+oy;color='#ffcf33' if k==selected.get() else '#35ff7b'
                canvas.create_oval(x-4,y-4,x+4,y+4,outline=color,width=2)
                canvas.create_text(x+8,y-8,text=f'{i+1}: {k}',anchor='sw',fill=color)
        status.set('  |  '.join(f'{i+1} {"가려짐" if not state["row"]["visible"][k] else "완료" if state["row"]["points"][k] is not None else "미표시"}' for i,k in enumerate(KEYS)))

    def fit(_=None):
        im=state['image']
        if im is None:return
        cw,ch=canvas.winfo_width(),canvas.winfo_height();state['scale']=min(cw/im.width,ch/im.height)*.97
        state['ox']=(cw-im.width*state['scale'])/2;state['oy']=(ch-im.height*state['scale'])/2;redraw()

    def load(index):
        state['index']=index%len(rows);template=rows[state['index']];p=labeldir/(template['annotation_id']+'.json')
        row=read_json(p) if p.exists() else __import__('copy').deepcopy(template)
        if sha256(row['rgb'])!=row['rgb_sha256']:raise ValueError('Source image hash mismatch')
        state['row']=row
        with Image.open(row['rgb']) as im:state['image']=im.convert('RGB')
        selected.set(KEYS[0]);title.set(f'{state["index"]+1}/40  {row["class_name"]}  {row["session_dir"]}  CV group {row["cv_group"]}');fit()

    def advance_point():selected.set(KEYS[(KEYS.index(selected.get())+1)%len(KEYS)]);redraw()
    def click(event):
        x=(event.x-state['ox'])/state['scale'];y=(event.y-state['oy'])/state['scale'];im=state['image']
        if 0<=x<im.width and 0<=y<im.height:
            k=selected.get();state['row']['points'][k]=[round(x,3),round(y,3)];state['row']['visible'][k]=True;advance_point()
    def invisible(_=None):
        k=selected.get();state['row']['points'][k]=None;state['row']['visible'][k]=False;advance_point()
    def zoom(event):
        factor=1.25 if (getattr(event,'num',None)==4 or getattr(event,'delta',0)>0) else .8
        old=state['scale'];new=min(12.,max(.1,old*factor));factor=new/old
        state['ox']=event.x-(event.x-state['ox'])*factor;state['oy']=event.y-(event.y-state['oy'])*factor;state['scale']=new;redraw()
    def pan_start(event):state['pan']=(event.x,event.y,state['ox'],state['oy'])
    def pan_move(event):
        if state['pan']:
            x,y,ox,oy=state['pan'];state['ox']=ox+event.x-x;state['oy']=oy+event.y-y;redraw()
    def save(_=None):
        row=state['row']
        try:validate_label(row,author.get())
        except ValueError as e:messagebox.showerror('저장 전 확인',str(e));return
        p=outpath(labeldir/(row['annotation_id']+'.json'))
        if p.exists():
            history=outpath(labeldir/'history'/(row['annotation_id']+'_'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f')+'.json'))
            dump(history,read_json(p))
        row.update(annotation_status='human_verified',source='human_manual_field',annotator=author.get().strip(),annotation_time=now())
        dump(p,row);load(state['index']+1)
    ttk.Button(top,text='전체 보기',command=fit).pack(side='left',padx=4)
    ttk.Button(top,text='가려짐(V)',command=invisible).pack(side='left',padx=4)
    ttk.Button(top,text='저장 후 다음(S)',command=save).pack(side='left',padx=4)
    ttk.Button(top,text='이전 (현재 미저장 변경 버림)',command=lambda:load(state['index']-1)).pack(side='left',padx=4)
    canvas.bind('<Button-1>',click);canvas.bind('<Button-3>',pan_start);canvas.bind('<B3-Motion>',pan_move)
    canvas.bind('<MouseWheel>',zoom);canvas.bind('<Button-4>',zoom);canvas.bind('<Button-5>',zoom);canvas.bind('<Configure>',redraw)
    # Bind shortcuts to the canvas so typing an annotator name cannot place labels or save.
    canvas.bind('<Enter>',lambda e:canvas.focus_set())
    for key in ['v','V']:canvas.bind(key,invisible)
    for key in ['s','S']:canvas.bind(key,save)
    for key in ['f','F']:canvas.bind(key,fit)
    for i in range(6):canvas.bind(str(i+1),lambda e,i=i:(selected.set(KEYS[i]),redraw()))
    win.after(100,lambda:load(0));win.mainloop()


if __name__=='__main__':main()
