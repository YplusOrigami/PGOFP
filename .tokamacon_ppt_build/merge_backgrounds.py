from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from lxml import etree as E
from copy import deepcopy
import re,json,hashlib
SRC=Path('/Users/wuaijia/Downloads/TokamaCon_结构增强版_18页_30分钟.pptx')
BG=Path('/Users/wuaijia/Library/Application Support/Codex/artifact-sessions/exports/latest/95c64837322fd9f4268aa49371a9c569ed2789979a15ef67c96643ece7f93521/background_pages.pptx')
OUT=Path('/Volumes/T7/projects/PGOFP/output/ppt/TokamaCon_背景补充版_20页.pptx')
N={'p':'http://schemas.openxmlformats.org/presentationml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships','pkg':'http://schemas.openxmlformats.org/package/2006/relationships','ct':'http://schemas.openxmlformats.org/package/2006/content-types','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
def dump(r):return E.tostring(r,encoding='UTF-8',xml_declaration=True,standalone=True)
with ZipFile(SRC) as z:original={n:z.read(n) for n in z.namelist()}
with ZipFile(BG) as z:bg={n:z.read(n) for n in z.namelist()}
files=original.copy()
mapping={i:i+(i>=5)+(i>=8) for i in range(1,19)}
for i,new in mapping.items():
 name=f'ppt/slides/slide{i}.xml'
 # Surgical string changes preserve the user's complete equation XML unchanged.
 old=f'{i:02d} / 18'.encode();newtext=f'{new:02d} / 20'.encode()
 assert old in files[name],(i,'missing footer')
 files[name]=files[name].replace(old,newtext)
replacements={
 2:{'04–10':'04–12','11–18':'13–20'},
 4:{'05–07   IsoFlux：线性动力学、稀疏诊断与跟踪奖励':'05–08   IsoFlux：背景、线性动力学、稀疏诊断与奖励',
    '08–10   Profile：输运耦合、NTM 抑制与储能目标':'09–12   Profile：背景、输运耦合、NTM 抑制与储能目标'},
 11:{'12–15':'14–17','16–18':'18–20'},
}
for i,changes in replacements.items():
 name=f'ppt/slides/slide{i}.xml'
 for a,b in changes.items():
  assert a.encode() in files[name],(i,a)
  files[name]=files[name].replace(a.encode(),b.encode())
# Use the source deck's own blank slide layout and notes master.
rels=E.fromstring(files['ppt/slides/_rels/slide4.xml.rels'])
layout=next(x.get('Target') for x in rels if x.get('Type').endswith('/slideLayout'))
for source_idx,target_idx in [(1,19),(2,20)]:
 files[f'ppt/slides/slide{target_idx}.xml']=bg[f'ppt/slides/slide{source_idx}.xml']
 rr=E.fromstring(bg[f'ppt/slides/_rels/slide{source_idx}.xml.rels'])
 for r in rr:
  if r.get('Type').endswith('/slideLayout'):r.set('Target',layout)
  elif r.get('Type').endswith('/notesSlide'):r.set('Target',f'../notesSlides/notesSlide{target_idx}.xml')
  else:raise ValueError('Unexpected background dependency: '+r.get('Type'))
 files[f'ppt/slides/_rels/slide{target_idx}.xml.rels']=dump(rr)
 files[f'ppt/notesSlides/notesSlide{target_idx}.xml']=bg[f'ppt/notesSlides/notesSlide{source_idx}.xml']
 rr=E.fromstring(bg[f'ppt/notesSlides/_rels/notesSlide{source_idx}.xml.rels'])
 for r in rr:
  if r.get('Type').endswith('/slide'):r.set('Target',f'../slides/slide{target_idx}.xml')
  elif r.get('Type').endswith('/notesMaster'):r.set('Target','../notesMasters/notesMaster1.xml')
  else:raise ValueError(r.get('Type'))
 files[f'ppt/notesSlides/_rels/notesSlide{target_idx}.xml.rels']=dump(rr)
root=E.fromstring(files['ppt/presentation.xml']);sl=root.find('p:sldIdLst',N)
old=list(sl);assert len(old)==18
pr=E.fromstring(files['ppt/_rels/presentation.xml.rels'])
newids=[]
for k,part in enumerate([19,20],1):
 rid=f'rIdBackground{k}';r=E.SubElement(pr,'{'+N['pkg']+'}Relationship')
 r.set('Id',rid);r.set('Type',N['r']+'/slide');r.set('Target',f'slides/slide{part}.xml')
 sid=E.Element('{'+N['p']+'}sldId',id=str(max(int(x.get('id')) for x in old)+k))
 sid.set('{'+N['r']+'}id',rid);newids.append(sid)
ordered=old[:4]+[newids[0]]+old[4:7]+[newids[1]]+old[7:]
sl[:]=ordered
files['ppt/presentation.xml']=dump(root);files['ppt/_rels/presentation.xml.rels']=dump(pr)
ct=E.fromstring(files['[Content_Types].xml'])
for j in [19,20]:
 for part,typ in [(f'/ppt/slides/slide{j}.xml','application/vnd.openxmlformats-officedocument.presentationml.slide+xml'),(f'/ppt/notesSlides/notesSlide{j}.xml','application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml')]:
  x=E.SubElement(ct,'{'+N['ct']+'}Override');x.set('PartName',part);x.set('ContentType',typ)
files['[Content_Types].xml']=dump(ct)
if 'docProps/app.xml' in files:
 root=E.fromstring(files['docProps/app.xml'])
 for e in root.iter():
  if E.QName(e).localname in ['Slides','Notes']:e.text='20'
 files['docProps/app.xml']=dump(root)
# Preserve every native equation subtree, including the user's manual fixes.
counts=[]
for i in range(1,19):
 part=f'ppt/slides/slide{i}.xml'
 a=E.fromstring(original[part]);b=E.fromstring(files[part])
 am=a.findall('.//m:oMath',N);bm=b.findall('.//m:oMath',N)
 assert len(am)==len(bm)
 assert [E.tostring(x,method='c14n') for x in am]==[E.tostring(x,method='c14n') for x in bm],i
 counts.append(len(am))
# All untouched source package parts remain byte-identical.
changed=[k for k in original if files[k]!=original[k]]
assert all(k.startswith('ppt/slides/slide') or k in ['ppt/presentation.xml','ppt/_rels/presentation.xml.rels','[Content_Types].xml','docProps/app.xml'] for k in changed)
assert not OUT.exists()
with ZipFile(OUT,'w',ZIP_DEFLATED) as z:
 for n,b in files.items():z.writestr(n,b)
print(json.dumps({'output':str(OUT),'slides':20,'background_positions':[5,9],'preserved_native_equations':sum(counts),'all_original_notes_unchanged':True,'bytes':OUT.stat().st_size},ensure_ascii=False))
