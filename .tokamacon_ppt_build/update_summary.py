from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from lxml import etree as E
import re,json
base=Path('/Volumes/T7/projects/PGOFP')
src=base/'output/ppt/TokamaCon_背景补充版_20页.pptx'
new=Path('/Users/wuaijia/Library/Application Support/Codex/artifact-sessions/exports/latest/d905360f70918a84c6dfb356c2c4a1434afcb3d86f2386bf777d1884645cefe1/concrete_summary.pptx')
out=base/'output/ppt/TokamaCon_背景补充与总结完善版_19页.pptx'
with ZipFile(src) as z: orig={n:z.read(n) for n in z.namelist()}
with ZipFile(new) as z: add={n:z.read(n) for n in z.namelist()}
f=orig.copy()
ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
def dump(x):return E.tostring(x,encoding='UTF-8',xml_declaration=True,standalone=True)
for name in list(f):
 if re.fullmatch(r'ppt/slides/slide\d+.xml',name):f[name]=re.sub(rb'(\d\d) / 20',rb'\1 / 19',f[name])
f['ppt/slides/slide2.xml']=f['ppt/slides/slide2.xml'].replace('13–20'.encode(),'13–19'.encode())
f['ppt/slides/slide11.xml']=f['ppt/slides/slide11.xml'].replace('18–20   未见平衡态测试、复现边界与研究讨论'.encode(),'18–19   未见平衡态测试、总结与后续研究'.encode())
layout=next(x.get('Target') for x in E.fromstring(orig['ppt/slides/_rels/slide4.xml.rels']) if x.get('Type').endswith('/slideLayout'))
f['ppt/slides/slide18.xml']=add['ppt/slides/slide1.xml']
rr=E.fromstring(add['ppt/slides/_rels/slide1.xml.rels'])
for r in rr:
 if r.get('Type').endswith('/slideLayout'):r.set('Target',layout)
 elif r.get('Type').endswith('/notesSlide'):r.set('Target','../notesSlides/notesSlide18.xml')
 else:raise ValueError(r.get('Type'))
f['ppt/slides/_rels/slide18.xml.rels']=dump(rr)
f['ppt/notesSlides/notesSlide18.xml']=add['ppt/notesSlides/notesSlide1.xml']
rr=E.fromstring(add['ppt/notesSlides/_rels/notesSlide1.xml.rels'])
for r in rr:
 if r.get('Type').endswith('/slide'):r.set('Target','../slides/slide18.xml')
 elif r.get('Type').endswith('/notesMaster'):r.set('Target','../notesMasters/notesMaster1.xml')
 else:raise ValueError(r.get('Type'))
f['ppt/notesSlides/_rels/notesSlide18.xml.rels']=dump(rr)
rels=E.fromstring(f['ppt/_rels/presentation.xml.rels'])
rid=next(x.get('Id') for x in rels if x.get('Target')=='slides/slide17.xml')
for x in list(rels):
 if x.get('Id')==rid:rels.remove(x)
f['ppt/_rels/presentation.xml.rels']=dump(rels)
p=E.fromstring(f['ppt/presentation.xml']);lst=p.find('p:sldIdLst',ns)
for x in list(lst):
 if x.get('{'+ns['r']+'}id')==rid:lst.remove(x)
assert len(lst)==19
f['ppt/presentation.xml']=dump(p)
removed=['ppt/slides/slide17.xml','ppt/slides/_rels/slide17.xml.rels','ppt/notesSlides/notesSlide17.xml','ppt/notesSlides/_rels/notesSlide17.xml.rels']
for name in removed:f.pop(name,None)
ct=E.fromstring(f['[Content_Types].xml'])
for x in list(ct):
 if x.get('PartName','').lstrip('/') in removed:ct.remove(x)
f['[Content_Types].xml']=dump(ct)
a=E.fromstring(f['docProps/app.xml'])
for x in a.iter():
 if E.QName(x).localname in ['Slides','Notes']:x.text='19'
f['docProps/app.xml']=dump(a)
count=0
for name,b in f.items():
 if re.fullmatch(r'ppt/slides/slide\d+.xml',name):
  eq=lambda data:[E.tostring(x,method='c14n') for x in E.fromstring(data).findall('.//m:oMath',ns)]
  assert eq(b)==eq(orig[name]),name
  count+=len(eq(b))
assert count==12
assert sum(bool(re.fullmatch(r'ppt/charts/chart\d+.xml',n)) for n in f)==6
with ZipFile(out,'w',ZIP_DEFLATED) as z:
 for n,b in f.items():z.writestr(n,b)
print(json.dumps({'output':str(out),'slides':19,'preserved_equations':count,'charts':6},ensure_ascii=False))
