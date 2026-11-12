from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from copy import deepcopy
from lxml import etree as E
import json

ROOT=Path('/Volumes/T7/projects/PGOFP/.tokamacon_math_build')
SRC=Path('/Volumes/T7/projects/PGOFP/output/ppt/TokamaCon_结构增强版_18页_30分钟.pptx')
OUT=ROOT/'candidate_native_math.pptx'
NS={
 'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
 'p':'http://schemas.openxmlformats.org/presentationml/2006/main',
 'm':'http://schemas.openxmlformats.org/officeDocument/2006/math',
 'a14':'http://schemas.microsoft.com/office/drawing/2010/main',
 'mc':'http://schemas.openxmlformats.org/markup-compatibility/2006',
 'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
}
def tag(prefix,name):return '{'+NS[prefix]+'}'+name
def node(prefix,name,**attrs):return E.Element(tag(prefix,name),attrs)
def frame(sp,box):
 t=sp.find('p:spPr/a:xfrm',NS)
 off=t.find('a:off',NS); ext=t.find('a:ext',NS)
 off.set('x',str(round(box[0]*9525)));off.set('y',str(round(box[1]*9525)))
 ext.set('cx',str(round(box[2]*9525)));ext.set('cy',str(round(box[3]*9525)))
def props(size,color):
 p=node('a','rPr',lang='en-US',sz=str(round(size*100)),dirty='0')
 f=node('a','solidFill');f.append(node('a','srgbClr',val=color));p.append(f)
 p.append(node('a','latin',typeface='Cambria Math'))
 p.append(node('a','ea',typeface='PingFang SC'))
 p.append(node('a','cs',typeface='Cambria Math'))
 return p
def math_sp(sp,omath,box,size,tex):
 sp=deepcopy(sp);frame(sp,box)
 cn=sp.find('p:nvSpPr/p:cNvPr',NS)
 cn.set('name','Native Office Math '+cn.get('id'))
 cn.set('descr','Editable Office Math (OMML). LaTeX: '+tex)
 color=sp.xpath('.//a:srgbClr/@val',namespaces=NS)[0]
 body=sp.find('p:txBody',NS);body.clear()
 bp=node('a','bodyPr',wrap='none',lIns='0',rIns='0',tIns='0',bIns='0',anchor='ctr')
 bp.append(node('a','noAutofit'));body.append(bp);body.append(node('a','lstStyle'))
 para=node('a','p');pp=node('a','pPr',algn='l')
 dr=props(size,color);dr.tag=tag('a','defRPr');pp.append(dr);para.append(pp)
 mm=deepcopy(omath)
 # DrawingML equations must contain DrawingML properties, not WordprocessingML.
 for el in list(mm.iter()):
  if E.QName(el).namespace==NS['w']:
   el.getparent().remove(el)
 for run in mm.findall('.//m:r',NS):
  rp=props(size,color)
  mrp=run.find('m:rPr',NS)
  run.insert(1 if mrp is not None else 0,rp)
 for pr in list(mm.iter()):
  if E.QName(pr).namespace==NS['m'] and E.QName(pr).localname in {
   'accPr','dPr','fPr','sSubPr','sSupPr','sSubSupPr','naryPr','funcPr','radPr'
  }:
   cp=pr.find('m:ctrlPr',NS)
   if cp is None:cp=E.SubElement(pr,tag('m','ctrlPr'))
   cp.append(props(size,color))
 mathwrap=E.SubElement(para,tag('a14','m'),nsmap={'a14':NS['a14'],'m':NS['m']})
 mp=E.SubElement(mathwrap,tag('m','oMathPara'))
 mppr=E.SubElement(mp,tag('m','oMathParaPr'))
 jc=E.SubElement(mppr,tag('m','jc'));jc.set(tag('m','val'),'left')
 mp.append(mm)
 end=props(size,color);end.tag=tag('a','endParaRPr');para.append(end)
 body.append(para)
 E.cleanup_namespaces(sp)
 return sp

equations=json.loads((ROOT/'equations.json').read_text())
with ZipFile(ROOT/'equations.docx') as z:
 doc=E.fromstring(z.read('word/document.xml'))
 omaths=doc.findall('.//m:oMath',NS)
assert len(equations)==len(omaths)==13
with ZipFile(SRC) as zin:
 files={n:zin.read(n) for n in zin.namelist()}
report=[]
for page in range(5,11):
 part=f'ppt/slides/slide{page}.xml';root=E.fromstring(files[part])
 shapes={int(s.find('p:nvSpPr/p:cNvPr',NS).get('id')):s for s in root.findall('.//p:sp',NS)}
 for (entry,omath) in zip(equations,omaths):
  pg,sid,tex,box,size=entry
  if pg!=page:continue
  old=shapes[sid];native=math_sp(old,omath,box,size,tex)
  fallback=deepcopy(old);frame(fallback,box)
  fallback.find('p:nvSpPr/p:cNvPr',NS).set('id',str(sid+1000))
  ac=E.Element(tag('mc','AlternateContent'),nsmap={'mc':NS['mc']})
  choice=E.SubElement(ac,tag('mc','Choice'),Requires='a14',nsmap={'a14':NS['a14']})
  choice.append(native);fb=E.SubElement(ac,tag('mc','Fallback'));fb.append(fallback)
  parent=old.getparent();parent.replace(old,ac)
  report.append({'slide':page,'shape':sid,'latex':tex,'fontPt':size,'frame':box})
 # Give stacked fractions and summation limits sufficient vertical room.
 adjustments={
  7:{10:[675,389,550,59],12:[675,244,550,36]},
  8:{6:[58,229,1160,37]},
  9:{6:[58,235,1160,43],7:[58,290,1160,43]},
 }
 for sid,box in adjustments.get(page,{}).items():frame(shapes[sid],box)
 if page==7:
  for rp in shapes[10].findall('.//a:rPr',NS)+shapes[10].findall('.//a:defRPr',NS):rp.set('sz','1575')
  # Retain the original plot with a small reduction to clear the new formula rows.
  pic=root.find('.//p:pic',NS)
  xfrm=pic.find('p:spPr/a:xfrm',NS)
  xfrm.find('a:off',NS).set('y',str(460*9525));xfrm.find('a:ext',NS).set('cy',str(194*9525))
 files[part]=E.tostring(root,encoding='utf-8',xml_declaration=True,standalone=True)
with ZipFile(OUT,'w',ZIP_DEFLATED) as zout:
 for n,b in files.items():zout.writestr(n,b)
(ROOT/'math_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
# Structural invariants: all 13 are real OMML, with no Word namespace children.
with ZipFile(OUT) as z:
 total=0
 for page in range(5,11):
  r=E.fromstring(z.read(f'ppt/slides/slide{page}.xml'))
  for m in r.findall('.//a14:m',NS):
   assert len(m.findall('.//m:oMath',NS))==1
   assert not any(E.QName(e).namespace==NS['w'] for e in m.iter())
   total+=1
 assert total==13
 assert sum(n.endswith('.xlsx') for n in z.namelist())==6
 # All slide parts outside the requested range and all notes remain byte-identical.
 for n in z.namelist():
  if n.startswith('ppt/slides/slide') and n.endswith('.xml') and n not in {f'ppt/slides/slide{k}.xml' for k in range(5,11)}:
   assert z.read(n)==files[n]
print(json.dumps({'output':str(OUT),'native_equations':total,'slides':18,'charts':6,'modified_slides':list(range(5,11))},ensure_ascii=False))
