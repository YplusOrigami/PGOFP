const F='PingFang SC',INK='#14283F',MUTED='#53667A',BLUE='#285DAD',TEAL='#167D86';
function text(s,value,x,y,w,h,size=26,color=INK,bold=false){
 const q=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 q.text=value;q.text.style={typeface:F,fontSize:size,color,bold,insets:0,wrap:'square',autoFit:'none'};return q;
}
export async function restructure(p){
 const old=(await p.inspect({kind:'slide',limit:30})).records;
 if(old.length!==15)throw new Error('Expected original 15 slides');
 const toc=p.slides.add({width:1280,height:720});toc.background.fill='#FFFFFF';
 text(toc,'目录',56,42,1168,66,44,INK,true);
 const rows=[
  ['01','研究背景与任务分工','快慢控制回路、EAST 配置与环境边界','03',160,BLUE],
  ['02','环境与方法','IsoFlux 线性响应、稀疏观测与 Profile 输运和 NTM','04–10',320,BLUE],
  ['03','实验与评价','参考一致性、GPU 实现、基线结果与泛化局限','11–18',480,TEAL]
 ];
 for(const [n,h,b,page,y,color] of rows){text(toc,n,58,y,90,60,42,color,true);text(toc,h,160,y+2,850,52,34,INK,true);text(toc,b,160,y+64,930,45,25,MUTED);text(toc,page,1090,y+9,135,42,27,color,true);}
 toc.speakerNotes.text='建议用时 0.75 分钟。\n\n本次报告分三个部分。先用任务总览说明快慢回路的控制对象和环境边界。技术部分依次讲 IsoFlux 的线性响应、稀疏观测和奖励，再讲 Profile 的输运、磁岛反馈与储能折中。实验部分先核对参考引擎一致性和GPU吞吐，再看控制器结果与未见平衡态测试，最后讨论模型限制和研究问题。两张分节页只用于切换主题，各讲约半分钟，技术推导与原图分析仍占主要时间。\n\n来源：所给论文 §2–4、附录 C–F；目录为汇报结构。';
 function section(num,title,subtitle,line1,line2,color){
  const s=p.slides.add({width:1280,height:720});s.background.fill='#FFFFFF';
  text(s,title,56,270,1160,80,57,INK,true);
  text(s,num,56,112,200,125,110,color,true);
  text(s,subtitle,59,371,1150,60,31,color);
  text(s,line1,59,488,1150,49,27,INK);
  text(s,line2,59,556,1150,49,27,INK);
  return s;
 }
 const methods=section('02','环境与方法','从物理状态、动作约束到强化学习任务',
  '05–07   IsoFlux：线性动力学、稀疏诊断与跟踪奖励',
  '08–10   Profile：输运耦合、NTM 抑制与储能目标',BLUE);
 methods.speakerNotes.text='建议用时 0.5 分钟。\n\n下面进入技术部分，先讲千赫兹磁平衡回路，再讲百赫兹剖面回路。每个环境都围绕四个问题展开：内部状态怎样演化，策略能看到什么，动作经过哪些物理限制，以及奖励怎样组织控制目标。重点留意内置反馈环和固定参数，它们决定策略实际承担的任务范围。两个模型的时间尺度和控制对象不同，因此奖励数字不能跨环境直接比较。\n\n来源：论文 §3、附录 C.4–C.7。';
 const results=section('03','实验与评价','从数值一致性到控制性能与泛化',
  '12–15   参考引擎、训练协议与两类控制任务的结果',
  '16–18   未见平衡态测试、复现边界与研究讨论',TEAL);
 results.speakerNotes.text='建议用时 0.5 分钟。\n\n模型和任务定义之后，接下来分层看证据。首先确认GPU实现与参考引擎是否一致，再说明基线的信息权限和训练条件，随后分析主实例的奖励和动作使用，最后检查未见平衡态上的表现。数值一致性、标称控制性能与泛化是不同层次的结论。对PID与特权观测策略的比较尤其需要保留信息权限的差异。\n\n来源：论文 §4、附录 D–F。';
 const now=(await p.inspect({kind:'slide',limit:30})).records;
 const tocId=now.find(x=>x.title==='目录').id;
 const methodId=now.find(x=>x.title==='环境与方法').id;
 const resultId=now.find(x=>x.title==='实验与评价').id;
 const order=[old[0].id,tocId,old[1].id,methodId,...old.slice(2,8).map(x=>x.id),resultId,...old.slice(8).map(x=>x.id)];
 p.slides.reorder(order);
 const times=[.75,.75,1.5,.5,2.5,2,2,3,2.5,2,.5,2,.75,2,2,2,1.75,1.5];
 const records=(await p.inspect({kind:'textbox',limit:400})).records;
 for(const r of records){if(/^\d{2} \/ 15$/.test(r.text)){p.resolve(r.id).text=`${String(r.slideIndex+1).padStart(2,'0')} / 18`;}}
 for(let i=0;i<18;i++){
  const s=p.slides.items[i];
  s.speakerNotes.text=s.speakerNotes.text.replace(/^建议用时 [\d.]+ 分钟。/,`建议用时 ${times[i]} 分钟。`);
  if([1,3,10].includes(i))text(s,`${String(i+1).padStart(2,'0')} / 18`,1158,674,80,24,16,MUTED);
  if(i===2)text(s,'01  研究背景与任务分工',58,10,1150,22,16,MUTED);
  else if(i>=4&&i<=9)text(s,'02  环境与方法',58,10,1150,22,16,BLUE);
  else if(i>=11)text(s,'03  实验与评价',58,10,1150,22,16,TEAL);
 }
 return {slides:p.slides.items.length,minutes:times.reduce((a,b)=>a+b,0),order};
}
