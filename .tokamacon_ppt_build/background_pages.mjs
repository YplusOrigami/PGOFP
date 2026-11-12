const F='PingFang SC',I='#14283F',B='#285DAD',T='#167D86',M='#53667A';
function tx(s,t,x,y,w,h,z=25,c=I,b=false){const q=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});q.text=t;q.text.style={typeface:F,fontSize:z,color:c,bold:b,insets:0,wrap:'square',autoFit:'none'};}
function tab(s,rows){const t=s.tables.add({rows:rows.length,columns:2,left:665,top:238,width:560,height:355,columnWidths:[190,370],values:rows});t.borders.assign({fill:'#E1E7ED',width:1});t.cells.block({row:0,column:0,rowCount:rows.length,columnCount:2}).assign({fill:'#FFFFFF',textStyle:{typeface:F,fontSize:23,color:I},margins:{left:12,right:10,top:10,bottom:8}});t.cells.block({row:0,column:0,rowCount:1,columnCount:2}).assign({fill:'#EAF0F6',textStyle:{typeface:F,fontSize:23,color:I,bold:true}});}
export function addBackgrounds(p){
 p.theme.defaultFont=F;
 let s=p.slides.add({width:1280,height:720});s.background.fill='#FFFFFF';
 tx(s,'02  环境与方法',58,10,1150,22,16,B);
 tx(s,'IsoFlux 背景：磁平衡控制的目标与接口',56,36,1168,64,42,I,true);
 tx(s,'PF / CS 线圈电压改变等离子体电流与边界。等磁通控制要求边界控制点的磁通接近 X 点磁通。',58,125,1160,85,27);
 tx(s,'控制目标',58,240,560,42,30,B,true);
 tx(s,'维持目标等离子体电流 Ip\n使八个边界控制点满足等磁通\n使 X 点两分量极向磁场接近零',58,294,565,125,26);
 tx(s,'必要背景',58,444,560,42,30,B,true);
 tx(s,'PF / CS：调节电流和边界的线圈回路\nX 点：极向磁场为零的特殊点\n内部 VS 回路负责垂直稳定',58,498,565,115,24);
 tab(s,[['控制接口','本文设定'],['策略输入','12 维磁诊断相关信号'],['策略输出','12 路 PF / CS 电压'],['环境返回','观测、奖励、结束标志'],['控制周期','1 ms，即 1 kHz'],['回合长度','2 s，共 2000 步']]);
 tx(s,'默认观测不含全部线圈电流、被动涡流和延迟状态',58,628,1160,31,23,M);
 tx(s,'来源：论文 §2、§3.1，附录 C.4；概念与接口说明',56,674,1080,24,15,M);
 tx(s,'05 / 20',1158,674,80,24,16,M);
 s.speakerNotes.text='建议用时 1 分钟。\n\nIsoFlux 研究快速磁平衡控制。PF和CS线圈的电压改变线圈电流与磁响应，进而影响等离子体电流及边界。等磁通目标是让八个边界控制点的磁通靠近X点磁通，而不是把各点磁通固定到零。X点是极向磁场为零的特殊点，环向磁场并不因此为零。策略输入为12维诊断信号：8个控制点磁通、X点磁通、X点两个极向磁场分量和等离子体电流。策略输出12路归一化电压，经限幅和延迟作用于PF/CS回路。环境每1毫秒返回新观测、奖励和结束信息。大量线圈和被动涡流内部动态没有完全进入actor观测，因此后面讨论部分可观测性。环境另有内部VS环稳定垂直模态。\n\n来源：11951_TokamaCon_An_Experiment_.pdf，§2、§3.1、附录C.4。';
 s=p.slides.add({width:1280,height:720});s.background.fill='#FFFFFF';
 tx(s,'02  环境与方法',58,10,1150,22,16,B);
 tx(s,'Profile 背景：剖面、磁岛与局部电流驱动',56,36,1168,64,42,I,true);
 tx(s,'剖面描述温度、电流等量随径向位置的分布。任务通过局部加热与电流驱动抑制磁岛，同时维持热储能。',58,125,1160,85,27);
 tx(s,'控制目标',58,240,560,42,30,T,true);
 tx(s,'抑制 2/1 NTM 磁岛增长\n维持热储能 Wth，避免温度塌降\n在有限功率下选择合适沉积区域',58,294,565,125,26);
 tx(s,'必要背景',58,444,560,42,30,T,true);
 tx(s,'磁岛会改变局部输运，使温度变平\nECH 加热电子，ECCD 驱动局部电流\nq = 2 有理面是本场景的抑制区域',58,498,565,115,24);
 tab(s,[['控制接口','本文设定'],['策略输入','12 个合成 ECE 温度通道'],['策略输出','4 束波的位置、宽度、功率'],['执行器上限','四束总功率不超过 3.2 MW'],['环境返回','温度观测、奖励、结束标志'],['控制周期','10 ms，即 100 Hz']]);
 tx(s,'策略不直接观测磁岛宽度；ne 在回合内固定；默认回合 4 s',58,628,1160,31,23,M);
 tx(s,'来源：论文 §2、§3.2，附录 C.5–C.7；概念与接口说明',56,674,1080,24,15,M);
 tx(s,'09 / 20',1158,674,80,24,16,M);
 s.speakerNotes.text='建议用时 1 分钟。\n\n剖面指物理量随归一化径向位置的分布，不只是一项平均温度。Profile在64点网格上演化温度和极向磁通，但策略只看到12个合成ECE温度采样。任务考核磁岛宽度和热储能，温度、电流及安全因子剖面是中间物理状态。四束电子回旋波各有位置、宽度和功率三个参数，共12维动作，总功率上限3.2MW。ECH主要提供加热，ECCD提供局部电流驱动，策略需在q=2有理面附近提供稳定化作用。磁岛改变局部输运并使温度变平，温度变化则使策略间接感知失稳影响。策略不能直接命令磁岛缩小，也不能直接看到磁岛宽度。电子密度在回合内固定，内部PI环维持目标等离子体电流，默认回合4秒。NTM模型是未获直接实验验证的降阶模型。\n\n来源：11951_TokamaCon_An_Experiment_.pdf，§2、§3.2、附录C.5–C.7。';
 return {slides:p.slides.items.length};
}
