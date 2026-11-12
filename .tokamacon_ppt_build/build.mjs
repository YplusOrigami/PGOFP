import fs from 'node:fs/promises';
const ROOT='/Volumes/T7/projects/PGOFP/.tokamacon_ppt_build';
const C={ink:'#14283F',muted:'#53667A',blue:'#285DAD',teal:'#167D86',red:'#C25442',gold:'#A77B20',light:'#E1E7ED',bg:'#FFFFFF'};
const FONT='PingFang SC';
const source='/Users/wuaijia/Downloads/11951_TokamaCon_An_Experiment_.pdf';
export const timings=[1,2,2.5,2,2,3,2.5,2,2,1,2,2,2,2,2];
const notes=[
`本次报告讨论 TokamaCon，一套从 EAST 实验相关配置出发构建的托卡马克控制环境。先说明论文状态：用户提供的是 ICLR 2027 匿名在审稿，不能称为已录用工作。它的主贡献是环境与评估基础设施，PPO 等算法用于展示任务特性。我们围绕两个问题展开：物理过程怎样变成一个部分可观测的控制任务，以及现有基线究竟证明了什么。技术部分先讲磁平衡，再讲温度和磁岛，随后解释模型一致性、GPU 批量仿真与实验结果。最终需要区分环境的复用能力、策略的泛化能力和真实装置的部署能力，这三者不能混为一谈。`,
`从图左的线圈和磁通曲线看快速磁回路，从图右的径向剖面看慢速输运回路。磁控制以约一千赫兹调节电流和边界，剖面控制约一百赫兹调节温度、电流分布和稳定性。真实装置中这两层相互影响，但论文将它们分别实例化。IsoFlux 的主要结果使用放电152444，而 Profile 使用137220的配置，图中间的磁通图只是137220在2.5秒的示意，不是全部实验共享的平衡态。论文提供两种环境，统一观测、动作和评估约定，目的是降低不同控制算法比较时的环境差异。实验困难包含间接诊断、参数漂移、执行器边界及刚性动力学。强调这两个环境没有运行时状态交换，Profile 的压力或内感变化不会在线传给 IsoFlux，因此当前不能从结果推断完整放电的联合闭环性能。`,
`IsoFlux 的模型来自参考平衡态附近的线性化自由边界响应。内部172维状态由12个主动线圈电流、158个被动涡流模态、等离子体电流和垂直运动子状态组成。输入矩阵有16列，但策略只给12路PF和CS电压。其余输入由环境或场景控制，包含VS、环电压偏置及两个剖面扰动。这里最重要的是原始线性化模型有一个约225.85每秒的不稳定垂直模态，增长时间约4.43毫秒，环境已经用内部bang-bang控制环处理它。策略并没有独立学会垂直稳定。每个外步1毫秒，通过100次、每次10微秒的三阶RK积分推进。CS与PF的限幅不同，动作又经过5毫秒传输延迟，因此策略选出的动作不会立刻作用于状态。线性化模型只在参考平衡态附近有效，远离参考的强位形变化不能直接相信。`,
`这一页解释为什么局部线性模型仍然构成困难的学习问题。传感器给出12维信号，但输出模型有188维，其中大量线圈和被动涡流模式不可见。两个相同的瞬时观测可能对应不同的内部电流和待执行电压队列，因此它们后续的响应不同。对策略而言，条件应该是观测历史或其压缩后的信念状态，而不是仅看当前观测。LSTM可以利用历史，但不能保证恢复全部隐变量。下方原文图7的PPO-LSTM例子分别显示等磁通、X点两分量磁场和电流。要注意磁通曲线共同下降是正常的，目标是各边界点靠近X点磁通，而不是把磁通固定为零。文中将188维称作特权观测，但它未必包括执行器全部延迟缓存，所以PPO-FO只是得到文中定义的更多信息。附录还明确，普通PPO及LSTM的critic没有使用特权通道，不能把它们说成非对称critic训练。`,
`三个奖励误差具有不同物理单位，先分别归一化，最后组合。电流误差以千安计算，磁通误差是八个控制点与X点磁通差的平均绝对值，磁场误差是X点两分量的二范数。好误差默认零，坏误差是奖励下降到零的截断点。组合系数alpha为零时就是三项平均，负值更强调最差项，正值更强调最好项，因此同一个组合器能改变目标折中。主实验的动作平滑项权重为零，不能把结果归因于平滑正则。下方图2显示零动作、随机动作与几种学习控制器的磁通共漂移。图中灰线分散说明等磁通误差变大，红灰曲线接近说明边界控制改善。奖励坏误差与终止阈值不是同一参数，例如评估磁场死区更严格，不能拿0.05特斯拉的奖励尺度代替安全边界。`,
`Profile在64个径向网格上演化电子温度和极向磁通。屏幕上的温度方程是原文用于解释机制的紧凑式，真实实现还包含几何系数、单位换算与边界条件，不应直接据此复制一个完整求解器。极向磁通演化改变安全因子q，温度又通过电导和自举电流反馈到电流分布，形成耦合。电子密度在回合内固定，参考几何也主要固定，因此这不是完整的粒子、能量和电流联合演化。默认Felici经验闭合把热扩散率分成新经典、异常、内部输运垒和中心增强等项，可切换BgB闭合。图中chi、sigma和q来自原文参考一致性诊断，实线是本环境、虚线是RAPTOR。它们说明两份数值实现的关系，不是对真实输运的独立测量。右下高斯沉积曲线依据原文方程生成，只展示位置和宽度如何改变局部功率沉积，不代表实验采样。四束波的中心、宽度和功率共同构成12维动作，总功率上限3.2兆瓦。`,
`NTM用降阶的修正Rutherford方程描述。在q等于2的有理面种下2比1磁岛，经典项、自举项和外部驱动电流项竞争。公式是正文的机制概括，完整模型还加入曲率以及ECH等贡献。温度升高可能提高储能，同时温度梯度通过自举电流影响失稳驱动；把ECCD沉积在有理面附近，则提供稳定化电流。磁岛进一步放大有理面周围的有效热扩散率，导致温度变平，从而将磁岛信息间接传入12个ECE温度通道。策略不直接看到磁岛宽度。图3的学习策略在显示窗口中抑制了尖峰，而简单基线反复失败并自动重置。但图的阈值约为0.06米，与正文0.15米不同，图上起始宽度也接近零。不能将这个片段未经核对就理解为统一默认参数下的一次完整回合。作者明确承认磁岛宽度动力学没有直接实验验证，所以这一任务适合研究控制机制，不构成定量装置NTM验证。`,
`奖励把指数磁岛惩罚和相对储能相乘。默认a等于5每米，b等于1，软阈值为零，参考储能60千焦。左图按这个确切公式绘制，在固定储能下，磁岛从0增加到0.15米，因子从1下降到约0.472。图的几条线表示不同相对储能，以说明一个因素可部分补偿另一个因素；乘法奖励鼓励兼顾目标，但不保证分别满足硬约束。文字配置规定磁岛大于0.15米终止并额外扣10，硬终止和软奖励应分开看。右表解释动作解码。归一化零动作会映射到中间值，每束功率0.55兆瓦，合计2.2兆瓦，所以ZeroAction若直接走相同接口就不是关机。每步动作先加标准差0.1的高斯噪声再裁剪。另外内部PI环保持目标电流，因此RL没有独自承担全部慢回路控制。离子储能来自固定电子/离子剖面比例，也属于闭合假设。`,
`GPU价值来自将许多环境的积分与观测留在同一张设备上，避免逐步回到CPU。上左是规定加热轨迹下Profile与RAPTOR的温度剖面，EC功率在1.099秒从700千瓦升到1.35兆瓦。相对L2误差约1.94%，作者解释为刚性方程求解及离散差异。IsoFlux对应PCSVP的误差为9.81乘10的负5次方，测试步数和批量与温度对比不同。这两者不能称为真实实验误差。表内吞吐为批量256下每秒全部环境推进的总步数，CPU和GPU执行方式不同。7071除以1.57约4504倍，Profile约3.47倍。大并行吞吐对训练有益，但并不证明单次观测到电压输出的闭环延迟小于1毫秒。实现方程一致性、上游模型实验校准和学习策略部署验证是三个不同证据层次。原文称每个基线训练约9小时、80GB显存GPU，但未在对应段落明确型号。`,
`基线设计用不同的信息权限探查任务。PPO为两层256单元前馈网络，LSTM在观测编码后加入128维循环状态，FO让actor获得特权观测。PID只在IsoFlux提供，Profile缺少同等级传统闭环参照。三种RL共用主要PPO超参数：三千万交互，32步rollout，gamma0.99，GAE0.95，clip0.2。并行数IsoFlux为256，Profile为1024，后者不同于吞吐表的测速批量256。报告指标除奖励还包括跟踪误差、MHD代理违规率、归一化失效时间和饱和程度。超时和物理失败必须用不同bootstrap处理，环境还会自动重置完成的行，接入新的算法时需要保留真正终止前的末观测。这些训练规则会显著影响公平性。主要结果没有给出完整种子数和置信区间，因此下一页的差距首先是论文报告的点估计。`,
`左图使用表3精确数字重绘。PPO0.7441，LSTM0.9238，PID0.9478，FO0.9741。循环策略提升0.1797，消除了PPO到FO差距的约78.1%，这支持历史信息有价值的解释。右边原文训练曲线显示类似排序。可是可部署的传感器策略仍低于PID。FO比PID高0.0263，是额外信息与控制方法共同变化后的对比，不是同等信息条件下RL胜出。主实例饱和比例为PPO0.48%、LSTM0.18%、FO0.11%，看起来更丰富信息减少了靠近边界的动作，但这个规律只适用于当前实例。它不能推广到Profile或另一平衡态，也不能直接当作安全性概率。训练架构差异也意味着不能将LSTM的所有增益严格归于状态估计一种机制。`,
`Profile奖励的最大差距只有0.0091，但饱和指标从1.02%到20.94%，相差约20.5倍。两个纵轴分别画，避免截断奖励坐标夸大微小差距。作者解释为功率上限限制驱动电流，所以FO额外信息无法提高可达到的奖励。该解释具有物理合理性，但需要功率上限扫描或控制变量消融进一步验证。饱和指标把12个通道一起统计，包括沉积位置、宽度以及功率的上下界，所以20.94%不能翻译成最大功率持续时间。原文正文的任一通道触边比例与附录通道平均公式又不一致，因此此处保留表中数值，只标为报告的饱和指标。实际工程评估应同时看任务成功、失效时间、逐通道边界使用以及未用动作余量。奖励差异没有置信区间，不能给出显著性结论。`,
`左侧是表10的关键补充实验：作者在137220平衡态A重新训练，在B上冻结测试，不是把152444策略零样本迁移过去。FO在A达到0.9536，B只剩0.2722，下降71.5%；LSTM下降40.2%。PID从0.9539变成0.9339，只降约2.1%。PPO相对降幅较小，但本来表现就低，不能因此判定它全面更好。说明环境管线能换模型包复用，与已经训练的策略能跨模型泛化，是不同命题。右图是原始152444实例的噪声、观测延迟、beta和内感漂移扫描，它没有替换新的EFIT模型包，横轴又是归一化严重程度。两组实验分别测参数/诊断扰动和新平衡态，两者相互补充。Profile局部chi上下2%测试的奖励排序虽稳定，但FO饱和从17.73%到24.33%变化，不能由奖励稳定推出动作裕度稳定，也不能把参数扰动等同于约2%的轨迹输出误差。`,
`这一页集中处理影响复现的文稿冲突，避免散落到每个技术页打断叙事。种子宽度在正文表1、表5是0.15米，而附录表8是0.015米，相差十倍。图3阈值约0.06米，文字终止0.15米。我们没有检查代码，不能猜哪一个才是实际实验参数。饱和定义的差异尤其会改变数值：如果每个时刻恰有一个动作通道触边，任意通道触边的时间比例是100%，12通道平均却为8.33%。必须先确认评估实现。ZeroAction也需要确认是否绕过动作解码。除此之外，研究范围的物理限制是IsoFlux局部线性化、Profile固定密度与经验输运闭合、合成ECE和未验证的NTM，以及两个回路独立。统计上主要表格缺少种子数量和不确定性。所有问题都来自给定PDF交叉检查，不是已完成的源码审计，也不能据此否定整个环境框架。`,
`最后把证据落到可开展的研究上。第一，平台价值在于将数据相关模型和统一批量接口开放给控制学习研究。第二，当前结果最清楚展示了三个困难：IsoFlux稀疏观测有代价，Profile相同奖励掩盖不同动作使用，跨平衡态的高标称性能不保证鲁棒。PID尤其值得保留为强基线，可以检验学习策略是否在同等信息下改进样本效率、跟踪与泛化。对于控制先验与流策略方向，合理实验是同条件的PID单独、先验加噪声以及学习策略，训练预算和观测权限一致，并把先验计算开销计入推理时延。这个方向是阅读后的研究设想，论文没有验证PGOFP或任何流策略收益。结束时可讨论两个问题：哪些隐变量最值得估计，以及怎样通过多平衡态训练和真正未见测试建立泛化证据。30分钟包括本页约两分钟结论与交流，若现场讨论较多，可缩短训练配置页。`
];

function tx(s,text,x,y,w,h,size=25,color=C.ink,bold=false){
 const q=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 q.text=text;q.text.style={fontSize:size,typeface:FONT,color,bold,wrap:'square',insets:0,verticalAlignment:'top',autoFit:'none'};return q;
}
function head(p,n,title,ref){
 const s=p.slides.add({width:1280,height:720});s.background.fill=C.bg;
 tx(s,title,56,36,1168,64,42,C.ink,true);
 tx(s,`${String(n).padStart(2,'0')} / 15`,1158,674,80,24,16,C.muted);
 tx(s,`来源：论文 ${ref}`,56,674,1070,24,15,C.muted);
 s.speakerNotes.text=`建议用时 ${timings[n-1]} 分钟。\n\n${notes[n-1]}\n\n来源：${source}；${ref}。图表重绘仅使用论文表内数值；公式曲线另标为示意。`;return s;
}
async function img(s,name,x,y,w,h){const b=await fs.readFile(`${ROOT}/assets/${name}.png`);s.images.add({dataUrl:`data:image/png;base64,${b.toString('base64')}`,contentType:'image/png',alt:`TokamaCon论文图：${name}`,fit:'contain',position:{left:x,top:y,width:w,height:h}});}
function para(s,label,body,x,y,w=540){tx(s,label,x,y,w,34,27,C.blue,true);tx(s,body,x,y+43,w,110,25,C.ink);}
function table(s,values,x,y,w,h,widths){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,columnWidths:widths,values});
 t.borders.assign({fill:C.light,width:1,style:'solid'});
 t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({fill:'#FFFFFF',textStyle:{fontSize:23,typeface:FONT,color:C.ink},margins:{left:12,right:10,top:10,bottom:8}});
 t.cells.block({row:0,column:0,rowCount:1,columnCount:values[0].length}).assign({fill:'#EAF0F6',textStyle:{fontSize:23,typeface:FONT,bold:true,color:C.ink}});
 return t;
}
function chart(s,{x,y,w,h,cats,series,max=1,format='0.0',legend=false,title,kind='bar',labels=true}){
 const q=s.charts.add(kind,{position:{left:x,top:y,width:w,height:h},categories:cats,series,hasLegend:legend,legend:{position:'bottom',overlay:false,textStyle:{fontSize:17,fill:C.muted}},barOptions:{direction:'column',grouping:'clustered',gapWidth:95},title,titleTextStyle:{fontSize:24,fill:C.ink,bold:true},chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF',xAxis:{textStyle:{fontSize:18,fill:C.ink},line:{fill:C.light,width:1}},yAxis:{min:0,max,numberFormatCode:format,textStyle:{fontSize:17,fill:C.muted},majorGridlines:{fill:C.light,width:1}},dataLabels:{showValue:labels,position:'outEnd',numberFormatCode:format,textStyle:{fontSize:20,bold:true,fill:C.ink}}});return q;
}
export async function populate(p){
 p.theme.defaultFont=FONT;
 p.theme.colorScheme={name:'TokamaCon',themeColors:{accent1:C.blue,accent2:C.teal,accent3:C.red,accent4:C.gold,accent5:'#7789A1',accent6:'#64626B',bg1:'#FFFFFF',bg2:'#F4F6F8',tx1:C.ink,tx2:C.muted,dk1:'#000000',dk2:C.ink,lt1:'#FFFFFF',lt2:C.light,hlink:C.blue,folHlink:C.teal}};
 let s;
 // 1. Minimal title slide.
 s=head(p,1,'TokamaCon','标题页，pp.1–3');
 tx(s,'基于 EAST 数据的托卡马克控制环境',58,178,1130,78,48,C.ink,true);
 tx(s,'快速磁平衡控制与慢速剖面控制的技术解读',60,278,1120,52,32,C.blue);
 tx(s,'1 kHz  /  IsoFlux',60,416,550,50,36,C.ink,true);
 tx(s,'100 Hz  /  Profile',650,416,550,50,36,C.teal,true);
 tx(s,'论文汇报 · 30 分钟',60,568,600,38,25,C.muted);
 tx(s,'匿名在审稿，页眉标注 ICLR 2027',60,612,900,28,21,C.muted);
 // 2. Overview.
 s=head(p,2,'两种控制环境的任务分工','图 1，§2–3，附录 C.2');
 await img(s,'overview',56,120,730,390);
 tx(s,'IsoFlux',840,140,360,42,32,C.blue,true);
 tx(s,'12 路线圈电压\n1 ms 控制周期，2 s 回合\nEAST 152444 主实例',840,195,370,140,25);
 tx(s,'Profile',840,365,360,42,32,C.teal,true);
 tx(s,'4 束 ECH / ECCD\n10 ms 控制周期，4 s 回合\nEAST 137220 配置',840,420,370,145,25);
 tx(s,'两环境独立实例化，没有在线状态交换',58,580,1140,45,31,C.red,true);
 // 3. magnetic model.
 s=head(p,3,'IsoFlux 的线性响应与执行器动力学','§3.1，附录 C.4，表 6–7');
 tx(s,'ẋ = Ax + Bu',60,130,630,62,43,C.blue,true);
 tx(s,'y = C(x − xeq) + yeq',60,205,650,60,38);
 tx(s,'x ∈ ℝ¹⁷²     u ∈ ℝ¹⁶     y ∈ ℝ¹⁸⁸',60,289,670,45,29);
 tx(s,'12 主动电流 + 158 被动模态\n+ 等离子体电流 + 垂直运动状态',60,359,665,90,27);
 table(s,[['模型与执行器','原始实例设置'],['策略动作','12 路 PF / CS 电压'],['CS / PF 限幅','±300 V / ±900 V'],['动作延迟','5 ms，即 5 个控制步'],['数值推进','100 × 10 μs RK3']],755,137,470,350,[215,255]);
 tx(s,'内部 VS 控制环已抑制垂直不稳定模态',60,534,1120,45,30,C.red,true);
 tx(s,'λ ≈ +225.85 s⁻¹，增长时间约 4.43 ms；策略不控制该回路',60,591,1130,38,24,C.muted);
 // 4. Observability.
 s=head(p,4,'稀疏诊断与 5 ms 延迟构成部分可观测问题','§3.1，图 7，附录 D.2');
 tx(s,'12 维 actor 观测',60,132,540,42,31,C.blue,true);
 tx(s,'8 个控制点磁通 + X 点磁通\nX 点两个磁场分量 + Ip',60,188,540,94,27);
 tx(s,'188 维特权输出',680,132,540,42,31,C.teal,true);
 tx(s,'追加线圈电流、位置与被动涡流模态\n瞬时传感观测仍缺少内部动态信息',680,188,540,94,26);
 tx(s,'π(aₜ | oₜ)     与     π(aₜ | o₀:ₜ)',60,305,1160,55,35,C.ink,true);
 await img(s,'magnetic_trace',60,387,1160,218);
 tx(s,'原文 PPO-LSTM 示例：等磁通共漂移、X 点磁场与电流跟踪',60,625,1150,30,21,C.muted);
 // 5. reward & rollouts.
 s=head(p,5,'IsoFlux 奖励同时约束三类跟踪误差','式 (7)，图 2，附录 C.7.1');
 tx(s,'eIp = |Ip / 1000 − Iref|',58,130,590,48,30,C.blue,true);
 tx(s,'eψ = ⅛ Σᵢ |ψcp,i − ψX|',58,192,590,48,30,C.blue,true);
 tx(s,'eBX = ‖BX‖₂',58,254,590,48,30,C.blue,true);
 tx(s,'rₖ = clip[(bₖ − eₖ)/(bₖ − gₖ), 0, 1]',670,135,555,48,27);
 tx(s,'gₖ = 0；bₖ 为奖励归零的误差',670,192,555,35,23,C.muted);
 tx(s,'默认 r = (rIp + rψ + rBX) / 3',670,236,555,48,29,C.ink,true);
 tx(s,'动作平滑项默认关闭\n奖励尺度与终止阈值分别配置',670,303,550,80,24,C.muted);
 await img(s,'flux_rollout',58,420,1160,213);
 // 6. Profile solver.
 s=head(p,6,'Profile 的温度、磁通与电流驱动耦合','式 (2)、(5)，图 8，附录 C.5–C.6');
 tx(s,'(3/2) ne ∂Te/∂t = ∂ρ(ne χe ∂ρTe) + Stot',58,126,1165,58,32,C.teal,true);
 tx(s,'64 点径向网格；ψ 与 Te 演化，ne 固定；默认 Felici 经验输运闭合',58,195,1160,42,25);
 await img(s,'diagnostics',58,260,670,255);
 tx(s,'χe、σ∥ 与 q 的参考引擎对照',58,535,660,34,23,C.muted);
 const rr=Array.from({length:11},(_,i)=>(i/10).toFixed(1));
 chart(s,{x:770,y:258,w:455,h:294,cats:rr,series:[{name:'rdep=0.40',values:rr.map(v=>Math.exp(-4*(Number(v)-.4)**2/.2**2)),line:{fill:C.teal,width:3},marker:{symbol:'none'}},{name:'rdep=0.60',values:rr.map(v=>Math.exp(-4*(Number(v)-.6)**2/.2**2)),line:{fill:C.blue,width:3},marker:{symbol:'none'}}],max:1,legend:true,kind:'line',labels:false,title:'高斯沉积核（归一化示意）'});
 tx(s,'dP/dV ∝ Pin exp[−4((ρ − r)/w)²]',770,566,460,54,24,C.teal,true);
 tx(s,'4 束波 ×（位置、宽度、功率）= 12 维动作',58,609,1160,37,27,C.ink,true);
 // 7. NTM.
 s=head(p,7,'NTM 磁岛把稳定性与温度剖面连接起来','式 (3)、(6)，图 3，§3.2');
 tx(s,'dw/dt ∝ Δ′ + a₁ Jbs / (w² + wd²) − a₂ Jcd / (w² + wmarg²)',58,128,1168,62,29,C.teal,true);
 tx(s,'自举电流驱动磁岛增长；在 q = 2 有理面沉积 ECCD 可提供稳定化电流',58,214,1160,43,25);
 tx(s,'磁岛增大提高局部有效 χe，温度剖面变平，ECE 只间接观测这一变化',58,271,1160,43,25);
 await img(s,'island_rollout',58,348,1160,280);
 tx(s,'图示阈值约 0.06 m，与文字 0.15 m 不一致；NTM 动力学未获直接实验验证',58,632,1160,29,20,C.red);
 // 8. Reward/action.
 s=head(p,8,'Profile 的储能目标与磁岛惩罚存在折中','式 (8)，附录 C.6–C.7.2');
 tx(s,'r = exp[−5w] × (Wth / 60 kJ)',58,130,1170,58,38,C.teal,true);
 const ww=Array.from({length:7},(_,i)=>i*.025);
 chart(s,{x:50,y:222,w:600,h:345,cats:ww.map(v=>v.toFixed(3)),series:[.8,1,1.2].map((e,i)=>({name:`W/Wref=${e.toFixed(1)}`,values:ww.map(v=>e*Math.exp(-5*v)),line:{fill:[C.muted,C.teal,C.blue][i],width:3},marker:{symbol:'circle',size:4}})),max:1.3,legend:true,kind:'line',labels:false,title:'按原文奖励公式计算的示意曲线'});
 tx(s,'磁岛宽度 w（m）',205,577,420,28,22,C.muted);
 table(s,[['单束动作','范围','零动作解码'],['rdep','0–0.85','0.425'],['wdep','0.15–0.25','0.20'],['Pin','0.3–0.8 MW','0.55 MW']],705,225,520,250,[160,185,175]);
 tx(s,'原始零动作对应总功率 2.2 MW',705,507,520,42,27,C.red,true);
 tx(s,'文本终止：w > 0.15 m，额外惩罚 −10\n动作噪声 σa = 0.1，裁剪后仿射解码',705,567,520,74,23);
 // 9. numerical parity and gpu.
 s=head(p,9,'参考引擎一致性与 GPU 批量仿真','图 4–5，表 2，附录 C.3、D.6');
 await img(s,'parity',58,132,620,190);
 tx(s,'Profile 与 RAPTOR 的温度剖面对照',58,338,620,31,22,C.muted);
 await img(s,'runtime',725,132,500,234);
 table(s,[['环境','相对 L2 误差','CPU 步/s','GPU 步/s'],['IsoFlux','9.81 × 10⁻⁵','1.57','7071'],['Profile','1.94 × 10⁻²','795','2756']],58,409,1165,171,[230,355,290,290]);
 tx(s,'B = 256 的总环境吞吐；单次训练约 9 h，使用 80 GB 显存 GPU',58,602,1160,38,25,C.ink);
 // 10. baselines.
 s=head(p,10,'基线设计隔离观测、记忆与工程控制知识','表 9，附录 C.8、D.1–D.4');
 table(s,[['控制器','actor 信息','设计目的'],['PPO','12 维传感观测','前馈学习参照'],['PPO-LSTM','传感观测历史','研究历史信息的价值'],['PPO-FO','文中定义的特权观测','研究额外状态信息'],['PID','工程反馈设计知识','IsoFlux 工程基准']],58,130,1165,320,[260,425,480]);
 tx(s,'30 M 交互，γ = 0.99，GAE λ = 0.95，rollout = 32，clip = 0.2',58,489,1160,40,27,C.blue,true);
 tx(s,'PPO 的 actor 与 critic 均使用传感观测；Profile 未提供工程闭环基线',58,551,1160,40,25);
 tx(s,'评估需同时保留跟踪误差、违规率、失效时间与动作边界使用',58,611,1160,38,25,C.muted);
 // 11. Iso result.
 s=head(p,11,'IsoFlux：记忆明显改善 PPO，PID 仍然很强','表 3，图 6(a)，§4.3');
 chart(s,{x:52,y:139,w:608,h:377,cats:['PPO','PPO-LSTM','PID','PPO-FO'],series:[{name:'奖励',values:[.7441,.9238,.9478,.9741],fill:C.blue,points:[{idx:0,fill:'#91A6C2'},{idx:1,fill:C.blue},{idx:2,fill:C.teal},{idx:3,fill:'#4D718C'}]}],max:1.1,format:'0.0000',title:'主实例评估奖励（EAST 152444）'});
 await img(s,'learning',715,145,510,330);
 tx(s,'LSTM 缩小约 78.1% 的 PPO–FO 奖励差距',58,548,1160,45,30,C.blue,true);
 tx(s,'FO 比 PID 高 0.0263，但使用更多状态信息；结果未证明传感器 RL 超过 PID',58,609,1160,40,24);
 // 12. Profile results.
 s=head(p,12,'Profile：相近奖励掩盖了动作边界使用差异','表 3，§4.3，附录 C.8.4');
 const cats=['PPO','PPO-LSTM','PPO-FO'];
 chart(s,{x:48,y:140,w:575,h:367,cats,series:[{name:'奖励',values:[.642,.6412,.6329],fill:C.teal}],max:.75,format:'0.0000',title:'评估奖励'});
 chart(s,{x:655,y:140,w:575,h:367,cats,series:[{name:'饱和指标',values:[.0102,.0877,.2094],fill:C.red}],max:.25,format:'0.00%',title:'论文报告的饱和指标'});
 tx(s,'奖励范围仅 0.0091，饱和指标最大相差约 20.5 倍',58,548,1160,44,30,C.teal,true);
 tx(s,'统计包含位置、宽度和功率通道；缺少误差条，不能直接判断微小差异显著',58,610,1160,40,24);
 // 13. generalization.
 s=head(p,13,'未见平衡态测试暴露了学习策略的泛化不足','表 10，图 6(b)，附录 E.1–E.2');
 chart(s,{x:48,y:140,w:665,h:376,cats:['PID','PPO','PPO-LSTM','PPO-FO'],series:[{name:'A：训练平衡态',values:[.9539,.5952,.8019,.9536],fill:C.blue},{name:'B：未见平衡态',values:[.9339,.5465,.4795,.2722],fill:C.red}],max:1.1,format:'0.0000',legend:true,title:'EAST 137220，A 上训练后冻结测试'});
 await img(s,'ood',758,154,465,315);
 tx(s,'原实例的噪声、延迟与参数漂移扫描',758,489,465,56,22,C.muted);
 tx(s,'FO 在 B 上奖励下降 71.5%；PID 下降约 2.1%',58,561,1160,44,31,C.red,true);
 tx(s,'137220 策略在 A 上重新训练；该实验不属于 152444 策略的零样本迁移',58,616,1160,34,23);
 // 14. limitations and audit.
 s=head(p,14,'模型范围与复现参数仍需明确','表 1、8，图 3，式 (12)，附录 F');
 table(s,[['核对项','PDF 中的不同表述','影响'],['磁岛种子','正文 0.15 m；表 8 为 0.015 m','初始难度相差明显'],['破裂阈值','文字 0.15 m；图 3 约 0.06 m','图与配置无法直接对齐'],['饱和比例','任一通道触边；时间 × 通道平均','两种统计不等价'],['ZeroAction','零原始动作；非零物理功率','需核对基线路径']],58,134,1165,310,[220,575,370]);
 tx(s,'IsoFlux 局部线性化；Profile 固定 ne、经验输运、合成 ECE 与降阶 NTM',58,490,1160,70,26,C.ink);
 tx(s,'两个回路独立；主表缺少置信区间；模拟器一致性不构成实机部署验证',58,581,1160,65,25,C.red);
 // 15. conclusions and research.
 s=head(p,15,'结论与后续研究问题','§5，附录 B.3、F；研究设想另标');
 para(s,'平台价值','统一 EAST 相关模型、任务接口和 GPU 批量 rollout',58,140,1160);
 para(s,'最有力的实验认识','记忆有价值；奖励难以概括动作行为；跨平衡态泛化仍弱',58,276,1160);
 para(s,'研究设想：控制先验与鲁棒学习','同等观测下比较 PID、先验加噪声与学习策略，保留未见平衡态测试',58,412,1160);
 tx(s,'讨论：哪些隐变量最值得估计？怎样验证模型变化下的控制收益？',58,590,1160,54,29,C.blue,true);
 return {slideCount:p.slides.items.length,totalMinutes:timings.reduce((a,b)=>a+b,0)};
}
