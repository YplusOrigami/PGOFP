from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader
import html

root = Path('/Volumes/T7/projects/PGOFP')
out = root / 'output/pdf'
items = [
('1. 请简述本文的摘要。', '本文揭示了知识蒸馏中的一种新威胁——蒸馏条件后门攻击（DCBA）：攻击者在教师模型中植入休眠后门，使教师对正常输入和带触发器输入均表现正常，但开发者利用干净数据进行知识蒸馏后，学生模型会对带触发器输入产生攻击者指定的错误分类。为实现这种条件性激活，作者提出 SCAR，将教师模型训练表述为双层优化问题：内层训练代理学生以模拟蒸馏，外层依据代理学生的表现调整教师。该方法结合预优化触发器与隐式微分，在多种数据集、模型结构及蒸馏方式上验证了攻击有效性和教师端隐蔽性，说明通过安全检测的教师与干净蒸馏数据仍不足以保证学生安全。'),
('2. 本文是面向什么应用场景开展研究的？', '本文面向第三方预训练模型参与的模型压缩与轻量化部署场景，尤其是将大型图像分类模型蒸馏为适合物联网等资源受限设备的小模型。场景包含恶意模型提供者、负责安全检测的平台和下游开发者：攻击者控制教师模型的训练并将其上传平台；平台检测后发布模型；开发者获取教师并用自有干净数据蒸馏学生。攻击者只提供教师模型，不控制受害者的蒸馏数据和训练过程，也不知道实际采用的检测方法和蒸馏技术。研究关注的正是这种模型供应链中，教师检测通过而学生仍可能受攻击的安全缺口。'),
('3. 目前学术界已开展了哪些工作，现有的研究还存在哪些问题？', '已有研究包括 BadNets 等传统后门及隐形、多样化触发器设计；ADBA 等能够在知识蒸馏后保留后门的攻击；由剪枝、量化、微调等后处理激活的条件后门；以及 Neural Cleanse、BTI-DBF 等触发器反演检测和 SCALE-UP 等行为检测。其不足是：普通后门往往难以通过干净数据蒸馏存活；抗蒸馏攻击主要追求已有活跃后门的延续，未解决教师端休眠、学生端激活的问题；既有条件后门也未充分研究蒸馏这一条件。作者尝试将 ADBA 教师微调至掩盖后门，但其在多种学生结构和 ImageNet 子集上效果有限，可能因为仅依据教师输出调整，缺少蒸馏过程的反馈。依赖教师当前异常行为的检测，也容易遗漏这种潜在风险。'),
('4. 简述本文所提的方法的基本原理与实现步骤。', 'SCAR 的基本原理是把“教师对触发器保持正常”和“蒸馏后学生对触发器发生定向误分类”同时纳入训练目标。首先，利用正常预训练教师及其蒸馏学生，预优化一个受幅度约束的通用加性触发器，使其具备跨模型迁移能力。其次，构建双层优化：内层每轮重新初始化代理学生，仅用干净数据，通过真实标签监督和模仿教师输出进行有限步蒸馏；外层同时约束教师在干净、触发器样本上预测真实类别，代理学生在干净样本上预测真实类别、在触发器样本上预测攻击目标类别。再次，利用隐式微分和有限次不动点迭代近似学生训练对教师参数的依赖，获得外层梯度并更新教师，交替迭代后输出教师模型。其解释是干净数据蒸馏存在信息损失，可能传递与任务耦合的触发器特征，却未完整保留教师抵抗触发器的相关信息；这一解释仍属初步分析。'),
('5. 简述本文的创新点。', '本文的创新主要体现在三个方面：一是提出蒸馏条件后门攻击范式，揭示后门能够在教师中休眠、经干净数据蒸馏后在学生中激活，拓展了知识蒸馏的安全威胁模型；二是提出 SCAR 双层优化框架，将教师正常性、学生正常精度与学生攻击效果联合建模，利用代理学生的蒸馏反馈指导教师训练，并结合隐式微分与触发器预优化求解；三是从多种教师和学生结构、输出式／特征式／关系式蒸馏以及多类检测方法进行系统验证，并通过消融实验说明代理学生和预优化触发器的重要性。其关键贡献是揭示并实现这一安全缺口，而非提出一种新的知识蒸馏技术。'),
('6. 本文得出了什么结论？', '实验表明，教师模型通过后门检测、蒸馏数据保持干净，并不能保证学生模型不存在后门。以表 1 的 ResNet-50 教师为例，在 CIFAR-10 上教师攻击成功率仅为 1.50%，而三类蒸馏得到的不同学生攻击成功率为 77.28%—99.94%；在 50 类 ImageNet 子集上，教师为 2.12%，学生为 52.98%—91.96%。SCAR 还能在所测试条件下规避 Neural Cleanse、SCALE-UP 等教师端检测。因此，作者建议对蒸馏后的学生独立开展后门检测，不能直接沿用教师的安全结论。但方法在较复杂数据集上效果下降，存在正常精度损失与额外计算开销；实验主要针对图像分类，不能据此认定其对所有模型、任务和检测方法均有效。'),
]

pdfmetrics.registerFont(TTFont('Chinese', '/System/Library/Fonts/Supplemental/Arial Unicode.ttf'))
body = ParagraphStyle('body', fontName='Chinese', fontSize=11, leading=19, wordWrap='CJK', textColor=colors.HexColor('#17202A'), spaceAfter=14)
title = ParagraphStyle('title', parent=body, fontSize=16, leading=23, spaceAfter=8)
small = ParagraphStyle('small', parent=body, fontSize=9, leading=14, textColor=colors.HexColor('#52606D'), spaceAfter=13)
story = [Paragraph('《Taught Well Learned Ill》论文阅读报告', title), Paragraph('Chen Y, Li B, Yuan Y, et al. · NeurIPS 2025 · 第 38 卷', small)]
for i,(q,a) in enumerate(items):
    if i == 3:
        story.append(PageBreak())
    story.append(Paragraph(html.escape(q) + '<br/>' + html.escape(a), body))
story.append(Paragraph('依据：论文正文第 1—5 节、表 1—3，以及附录 A、B、C、H、K。<br/>来源：NeurIPS 官方会议论文 PDF（含附录，共 35 页）。', small))

def footer(c, doc):
    c.setFont('Chinese', 9)
    c.setFillColor(colors.HexColor('#64748B'))
    c.drawCentredString(A4[0]/2, 28, f'{doc.page} / 2')

target=out/'SCAR_论文阅读报告_两页A4.pdf'
doc=SimpleDocTemplate(str(target), pagesize=A4, rightMargin=48, leftMargin=48, topMargin=42, bottomMargin=43, title='Taught Well Learned Ill 论文阅读报告', author='')
doc.build(story, onFirstPage=footer, onLaterPages=footer)
assert len(PdfReader(target).pages)==2
md = '# 《Taught Well Learned Ill》论文阅读报告\n\n'
md += '> 官方会议记录：NeurIPS 2025，第38卷。用户提供的2026年与官方记录不一致。\n\n'
md += '\n\n'.join('**'+q+'** '+a for q,a in items)
(out/'SCAR_论文阅读报告.md').write_text(md+'\n',encoding='utf-8')
print(f'Created {target}; 2 A4 pages; answer characters: {sum(len(a) for _,a in items)}')
