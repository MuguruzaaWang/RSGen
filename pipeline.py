from Summarizationpipeline import Summarizationpipeline
text_path = [r'图像对象定位可提供准确的对象区域,有效提高图像对象识别和分类准确率.基于此,文中提出基于多重图像分割评价的图像对象定位方法.通过图像的多层次分割,确定图像不同区域之间的语义约束关系,应用此约束关系对不同层次的对象区域模式进频繁项集挖掘和评分,并按照此模式评分逐次合并每层图像分割中的重要区域,最终实现整个对象区域的精确定位.MSRC和GRAZ的定位实验表明,文中方法可有效定位图像的前景目标,在Caltech图像目标分类实验中也证明文中方法的有效性.', r'三维主动外观模型将肺区的三维外观矩阵转化为一维向量时,原三维灰度分布受到破坏,分割精确度受到影响,且生成过大向量,影响分割效率.基于此,文中提出张量模式的三维主动外观模型,旨在借助高维奇异值分法直接处理肺区的三维外观矩阵,从而避免其向一维向量的转换.首先在张量理论基础上建立主动外观模型并推导参数;然后设计分块Kronecker方法确定外观张量低秩表示模式的最佳方案']
vocab_path = r'/data/run01/scv0028/wpc/survey_generation/HeterEnReTSumGraph_ABS_ablation_ForJZQ/dataset/vocab'
pipeline = Summarizationpipeline(text_path,vocab_path)
model_path = r'/data/run01/scv0028/wpc/survey_generation/HeterEnReTSumGraph_ABS_ablation_ForJZQ/trained_model/model_step_100.pt'
output = pipeline(max_length=200,min_length=100,model_path=model_path)
print(output)