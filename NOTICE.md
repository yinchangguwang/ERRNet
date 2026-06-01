#### 简单写个文档（洪燊）

我具体修改的代码可以在commit里看一下，主要就是加了`models/arch/cascade_errnet.py`这个文件，然后修改了`models/arch/__init__.py`和`models/errnet_model.py`，你们可以试一下能不能跑通。

主要是按照《Single Image Reflection Removal Through Cascaded Refinement（IBCLN）》CVPR2020，引入了级联网络，残差学习，深度监督。

训练baseline用pj的md文档给的命令，这个改进算法版的加个参数`--inet cascade_errnet`应该就行了。

如果不要深度监督就去`models/arch/cascade_errnet.py`把最后的`return t1, t2, t3`改成`return t3`，剩下的应该不用改能支持。

`models/errnet_model.py`的292~316里面这几个乘的系数可能可以调一下。

如果你们还没看这个项目的代码，那大概就去主要看一下`models/arch`里面的，还有`models/`的`base_model.py`和`errnet_model.py`什么的，命令行参数看`options/`下面的几个文件。