
# Vulkan 开发系统性学习教程

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/githubhaohao/NDK_OpenGLES_3_0/blob/master/LICENSE.txt)
![Build](https://img.shields.io/badge/build-passing-brightgreen)

第一阶段的更新覆盖了 Vulkan 的大部分概念和知识点，众所周知，Vulkan 编程的代码量相对于 OpenGL 多了一个数量级（不用害怕，后面Vulkan封装一下，用起来也会非常简洁）。

这一系列文章都在避免一上去就讲一大堆代码（精简了很多没必要展示的代码），追求一篇文章只讲一个概念，奉行概念先行。

这样，Vulkan 的概念掌握的差不多了，我们再去看代码,  这样很容易把握住整体代码逻辑，不至于一上来全是他喵的结构体，眼花缭乱，看着费劲。

第二阶段的更新，重点是在编程实战，前提是对 Vulkan 进行封装，避免上来贴一大坨代码，文章大致结构以代码讲解 + demo 展示为主，然后顺便回顾前面的知识点。

## Vulkan 入门系列文章（理论篇）

1. [开篇，Vulkan 概述](http://mp.weixin.qq.com/s?__biz=MzIwNTIwMzAzNg==&mid=2654177035&idx=1&sn=48ab8877a7ae1620845dc63b2e7cb070&chksm=8cf35438bb84dd2e919d288deaa06f16580e5cb339d2be2ec606fbf772377dd02bc111df5e34&scene=21#wechat_redirect)
2. [Vulkan 实例 Instance](https://mp.weixin.qq.com/s?__biz=Mzg2NDc1OTIzOQ==&mid=2247483992&idx=1&sn=d257569320798c4513752271abc8a77c&scene=21#wechat_redirect)
3. [Vulkan 物理设备](https://mp.weixin.qq.com/s?__biz=Mzg2NDc1OTIzOQ==&mid=2247484002&idx=1&sn=fc4e1bacb8485b71bef29506764b6a54&scene=21#wechat_redirect)
4. [Vulkan 设备队列](https://mp.weixin.qq.com/s?__biz=Mzg2NDc1OTIzOQ==&mid=2247484010&idx=1&sn=6613d788307659ee1552a309f5d5a48e&scene=21#wechat_redirect)
5. [Vulkan 逻辑设备](https://mp.weixin.qq.com/s?__biz=Mzg2NDc1OTIzOQ==&mid=2247484024&idx=1&sn=86bf2a35039f418c41fe7b89f0664824&scene=21#wechat_redirect)
6. [Vulkan 内存管理](https://mp.weixin.qq.com/s?__biz=Mzg2NDc1OTIzOQ==&mid=2247484078&idx=1&sn=db5fea6f4847368b0bc10d9f8081850d&scene=21#wechat_redirect)
7. [Vulkan 缓存](https://mp.weixin.qq.com/s?__biz=Mzg2NDc1OTIzOQ==&mid=2247484045&idx=1&sn=9b46909e161dd7ea46a80223b784aa43&scene=21#wechat_redirect)
8. [Vulkan 图像](https://mp.weixin.qq.com/s?__biz=Mzg2NDc1OTIzOQ==&mid=2247484057&idx=1&sn=6e82e29bd2487495f856f57fb8facdbc&scene=21#wechat_redirect)

9. [Vulkan 图像视图](https://mp.weixin.qq.com/s?__biz=Mzg2NDc1OTIzOQ==&mid=2247484081&idx=1&sn=d569058f15352b094000a2a67dc8f2be&scene=21#wechat_redirect)

10. [Vulkan 窗口表面（Surface）](https://mp.weixin.qq.com/s?__biz=Mzg2NDc1OTIzOQ==&mid=2247484096&idx=1&sn=1d8e5e37927a9294835de2b7086b9cac&scene=21#wechat_redirect)

11. [Vulkan 交换链](https://mp.weixin.qq.com/s?__biz=Mzg2NDc1OTIzOQ==&mid=2247484100&idx=1&sn=3293d353c3f27f914fc57ea2a32e47dd&scene=21#wechat_redirect)

12. [Vulkan 渲染通道](https://mp.weixin.qq.com/s?__biz=Mzg2NDc1OTIzOQ==&mid=2247484102&idx=1&sn=13570a8fa2a1a142041eba8a7de3c7a6&scene=21#wechat_redirect)

13. [Vulkan 帧缓冲区（FrameBuffer）](https://mp.weixin.qq.com/s?__biz=Mzg2NDc1OTIzOQ==&mid=2247484104&idx=1&sn=4c0f709c30f215d96f68a45bb59c9fbe&scene=21#wechat_redirect)

14. [Vulkan 图形管线](https://mp.weixin.qq.com/s?__biz=Mzg2NDc1OTIzOQ==&mid=2247484106&idx=1&sn=8ee3a34998635041822beb9d52dcea98&scene=21#wechat_redirect)

15. [Vulkan 着色器](https://mp.weixin.qq.com/s?__biz=Mzg2NDc1OTIzOQ==&mid=2247484120&idx=1&sn=c14e3390020281e22eac9bc9f1dbe5a2&scene=21#wechat_redirect)

16. [Vukan 描述符集](https://mp.weixin.qq.com/s?__biz=Mzg2NDc1OTIzOQ==&mid=2247484131&idx=1&sn=fa06b8b700151df47876dd26aa6a900b&scene=21#wechat_redirect)

17. [Vulkan 指令缓存](https://mp.weixin.qq.com/s?__biz=Mzg2NDc1OTIzOQ==&mid=2247484151&idx=1&sn=103fd546f056d02f6563f7ea78f8aa7c&scene=21#wechat_redirect)

18. [Vulkan 同步机制](https://mp.weixin.qq.com/s?__biz=Mzg2NDc1OTIzOQ==&mid=2247484159&idx=1&sn=1afd8c236a7b7a9a74ee42447ed4889f&scene=21#wechat_redirect)

## Vulkan 入门系列文章（实践篇）

1. [Vulkan 绘制一个三角形需要一千行代码，是真的吗？](https://mp.weixin.qq.com/s/DTLxnaDVD55_Y5Y31wT3dg)
2. [Vulkan 绘制第一个三角形](https://mp.weixin.qq.com/s/pRIr0UpNNbK1nvJvXV3JQA)

## 相关推荐
- [OpenGL ES 3.0 从入门到精通系统性学习教程](https://github.com/githubhaohao/NDK_OpenGLES_3_0)
- [Android OpenGL Camera 2.0 实现 30 种滤镜和抖音特效](https://github.com/githubhaohao/OpenGLCamera2)
- [Android FFmpeg 音视频开发教程](http://mp.weixin.qq.com/s?__biz=MzIwNTIwMzAzNg==&mid=506681298&idx=1&sn=50177285bf0d330d0dfc4e0954d5ad12&chksm=0cf384e13b840df76f89aeb8ac76939ff32b2f9bf600729782d61698181af60d92cce61ee150#rd)


## 联系交流
有疑问或技术交流可以扫码添加**我的微信：Byte-Flow ，拉你入相关技术交流群**，里面很多牛人帮你解答。

![字节流动](https://github.com/githubhaohao/NDK_OpenGLES_3_0/blob/master/doc/img/accountID.jpg)

## 付费社群
项目疑难问题解答、大厂内部推荐、面试指导、简历指导、代码指导、offer 选择建议、学习路线规划。

一个人可以走的很快，但是一群人可以走得更远。如果你需要一个良好的学习环境，可以加入我的知识星球。

知识星球加入规则做了调整，按照实际加入日期算一年，不再按照一期一期地算了，有需要的同学快来加入吧！

知识星球权益：
音视频、OpenGL ES、Vulkan 、Metal、图像滤镜、视频特效及相关渲染技术问题解答，面试指导，1v1 简历服务，职业规划。

附加权益: 字节流动所有付费文章全部免费，后续录制的视频教程免费。

![知识星球](https://github.com/githubhaohao/NDK_OpenGLES_3_0/blob/master/doc/img/zsxq.jpeg?raw=true)

