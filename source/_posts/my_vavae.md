---
title: 视觉表征漫谈2——VFM Tokenizer Design Space for Image Generation
date: 2026-03-02
tags:
  - 生成模型
  - 视觉表征
  - Diffusion
  - VAE
categories:
  - 科研随笔
index_img: /new_pic/stage2.png
math: true
---

上一次我们聊到了 VAVAE[^1] 和 RAE[^2] 类工作的困境，以及可能的解法。
![Scale Up](../../../../img/scale.jpg)
这种解法可能是 DINOv4 探索的方案——一个在训练 VFM 的时候就能够良好保持住结构等信息的、模型参数量大的、数据更 diverse 的（比如见过很多字，这样高频信息的 gap 似乎有机会弥补），这样训练出来的 tokenizer 既保持了语义，又具备了很好的重建能力，但似乎是很久之后才会做到的事情。

## 有没有真的很优雅的解法？
![layerdiffuse的pipeline](../../../../new_pic/pipe1.png)
### 图像水印学说--将图像"藏"进噪声
![face2ramen,将人脸的image藏入拉面的image](../../../../new_pic/face2ramen.png)
很久之前读过一篇工作——**LayerDiffuse[^3]**（2402.17113）。它要解决的问题很具体：如何让已经训练好的 Stable Diffusion 直接生成带透明度（alpha channel）的图像，而不破坏原有的生成质量。

在这篇文章的 Related Work 2.1 节中，作者回顾了一个在多个领域都被反复验证的现象：**神经网络可以将一种信息"藏"在另一种信息的扰动里，且不改变整体的特征分布**。CycleGAN[^4] 最早展示了这件事——face-to-ramen 实验里，人脸的身份信息可以在视觉上完全隐匿于一碗拉面的图片之中；可逆下采样和可逆灰度化的工作进一步证明，一张完整的大图可以被编码进一张更小的图而不损失任何信息；Goodfellow 的对抗样本研究则从另一个侧面佐证了这一点——人眼不可见的微小扰动可以携带足以"欺骗"神经网络的完整语义信号。

LayerDiffuse[^3] 把这个原理直接用在了 latent space 上：将 alpha channel 信息编码为一个幅度受约束的 latent offset，注入到 SD 的 latent 中，同时用分布对齐约束确保 latent 的统计特性不变。这样，原来对透明度一无所知的扩散模型，可以在几乎不改变原始 latent 空间结构的前提下，"感知"到被隐藏进去的透明度语义，而已有的 ControlNet、LoRA 等生态也可以无缝复用。

**这个想法是否可以平行地迁移到 VFM tokenizer 的困境上？**
![我的方案的stage1](../../../../new_pic/stage1.png)
![我的方案的stage2](../../../../new_pic/stage2.png)
### DINO 的语义是否可以被"藏"进VAE latent中？

SVG[^5] 和 RAE[^6] 给出的方案，本质上都是重建和生成的trade-off——要么拼接 DINO 特征，要么直接替换 VAE。这两种方式都不可避免地改变了 latent 的维度和分布，下游的 DiT 必须针对新 latent 从头训练，已有的 SD 生态也很难复用。此外，现在的方案如果语义有大幅提升，重建性能必然大幅缩水；反之亦然。

但如果借鉴 LayerDiffuse 的思路，问题就可以换一种方式提出：能否学到一个轻量的映射 $f$，把 DINO 的语义以 offset 的形式"写入" VAE latent，使得修正后的 latent 在统计分布上和原始 latent 无异，但实际上携带了更丰富的语义结构，甚至语义含量与VFM本身无异？

$$
z_{\text{sem}} = z_{\text{VAE}} + f\!\left(F_{\text{DINO}}\right)
$$

该方案需要同时满足三重约束：latent 的分布不能漂移（否则已有的 DiT 会失效），VAE decoder 的重建质量不能下降，同时修正后的 latent 对 DINO 特征的语义相似度要显著高于原始 latent。

如果这种特征隐写可以成功，那么：已有的 DiT 可以直接在新的 latent 上 fine-tune，无需改架构，无需重新训练 tokenizer，重建性能与VAE相当，rfid指标更好。此外，因为 latent 里天然携带了语义结构，训练收敛也会更快，最终的语义含量更高，gfid指标更好。

### 理论上为什么这件事可能是对的？

LayerDiffuse[^3] 的分布对齐约束提供了一个直接可用的ref -- 把 offset 的均值和方差强制对齐到原始 latent 的统计量上：

$$
\Delta z_{\text{aligned}} = \frac{\Delta z - \mu(\Delta z)}{\sigma(\Delta z)} \cdot \sigma(z_{\text{VAE}}) + \mu(z_{\text{VAE}})
$$

SVG[^5] 的 distribution alignment 做的是完全相同的事，只是应用在了不同的地方。在合适的约束下，信息可以被完整地编码进一个分布不变的扰动中，并且可以被准确还原。我们想做的事情，在 INN 的框架下，是被证明可行的。

### A future design space to be continued...

这个方向我在实习后期花了相当长的时间尝试，最终没有训出一个收敛的版本。

实验时，最核心的难点是 offset 幅度的控制和如何让decoder感知到这个微小的$\Delta z$。$\Delta z$ 太小，语义信息实际上没有被有效写入，DiT 感知不到任何差异；太大，latent 的分布就会发生漂移，重建质量和 DiT 兼容性同时下降。这个工作点的调节极其敏感，而且会随着训练动态漂移，没有找到稳定的解。其次是多目标 loss 的梯度冲突——分布对齐、语义对齐、重建三路 loss 同时优化时，梯度方向经常相互抵消，训练曲线极不稳定，很难判断模型究竟在往哪个方向收敛。此外，如何让decoder感知到这个微小的$\Delta z$ 并在后续的架构中设计放大，准确解码出来，也是一个巨大的挑战。


---
## 参考文献
[^1]: Jingfeng Yao et al. "Reconstruction vs. Generation: Taming Optimization Dilemma in Latent Diffusion Models." *CVPR 2025*. arXiv: 2501.01423
[^2]: Sihyun Yu et al. "Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think." *ICLR 2025*. arXiv: 2410.06940
[^3]: Zhang, Lvmin, and Maneesh Agrawala. "Transparent image layer diffusion using latent transparency." arXiv preprint arXiv:2402.17113 (2024).
[^4]: Zhu, Jun-Yan, et al. "Unpaired image-to-image translation using cycle-consistent adversarial networks." Proceedings of the IEEE international conference on computer vision. 2017.
[^5]: Minglei Shi et al. "Latent Diffusion Model without Variational Autoencoder." *ICLR 2026*. arXiv: 2510.15301
[^6]: Boyang Zheng et al. "Diffusion Transformers with Representation Autoencoders." *ICLR 2026*. arXiv: 2510.11690