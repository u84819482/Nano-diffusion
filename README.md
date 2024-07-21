# Nano-diffusion 
Minimal DDPM/DiT-based generation of MNIST digits. Heavily commented, self-contained script with all non-essential complexities stripped. 

Added an extra coefficient into denoising function to adjust the strength of extra noise term, enabling to tradeoff the variety of generated samples with their quality.


High variety / low quality:  

<img src="https://github.com/user-attachments/assets/64a3f320-d394-47ca-a6bd-558af9b0c80c" width="800" /> 



No variety / high quality: 

<img src="https://github.com/user-attachments/assets/a8ca296a-16fb-4378-b20d-2757cabb28a7" width="800" />



Decent variety / decent quality:

<img src="https://github.com/user-attachments/assets/132bf1a9-14b3-45c0-bc82-ff47af14ec73" width="800" />


References:
- https://arxiv.org/abs/2006.11239
- https://arxiv.org/abs/2212.09748
