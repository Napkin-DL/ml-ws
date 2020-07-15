# image_segmentation

<p> tensorflow 2.x 기반의 tensorflow 공식 홈페이지에서 제공하는 <strong><a href="https://www.tensorflow.org/tutorials/images/segmentation" target="_blank" class ='btn-default'>image segmentation</a></strong>을 AWS의 SageMaker에서 활용할 수 있도록 변경한 코드입니다. </p>
<p> 기본적인 아키텍처 구조는 <strong><a href="https://arxiv.org/abs/1505.04597" target="_blank" class ='btn-default'>U-Net</a></strong> 로서 pixel-wise binary classification을 수행하게 됩니다. </p>
<p> 최종 결과는 Accuracy와 Loss 함수를 볼 수 있으며 특정 epoch마다 저장된 mask 결과 값을 함께 볼 수 있도록 구현하였으며, inference에 대해 endpoint에서 추론할 수 있는 방법을 담고 있습니다. 추가적으로 현재 serving 환경에 없는 opencv를 사용할 수 있도록 custom docker container를 만들 수 있는 가이드도 함께 추가하였습니다.</p>


